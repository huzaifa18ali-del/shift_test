import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import adaptation as ad
from evaluate import relative_improvement, mse, mae
from train import TrainConfig, Normalizer, train_mlp, params_fingerprint
from models import MLP

SZ = {"train": 16, "val": 4, "test": 4}
NS = 10


def small_run(seed=0):
    return ad.run_seed(seed, budgets=(4, 8), epochs=1, hidden=8, batch_size=32, sizes=SZ, n_steps=NS)


def test_new_world_uses_k_new():
    w = ad.make_world(ad.K_NEW, ad.seed_plan(0)["new"], SZ, NS)
    for s in w.values():
        assert np.allclose(s["acc"], -9.0 * s["cur"][:, :2])


def test_splits_disjoint_and_seeds_distinct():
    old, new = (ad.make_world(k, ad.seed_plan(0)[w], SZ, NS) for k, w in ((ad.K_OLD, "old"), (ad.K_NEW, "new")))
    ad.assert_disjoint(old, new)
    seeds = [v for s in (0, 1, 2) for w in ad.seed_plan(s).values() for v in w.values()]
    assert len(set(seeds)) == len(seeds)
    try:
        ad.assert_disjoint(old, {"train": new["train"], "val": old["val"], "test": new["test"]})
        assert False, "overlap not detected"
    except AssertionError as e:
        assert "overlap" in str(e)


def test_reproducible_data_and_run():
    a = ad.make_world(ad.K_NEW, ad.seed_plan(1)["new"], SZ, NS)
    b = ad.make_world(ad.K_NEW, ad.seed_plan(1)["new"], SZ, NS)
    assert all(np.array_equal(a[k]["traj"], b[k]["traj"]) for k in a)
    assert json.dumps(small_run(), sort_keys=True) == json.dumps(small_run(), sort_keys=True)


def test_budget_counts():
    w = ad.make_world(ad.K_NEW, ad.seed_plan(0)["new"], SZ, NS)
    for b in (4, 8):
        d = ad.budget_transitions(w["train"], b)
        assert d["traj"].shape[1] == b and d["cur"].shape[0] == b * NS and d["acc"].shape == (b * NS, 2)
    assert np.array_equal(ad.budget_transitions(w["train"], 4)["traj"], ad.budget_transitions(w["train"], 8)["traj"][:, :4])
    r = small_run()
    assert [r["budgets"][k]["n_traj"] for k in ("4", "8")] == [4, 8]
    assert [r["budgets"][k]["n_trans"] for k in ("4", "8")] == [40, 80]


def _tiny_setup():
    w = ad.make_world(ad.K_NEW, ad.seed_plan(0)["new"], SZ, NS)
    xn, yn = Normalizer().fit(w["train"]["cur"]), Normalizer().fit(w["train"]["acc"])
    import torch
    torch.manual_seed(0)
    return w, xn, yn, MLP(4, 8, 2)


def test_finetune_updates_params_and_never_mutates_original():
    w, xn, yn, base = _tiny_setup()
    fp0 = params_fingerprint(base)
    cfg = TrainConfig(hidden=8, batch_size=16, epochs=2, seed=1)
    m, h = train_mlp(w["train"]["cur"], w["train"]["acc"], w["val"]["cur"], w["val"]["acc"], cfg, xn, yn, init_model=base)
    assert params_fingerprint(base) == fp0               # original untouched
    assert params_fingerprint(m) != fp0                  # adapted copy changed
    assert h["init_fingerprint"] == fp0


def test_every_budget_starts_from_same_checkpoint():
    r = small_run()
    fps = {e["finetune"]["init_fingerprint"] for e in r["budgets"].values()}
    assert fps == {r["pretrain"]["checkpoint_fingerprint"]}
    fresh = {e["fresh"]["init_fingerprint"] for e in r["budgets"].values()}
    assert len(fresh) == 1 and fresh != fps              # fresh: common random init, not the checkpoint


def test_selection_uses_validation_only():
    w, xn, yn, base = _tiny_setup()
    cfg = TrainConfig(hidden=8, batch_size=16, epochs=4, seed=1)
    _, h = train_mlp(w["train"]["cur"], w["train"]["acc"], w["val"]["cur"], w["val"]["acc"], cfg, xn, yn,
                     init_model=base, select_best_val=True)
    assert h["best_epoch"] == int(np.argmin(h["val_loss"])) + 1


def test_metrics_known_values_and_improvement():
    assert np.isclose(mse(np.array([1., 3.]), np.array([0., 1.])), 2.5) and np.isclose(mae(np.array([1., 3.]), np.array([0., 1.])), 1.5)
    assert np.isclose(relative_improvement(4.0, 1.0), 0.75)
    assert relative_improvement(4.0, 8.0) == -1.0
    assert relative_improvement(0.0, 1.0) is None


def test_oracle_zero_error():
    w = ad.make_world(ad.K_NEW, ad.seed_plan(0)["new"], SZ, NS)
    m = ad.score(w["test"], ad.force_for(ad.K_NEW)(w["test"]["cur"]))
    assert m["accel_mse"] == 0 and m["next_state_mse"] < 1e-24


def test_baseline_apis_still_work():
    import experiment
    d = experiment.make_splits()
    assert d["train"]["cur"].shape == (128 * 200, 4)
    w, xn, yn, _ = _tiny_setup()
    cfg = TrainConfig(hidden=8, batch_size=16, epochs=2, seed=0)
    m, h = train_mlp(w["train"]["cur"], w["train"]["acc"], w["val"]["cur"], w["val"]["acc"], cfg, xn, yn)
    assert h["best_epoch"] == 2 and len(h["val_loss"]) == 2   # default = final-epoch, no selection