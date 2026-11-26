import os, sys, json
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import adaptation as ad
from simulator import restoring_force, init_state, run

SZ = {"train": 8, "val": 4, "test": 4}
NS = 6


def tiny(**kw):
    a = dict(budgets=(2, 4), epochs=1, hidden=8, batch_size=32, sizes=SZ, n_steps=NS)
    a.update(kw)
    return ad.run_seed(0, **a)


def test_custom_k_values_in_data_and_record():
    r = tiny(k_old=1.0, k_new=5.0)
    assert (r["k_old"], r["k_new"]) == (1.0, 5.0)
    plan = ad.seed_plan(0)
    new = ad.make_world(5.0, plan["new"], SZ, NS)
    old = ad.make_world(2.0, plan["old"], SZ, NS)
    assert np.allclose(new["test"]["acc"], -5.0 * new["test"]["cur"][:, :2])
    assert np.allclose(old["test"]["acc"], -2.0 * old["test"]["cur"][:, :2])
    assert ad.budget_transitions(new["train"], 4, 5.0)["acc"].shape == (4 * NS, 2)


def test_custom_center():
    c = (0.5, -0.3)
    w = ad.make_world(3.0, ad.seed_plan(0)["new"], SZ, NS, center=c)
    assert np.allclose(w["test"]["acc"], -3.0 * (w["test"]["cur"][:, :2] - np.array(c)))
    d = ad.budget_transitions(w["train"], 4, 3.0, c)
    assert np.allclose(d["acc"], -3.0 * (d["cur"][:, :2] - np.array(c)))
    assert ad.force_for(3.0, c)(np.array([[0.5, -0.3, 1, 1]])).sum() == 0
    r = tiny(k_new=3.0, center=c)       # oracle is exact under the custom centre
    assert r["oracle"]["new_test"]["accel_mse"] == 0 and r["oracle"]["new_test"]["next_state_mse"] < 1e-24


def test_non_default_dt_propagates_to_data_and_scoring():
    dt = 0.05
    w = ad.make_world(4.0, ad.seed_plan(0)["new"], SZ, NS, dt)
    f = ad.force_for(4.0)
    ref = run(init_state(SZ["test"], ad.seed_plan(0)["new"]["test"]), f, dt, NS)
    assert np.array_equal(w["test"]["traj"], ref)
    assert np.allclose(w["test"]["nxt"][:, 2:], w["test"]["cur"][:, 2:] + w["test"]["acc"] * dt)
    r = tiny(k_new=4.0, dt=dt)           # oracle next-state error only ~0 if scoring uses the same dt
    for part in (r["oracle"]["new_test"],):
        assert part["next_state_mse"] < 1e-24
    r_wrong = ad.score(w["test"], w["test"]["acc"], 0.01)   # mismatched dt is detectable
    assert r_wrong["next_state_mse"] > 1e-12


def test_custom_sizes_and_budgets():
    sz = {"train": 6, "val": 3, "test": 2}
    r = tiny(sizes=sz, budgets=(1, 3, 6), n_steps=5)
    assert list(r["budgets"]) == ["1", "3", "6"]
    assert [r["budgets"][b]["n_trans"] for b in r["budgets"]] == [5, 15, 30]
    assert r["pretrain"]["n_traj"] == 6 and r["pretrain"]["n_trans"] == 30


def test_custom_seed_reproducible_and_distinct():
    a = ad.run_seed(3, budgets=(2,), epochs=1, hidden=8, batch_size=32, sizes=SZ, n_steps=NS, k_new=4.0)
    b = ad.run_seed(3, budgets=(2,), epochs=1, hidden=8, batch_size=32, sizes=SZ, n_steps=NS, k_new=4.0)
    c = ad.run_seed(4, budgets=(2,), epochs=1, hidden=8, batch_size=32, sizes=SZ, n_steps=NS, k_new=4.0)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


@pytest.mark.parametrize("kw", [
    dict(k_old=0.0), dict(k_new=-1.0), dict(k_new=float("nan")), dict(k_old=float("inf")),
    dict(dt=0.0), dict(dt=-0.01), dict(dt=float("nan")), dict(n_steps=0), dict(n_steps=2.5),
    dict(center=(0.0,)), dict(center=(0.0, float("inf"))),
    dict(sizes={"train": 0, "val": 4, "test": 4}), dict(budgets=()), dict(budgets=(0, 2)),
    dict(budgets=(2, 2)), dict(budgets=(9,)),
])
def test_invalid_params_rejected(kw):
    with pytest.raises(ValueError):
        ad.validate_params(**{**dict(sizes=SZ, budgets=(2, 4)), **kw})


def test_run_seed_validates_before_work():
    with pytest.raises(ValueError):
        tiny(budgets=(99,))


def test_invalid_seeds_rejected():
    for bad in ([], [0, 0], [-1]):
        with pytest.raises(ValueError):
            ad.validate_params(seeds=bad)


def test_cli_invalid_args_exit(tmp_path):
    for argv in (["--k-new", "0"], ["--dt", "-1"], ["--n-steps", "0"], ["--budgets", "500"],
                 ["--seeds", "0", "--repetitions", "2"], ["--repetitions", "0"]):
        with pytest.raises(SystemExit):
            ad.main(argv + ["--out", str(tmp_path)])
    assert not os.listdir(tmp_path)


def test_cli_custom_run_writes_actual_params(tmp_path):
    m = ad.main(["--repetitions", "1", "--epochs", "1", "--hidden", "8", "--budgets", "2", "4",
                 "--n-train", "4", "--n-val", "2", "--n-test", "2", "--n-steps", "5",
                 "--k-old", "1", "--k-new", "5", "--center-x", "0.5", "--center-y", "-0.2",
                 "--dt", "0.02", "--out", str(tmp_path)])
    c = m["config"]
    assert (c["k_new"], c["center"], c["dt"], c["n_steps"], c["seeds"]) == (5.0, (0.5, -0.2), 0.02, 5, [0])
    assert c["split_sizes"] == {"train": 4, "val": 2, "test": 2}
    assert json.load(open(tmp_path / "metrics.json"))["config"]["k_new"] == 5.0
    assert (tmp_path / "adaptation_curve.png").stat().st_size > 0


def test_plot_labels_reflect_parameters():
    import plot_adaptation as pa
    info = {"k_old": 2.0, "k_new": 5.5, "n_steps": 17}
    assert "17 transitions" in pa.traj_x(info) and "200" not in pa.traj_x(info)
    assert pa.traj_x(pa.DEFAULT_INFO) == pa.TRAJ_X


def test_backward_compatible_defaults_and_signatures():
    assert (ad.K_OLD, ad.K_NEW, ad.CENTER, ad.DT, ad.N_STEPS) == (1.0, 9.0, (0.0, 0.0), 0.01, 200)
    assert ad.BUDGETS == (8, 16, 32, 64, 128) and ad.SIZES == {"train": 128, "val": 32, "test": 32}
    ad.validate_params()                                   # defaults valid
    explicit = tiny()
    defaults = tiny(k_old=ad.K_OLD, k_new=ad.K_NEW, center=ad.CENTER, dt=ad.DT)
    assert json.dumps(explicit, sort_keys=True) == json.dumps(defaults, sort_keys=True)
    assert np.allclose(ad.force_for(9.0)(np.array([[1., 2, 0, 0]])), -9.0 * np.array([[1., 2.]]))