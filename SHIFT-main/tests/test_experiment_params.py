import os, sys, json
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import experiment as ex
from train import TrainConfig

SZ = {"train": 6, "val": 3, "test": 3}
CFG = TrainConfig(hidden=8, lr=1e-2, batch_size=32, epochs=2, seed=0)
KW = dict(sizes=SZ, n_steps=5)


def test_custom_k_centre_dt_in_data():
    phys = {"k": 5.0, "center_x": 0.4, "center_y": -0.3}
    d = ex.make_splits(sizes=SZ, n_steps=5, dt=0.02, phys=phys)
    for s in d.values():
        assert np.allclose(s["acc"], -5.0 * (s["cur"][:, :2] - np.array([0.4, -0.3])))
        assert np.allclose(s["nxt"][:, 2:], s["cur"][:, 2:] + s["acc"] * 0.02)


def test_custom_sizes_and_steps_shapes():
    d = ex.make_splits(sizes={"train": 7, "val": 2, "test": 3}, n_steps=4)
    assert d["train"]["cur"].shape == (28, 4) and d["val"]["traj"].shape == (5, 2, 4)
    assert d["test"]["acc"].shape == (12, 2)


def test_custom_data_seeds_reproducible_and_distinct():
    a = ex.make_splits(seeds={"train": 10, "val": 11, "test": 12}, **KW)
    b = ex.make_splits(seeds={"train": 10, "val": 11, "test": 12}, **KW)
    c = ex.make_splits(seeds={"train": 20, "val": 21, "test": 22}, **KW)
    assert all(np.array_equal(a[k]["traj"], b[k]["traj"]) for k in a)
    assert not np.array_equal(a["train"]["traj"], c["train"]["traj"])


def test_leakage_check_detects_overlap_with_any_sizes():
    d = ex.make_splits(**KW)
    ex.check_disjoint(d)
    d["val"]["traj"] = d["train"]["traj"][:, 2:4].copy()      # different shape than train, shared states
    with pytest.raises(AssertionError, match="overlap"):
        ex.check_disjoint(d)


def test_run_custom_records_actual_parameters(tmp_path):
    phys = {"k": 5.0, "center_x": 0.4, "center_y": -0.3}
    seeds = {"train": 5, "val": 6, "test": 7}
    r = ex.run_experiment(TrainConfig(8, 1e-2, 16, 2, 3), str(tmp_path), seeds, SZ, 5, 0.02, phys)
    assert (r["dt"], r["n_steps"], r["physics"], r["data_seeds"], r["trajectories_per_split"]) == (0.02, 5, phys, seeds, SZ)
    assert r["mlp_config"] == {"hidden": 8, "lr": 1e-2, "batch_size": 16, "epochs": 2, "seed": 3, "device": "cpu"}
    m = r["metrics"]
    orc = m["analytical_oracle (simulator reference, not a learned model)"]["test"]
    assert orc["accel_mse"] == 0 and orc["next_state_mse"] < 1e-24      # scoring uses the custom dt
    assert m["linear_regression"]["test"]["accel_mse"] < 1e-18          # affine law, fitted exactly
    assert (tmp_path / "training_curves.png").stat().st_size > 0
    assert (tmp_path / "prediction_comparison.png").stat().st_size > 0
    assert json.load(open(tmp_path / "metrics.json"))["physics"] == phys


def test_json_structure_matches_committed_baseline(tmp_path):
    r = ex.run_experiment(CFG, str(tmp_path), sizes=SZ, n_steps=5)
    ref = json.load(open(os.path.join(os.path.dirname(__file__), "..", "outputs", "baseline", "metrics.json")))
    def keys(x): return {k: keys(v) for k, v in x.items()} if isinstance(x, dict) else None
    r2 = json.loads(json.dumps(r))
    r2["versions"] = ref["versions"]
    assert keys(r2) == keys(ref)


def test_hyperparameters_affect_training_and_run_reproducible(tmp_path):
    a = ex.run_experiment(CFG, str(tmp_path / "a"), **KW)
    b = ex.run_experiment(CFG, str(tmp_path / "b"), **KW)
    c = ex.run_experiment(TrainConfig(8, 1e-2, 32, 2, 1), str(tmp_path / "c"), **KW)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["final_train_loss_normalised"] != c["final_train_loss_normalised"]


@pytest.mark.parametrize("kw", [
    dict(phys={"k": 0.0, "center_x": 0, "center_y": 0}), dict(phys={"k": float("nan"), "center_x": 0, "center_y": 0}),
    dict(phys={"k": 1.0, "center_x": float("inf"), "center_y": 0}),
    dict(dt=0.0), dict(dt=-1.0), dict(n_steps=0), dict(sizes={"train": 0, "val": 1, "test": 1}),
    dict(seeds={"train": 1, "val": 1, "test": 2}), dict(seeds={"train": -1, "val": 1, "test": 2}),
    dict(cfg=TrainConfig(hidden=0)), dict(cfg=TrainConfig(lr=0.0)), dict(cfg=TrainConfig(batch_size=0)),
    dict(cfg=TrainConfig(epochs=0)), dict(cfg=TrainConfig(seed=-1)),
])
def test_invalid_params_rejected(kw):
    with pytest.raises(ValueError):
        ex.validate_params(**kw)


def test_run_experiment_validates_first(tmp_path):
    with pytest.raises(ValueError):
        ex.run_experiment(CFG, str(tmp_path), dt=-1.0, **KW)
    assert not os.listdir(tmp_path)


def test_cli_invalid_and_overwrite_protection(tmp_path):
    for argv in (["--k", "0"], ["--dt", "-1"], ["--n-steps", "0"], ["--seed-val", "0"], ["--epochs", "0"]):
        with pytest.raises(SystemExit):
            ex.main(argv + ["--out", str(tmp_path / "x")])
    assert not (tmp_path / "x").exists()
    (tmp_path / "metrics.json").write_text("{}")
    with pytest.raises(SystemExit):                          # non-default run, existing results
        ex.main(["--k", "5", "--out", str(tmp_path)])
    assert (tmp_path / "metrics.json").read_text() == "{}"
    ex.check_overwrite(str(tmp_path), True, False)           # default run: legacy behaviour (no error)
    ex.check_overwrite(str(tmp_path), False, True)           # explicit --overwrite
    ex.check_overwrite(str(tmp_path / "new"), False, False)  # fresh dir


def test_cli_custom_run(tmp_path, capsys):
    r = ex.main(["--k", "5", "--center-x", "0.4", "--center-y", "-0.3", "--dt", "0.02", "--n-steps", "5",
                 "--n-train", "6", "--n-val", "3", "--n-test", "3", "--hidden", "8", "--epochs", "1",
                 "--seed-train", "30", "--seed-val", "31", "--seed-test", "32", "--seed", "2",
                 "--out", str(tmp_path)])
    assert r["physics"] == {"k": 5.0, "center_x": 0.4, "center_y": -0.3} and r["dt"] == 0.02
    assert r["data_seeds"] == {"train": 30, "val": 31, "test": 32} and r["mlp_config"]["seed"] == 2
    assert "linear_regression" in capsys.readouterr().out


def test_backward_compatible_defaults():
    assert ex.SEEDS == {"train": 0, "val": 1, "test": 2} and ex.SIZES == {"train": 128, "val": 32, "test": 32}
    assert (ex.N_STEPS, ex.DT, ex.PHYS) == (200, 0.01, {"k": 1.0, "center_x": 0.0, "center_y": 0.0})
    ex.validate_params()
    a = ex.make_splits(sizes=SZ, n_steps=5)
    b = ex.make_splits(ex.SEEDS, SZ, 5, ex.DT, ex.PHYS)
    assert all(np.array_equal(a[k]["traj"], b[k]["traj"]) for k in a)