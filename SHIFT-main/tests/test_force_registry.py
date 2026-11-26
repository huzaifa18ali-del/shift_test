import os, sys, json
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import simulator as sim
import experiment as ex
import adaptation as ad
from train import TrainConfig

SZ = {"train": 6, "val": 3, "test": 3}
CFG = TrainConfig(8, 1e-2, 32, 2, 0)
ST = np.array([[1.0, 2.0, 0.5, -0.5], [-1.0, 0.3, 0.0, 1.0]])


def test_registry_lookup_matches_factories_exactly():
    assert set(sim.FORCES) == {"restoring", "constant"}
    r = sim.make_force({"name": "restoring", "k": 3.0, "center_x": 0.5, "center_y": -1.0})
    assert np.array_equal(r(ST), sim.restoring_force(3.0, 0.5, -1.0)(ST))
    assert np.array_equal(sim.make_force({"name": "restoring", "k": 2})(ST), sim.restoring_force(2.0)(ST))
    c = sim.make_force({"name": "constant", "fx": 0.5, "fy": -2})
    assert np.array_equal(c(ST), sim.constant_force(0.5, -2.0)(ST)) and c(ST).shape == (2, 2)


def test_spec_normalised_and_inputs_not_mutated():
    spec = {"name": "constant", "fx": 1, "fy": 2}
    out = sim.validate_force_spec(spec)
    assert out == {"name": "constant", "fx": 1.0, "fy": 2.0} and spec == {"name": "constant", "fx": 1, "fy": 2}


@pytest.mark.parametrize("spec", [
    None, {}, {"k": 1.0}, {"name": "gravity"}, {"name": "restoring"}, {"name": "restoring", "k": 0},
    {"name": "restoring", "k": -1}, {"name": "restoring", "k": float("nan")},
    {"name": "restoring", "k": 1, "fx": 1}, {"name": "constant", "fx": 1}, {"name": "constant", "fx": 1, "fy": float("inf")},
    {"name": "constant", "fx": "a", "fy": 1}, {"name": "constant", "fx": True, "fy": 1},
])
def test_invalid_specs_rejected(spec):
    with pytest.raises(ValueError):
        sim.validate_force_spec(spec)


def test_baseline_pipeline_with_constant_force(tmp_path):
    spec = {"name": "constant", "fx": 0.5, "fy": -1.0}
    d = ex.make_splits(sizes=SZ, n_steps=5, dt=0.02, force=spec)
    for s in d.values():
        assert np.allclose(s["acc"], [0.5, -1.0])
        assert np.allclose(s["nxt"][:, 2:], s["cur"][:, 2:] + s["acc"] * 0.02)
    r = ex.run_experiment(CFG, str(tmp_path), sizes=SZ, n_steps=5, dt=0.02, force=spec)
    assert r["physics"] == spec                                   # force law documented in the record
    m = r["metrics"]
    orc = m["analytical_oracle (simulator reference, not a learned model)"]["test"]
    assert orc["accel_mse"] == 0 and orc["next_state_mse"] < 1e-24
    assert m["linear_regression"]["test"]["accel_mse"] < 1e-18
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "prediction_comparison.png").stat().st_size > 0


def test_default_and_explicit_restoring_give_identical_results(tmp_path):
    a = ex.run_experiment(CFG, str(tmp_path / "a"), sizes=SZ, n_steps=5)
    b = ex.run_experiment(CFG, str(tmp_path / "b"), sizes=SZ, n_steps=5,
                          force={"name": "restoring", "k": 1.0})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["physics"] == {"k": 1.0, "center_x": 0.0, "center_y": 0.0}      # legacy schema


def test_baseline_cli_force_options(tmp_path):
    for argv in (["--force", "constant"], ["--force", "constant", "--fx", "1"],
                 ["--force", "constant", "--fx", "1", "--fy", "1", "--k", "3"], ["--fx", "1", "--fy", "1"]):
        with pytest.raises(SystemExit):
            ex.main(argv + ["--out", str(tmp_path / "x")])
    r = ex.main(["--force", "constant", "--fx", "0.5", "--fy", "-1", "--n-steps", "5", "--n-train", "6",
                 "--n-val", "3", "--n-test", "3", "--hidden", "8", "--epochs", "1", "--out", str(tmp_path)])
    assert r["physics"] == {"name": "constant", "fx": 0.5, "fy": -1.0}


def test_adaptation_restoring_to_constant_world():
    new = {"name": "constant", "fx": 0.5, "fy": -1.0}
    r = ad.run_seed(0, budgets=(2, 4), epochs=1, hidden=8, batch_size=32, sizes=SZ, n_steps=5, force_new=new)
    assert r["k_old"] == 1.0 and r["k_new"] is None and r["force_new"] == new and "force_old" not in r
    assert r["oracle"]["new_test"]["accel_mse"] == 0
    w = ad.make_world(1.0, ad.seed_plan(0)["new"], SZ, 5, force=new)
    assert np.allclose(w["test"]["acc"], [0.5, -1.0])


def test_adaptation_rejects_bad_force_configs():
    const = {"name": "constant", "fx": 1.0, "fy": 0.0}
    with pytest.raises(ValueError, match="zero variance"):
        ad.run_seed(0, budgets=(2,), epochs=1, hidden=8, sizes=SZ, n_steps=5, force_old=const)
    with pytest.raises(ValueError, match="k/center"):
        ad.validate_params(force_new={"name": "restoring", "k": 2.0})
    with pytest.raises(ValueError):
        ad.validate_params(force_new={"name": "nope"})


def test_adaptation_cli_force_options(tmp_path):
    for argv in (["--force-new", "constant"], ["--force-new", "constant", "--fx-new", "1", "--fy-new", "1", "--k-new", "3"],
                 ["--fx-new", "1"], ["--force-old", "constant", "--fx-old", "1", "--fy-old", "1", "--epochs", "1"]):
        with pytest.raises(SystemExit):
            ad.main(argv + ["--out", str(tmp_path / "x")])
    m = ad.main(["--force-new", "constant", "--fx-new", "0.5", "--fy-new", "-1", "--repetitions", "1", "--epochs", "1",
                 "--hidden", "8", "--budgets", "2", "--n-train", "4", "--n-val", "2", "--n-test", "2",
                 "--n-steps", "5", "--out", str(tmp_path)])
    assert m["config"]["force_new"] == {"name": "constant", "fx": 0.5, "fy": -1.0} and "force_old" not in m["config"]
    assert (tmp_path / "adaptation_curve.png").stat().st_size > 0


def test_default_adaptation_record_schema_unchanged():
    r = ad.run_seed(0, budgets=(2,), epochs=1, hidden=8, batch_size=32, sizes=SZ, n_steps=5)
    assert set(r) == {"seed", "data_seeds", "k_old", "k_new", "pretrain", "unchanged", "oracle", "budgets"}