import os, sys, json, copy
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ExperimentConfig, SimulationConfig, DatasetConfig, ModelConfig, TrainingConfig


def test_round_trip_default_and_custom(tmp_path):
    ec = ExperimentConfig()
    assert ExperimentConfig.from_dict(json.loads(ec.to_json())) == ec
    ec2 = ExperimentConfig(SimulationConfig(0.02, 50), DatasetConfig(6, 3, 3, 10, 11, 12), ModelConfig(8),
                           TrainingConfig(2, 1e-2, 16, 3, "cpu"), force={"name": "constant", "fx": 0.5, "fy": -1.0},
                           force_new={"name": "constant", "fx": 1.0, "fy": 0.0}, budgets=(2, 4), adaptation_seeds=(0, 5))
    p = tmp_path / "c.json"; ec2.to_json(str(p))
    assert ExperimentConfig.from_json(str(p)) == ec2
    assert ec2.save(str(tmp_path / "out")).endswith("config.json")


def test_defaults_and_partial_dict():
    d = ExperimentConfig().to_dict()
    assert d["simulation"] == {"dt": 0.01, "n_steps": 200}
    assert d["force_old"]["k"] == 1.0 and d["force_new"]["k"] == 9.0 and d["budgets"] == [8, 16, 32, 64, 128]
    assert ExperimentConfig.from_dict({"model": {"hidden": 8}}).model.hidden == 8


@pytest.mark.parametrize("bad", [[], {"nope": 1}, {"model": {"width": 3}}, {"model": 5}, {"budgets": 3}])
def test_structural_errors(bad):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(bad)
    with pytest.raises(ValueError, match="invalid config JSON"):
        ExperimentConfig.from_json("{not json")


def test_adaptation_args_and_centre_conflict():
    ec = ExperimentConfig()
    assert ec.adaptation_args() == {"k_old": 1.0, "force_old": None, "k_new": 9.0, "force_new": None, "center": (0.0, 0.0)}
    ec.force_new = {"name": "restoring", "k": 9.0, "center_x": 1.0, "center_y": 0.0}
    with pytest.raises(ValueError, match="share one centre"):
        ec.adaptation_args()


# ---- below need the existing validators / pipelines (torch) ----
try:
    import torch, experiment, adaptation
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@needs_torch
def test_defaults_match_project_constants():
    ec = ExperimentConfig()
    assert (ec.simulation.dt, ec.simulation.n_steps) == (experiment.DT, experiment.N_STEPS)
    assert ec.sizes() == experiment.SIZES and ec.seeds() == experiment.SEEDS
    assert ec.baseline_args() == (experiment.PHYS, None)
    from train import TrainConfig
    assert ec.train_config() == TrainConfig()
    assert ec.adaptation_args()["k_old"] == adaptation.K_OLD and ec.adaptation_args()["k_new"] == adaptation.K_NEW
    assert ec.budgets == adaptation.BUDGETS and ec.sizes() == adaptation.SIZES
    ec.validate()


@needs_torch
@pytest.mark.parametrize("path,val,msg", [
    (("simulation", "dt"), -1.0, "dt"), (("simulation", "n_steps"), 0, "n_steps"),
    (("dataset", "n_val"), 0, "val"), (("dataset", "seed_val"), 0, "seeds"),
    (("model", "hidden"), 0, "hidden"), (("training", "lr"), 0.0, "lr"),
    (("training", "batch_size"), 0, "batch_size"), (("training", "epochs"), 0, "epochs"),
    (("training", "seed"), -1, "seed"), (("training", "device"), "", "device")])
def test_invalid_values_rejected(path, val, msg):
    d = ExperimentConfig().to_dict(); d[path[0]][path[1]] = val
    with pytest.raises(ValueError, match=msg):
        ExperimentConfig.from_dict(d).validate()


@needs_torch
def test_invalid_forces_and_budgets():
    for field_, spec in (("force", {"name": "gravity"}), ("force", {"name": "constant", "fx": 1.0}),
                         ("force_new", {"name": "restoring", "k": -1.0}), ("force_old", {"name": "constant", "fx": 0, "fy": float("nan")})):
        with pytest.raises(ValueError):
            ExperimentConfig.from_dict({field_: spec}).validate()
    with pytest.raises(ValueError, match="exceeds"):
        ExperimentConfig.from_dict({"budgets": [256]}).validate()


@needs_torch
def test_baseline_writes_config_and_custom_force(tmp_path):
    ec = ExperimentConfig.from_dict({"simulation": {"n_steps": 5}, "dataset": {"n_train": 6, "n_val": 3, "n_test": 3},
                                     "model": {"hidden": 8}, "training": {"epochs": 2, "batch_size": 32},
                                     "force": {"name": "constant", "fx": 0.5, "fy": -1.0}})
    rec = experiment.run_from_config(ec, str(tmp_path))
    assert rec["physics"] == {"name": "constant", "fx": 0.5, "fy": -1.0}
    assert ExperimentConfig.from_json(str(tmp_path / "config.json")) == ec
    assert "k_old" not in json.load(open(tmp_path / "metrics.json"))


@needs_torch
def test_adaptation_cli_writes_config(tmp_path):
    adaptation.main(["--seeds", "0", "--epochs", "1", "--budgets", "2", "--n-train", "4", "--n-val", "2", "--n-test", "2",
                     "--n-steps", "4", "--hidden", "4", "--force-new", "constant", "--fx-new", "0.5", "--fy-new", "-1",
                     "--out", str(tmp_path)])
    ec = ExperimentConfig.from_json(str(tmp_path / "config.json"))
    assert ec.force_new == {"name": "constant", "fx": 0.5, "fy": -1.0} and ec.force_old["k"] == 1.0
    assert ec.budgets == (2,) and ec.adaptation_seeds == (0,)