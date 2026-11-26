import os, sys, json
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import experiment as ex
from checkpoint import (save_checkpoint, load_checkpoint, CheckpointError, Checkpoint, FORMAT_NAME,
                        FORMAT_VERSION)
from config import ExperimentConfig
from models import MLP
from simulator import restoring_force, generate_dataset
from train import TrainConfig, Normalizer, train_mlp, predict_mlp, params_fingerprint

FORCE = {"name": "restoring", "k": 2.0, "center_x": 0.1, "center_y": -0.2}
SIM = {"dt": 0.01, "n_steps": 5}
CFG = TrainConfig(hidden=8, lr=1e-2, batch_size=32, epochs=2, seed=0)


@pytest.fixture(scope="module")
def trained():
    f = restoring_force(2.0, 0.1, -0.2)
    _, (cur, nxt, acc) = generate_dataset(f, 6, 5, 0.01, 0)
    _, (cv, _, av) = generate_dataset(f, 3, 5, 0.01, 1)
    xn, yn = Normalizer().fit(cur), Normalizer().fit(acc)
    model, hist = train_mlp(cur, acc, cv, av, CFG, xn, yn)
    return {"model": model, "xn": xn, "yn": yn, "hist": hist, "x": cv}


def _save(tmp_path, t, name="m.pt", **kw):
    kw.setdefault("force", FORCE); kw.setdefault("simulation", SIM); kw.setdefault("train_config", CFG)
    return save_checkpoint(str(tmp_path / name), t["model"], t["xn"], t["yn"], **kw)


def _raw(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _write(tmp_path, raw, name="bad.pt"):
    p = str(tmp_path / name); torch.save(raw, p); return p


# ---- 1. save / load -------------------------------------------------------------------------
def test_save_load_roundtrip_restores_everything(tmp_path, trained):
    ec = ExperimentConfig()
    p = _save(tmp_path, trained, experiment_config=ec, training_metadata={"best_epoch": 2})
    ck = load_checkpoint(p)
    assert isinstance(ck, Checkpoint) and isinstance(ck.model, MLP) and ck.model is not trained["model"]
    assert ck.architecture == {"type": "MLP", "in_dim": 4, "hidden": 8, "out_dim": 2}
    assert ck.force == {"name": "restoring", "k": 2.0, "center_x": 0.1, "center_y": -0.2}
    assert ck.simulation == SIM and ck.train_config == CFG.to_dict()
    assert ck.experiment_config == ec.to_dict() and ck.training_metadata == {"best_epoch": 2}
    assert ck.fingerprint == params_fingerprint(trained["model"])
    assert ExperimentConfig.from_dict(ck.experiment_config).to_dict() == ec.to_dict()
    assert not ck.model.training


def test_file_has_no_pickled_model_object(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained))            # weights_only=True would have refused a pickled model
    assert raw["format"] == FORMAT_NAME and raw["format_version"] == FORMAT_VERSION
    assert all(torch.is_tensor(v) for v in raw["state_dict"].values())
    assert not any(isinstance(v, torch.nn.Module) for v in raw.values())


def test_loads_into_fresh_instance_without_training(tmp_path, trained):
    ck = load_checkpoint(_save(tmp_path, trained))
    fresh = MLP(4, 8, 2)
    assert params_fingerprint(fresh) != params_fingerprint(trained["model"])
    fresh.load_state_dict(ck.model.state_dict())
    assert params_fingerprint(fresh) == params_fingerprint(trained["model"])


# ---- 2. prediction consistency ---------------------------------------------------------------
def test_predictions_match_before_and_after_loading(tmp_path, trained):
    x = trained["x"]
    before = predict_mlp(trained["model"], x, trained["xn"], trained["yn"])
    ck = load_checkpoint(_save(tmp_path, trained))
    after = ck.predict(x)
    assert after.shape == (len(x), 2) and np.allclose(before, after, rtol=1e-6, atol=1e-8)
    assert np.allclose(before, predict_mlp(ck.model, x, ck.x_norm, ck.y_norm), rtol=1e-6, atol=1e-8)


def test_predict_rejects_wrong_input_shape(tmp_path, trained):
    ck = load_checkpoint(_save(tmp_path, trained))
    with pytest.raises(ValueError, match="shape"):
        ck.predict(np.zeros((3, 5)))


# ---- 3. normalizer preservation --------------------------------------------------------------
def test_normalizer_preserved_exactly(tmp_path, trained):
    ck = load_checkpoint(_save(tmp_path, trained))
    for a, b in ((ck.x_norm, trained["xn"]), (ck.y_norm, trained["yn"])):
        assert np.array_equal(a.mean, b.mean) and np.array_equal(a.std, b.std) and a.eps == b.eps
        assert a.mean.dtype == np.float64
    x = trained["x"]
    assert np.array_equal(ck.x_norm.transform(x), trained["xn"].transform(x))
    assert np.allclose(ck.y_norm.inverse(ck.y_norm.transform(x[:, :2])), x[:, :2])


def test_save_rejects_unfitted_or_mismatched_normalizer(tmp_path, trained):
    with pytest.raises(CheckpointError, match="fitted"):
        save_checkpoint(str(tmp_path / "a.pt"), trained["model"], Normalizer(), trained["yn"])
    wrong = Normalizer().fit(np.random.default_rng(0).normal(size=(10, 3)))
    with pytest.raises(CheckpointError, match="needs"):
        save_checkpoint(str(tmp_path / "b.pt"), trained["model"], wrong, trained["yn"])
    assert not (tmp_path / "a.pt").exists() and not (tmp_path / "b.pt").exists()


# ---- 4. missing / invalid fields -------------------------------------------------------------
@pytest.mark.parametrize("key", ["format", "format_version", "architecture", "state_dict", "normalizers"])
def test_missing_required_field(tmp_path, trained, key):
    raw = _raw(_save(tmp_path, trained)); del raw[key]
    with pytest.raises(CheckpointError, match=key):
        load_checkpoint(_write(tmp_path, raw))


def test_missing_file_and_garbage_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_checkpoint(str(tmp_path / "nope.pt"))
    bad = tmp_path / "garbage.pt"; bad.write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError, match="not a readable"):
        load_checkpoint(str(bad))


def test_not_a_dict_wrong_format_and_version(tmp_path, trained):
    with pytest.raises(CheckpointError, match="dict"):
        load_checkpoint(_write(tmp_path, [1, 2, 3], "list.pt"))
    raw = _raw(_save(tmp_path, trained)); raw["format"] = "other"
    with pytest.raises(CheckpointError, match="not a SHIFT checkpoint"):
        load_checkpoint(_write(tmp_path, raw))
    raw = _raw(_save(tmp_path, trained, name="v.pt")); raw["format_version"] = 99
    with pytest.raises(CheckpointError, match="format_version"):
        load_checkpoint(_write(tmp_path, raw))


def test_unknown_top_level_field_rejected(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained)); raw["surprise"] = 1
    with pytest.raises(CheckpointError, match="unknown"):
        load_checkpoint(_write(tmp_path, raw))


def test_bad_architecture_fields(tmp_path, trained):
    for arch, msg in (({"type": "MLP", "in_dim": 4, "hidden": 8}, "missing"),
                      ({"type": "CNN", "in_dim": 4, "hidden": 8, "out_dim": 2}, "unsupported model type"),
                      ({"type": "MLP", "in_dim": 4, "hidden": 0, "out_dim": 2}, "integer >= 1"),
                      ({"type": "MLP", "in_dim": 4, "hidden": 8.5, "out_dim": 2}, "integer >= 1"),
                      ("MLP", "dict")):
        raw = _raw(_save(tmp_path, trained, name="ok.pt", overwrite=True)); raw["architecture"] = arch
        with pytest.raises(CheckpointError, match=msg):
            load_checkpoint(_write(tmp_path, raw))


def test_bad_normalizer_entries(tmp_path, trained):
    def mutate(f):
        raw = _raw(_save(tmp_path, trained, name="ok.pt", overwrite=True)); f(raw["normalizers"])
        return _write(tmp_path, raw)
    with pytest.raises(CheckpointError, match="'x' and 'y'"):
        load_checkpoint(mutate(lambda n: n.pop("y")))
    with pytest.raises(CheckpointError, match="missing field"):
        load_checkpoint(mutate(lambda n: n["x"].pop("std")))
    with pytest.raises(CheckpointError, match="needs"):
        load_checkpoint(mutate(lambda n: n["x"].update(mean=[0.0, 0.0, 0.0])))
    with pytest.raises(CheckpointError, match="std > 0"):
        load_checkpoint(mutate(lambda n: n["y"].update(std=[1.0, 0.0])))
    with pytest.raises(CheckpointError, match="finite"):
        load_checkpoint(mutate(lambda n: n["y"].update(mean=[float("nan"), 0.0])))
    with pytest.raises(CheckpointError, match="not numeric"):
        load_checkpoint(mutate(lambda n: n["x"].update(mean=["a", "b", "c", "d"])))


def test_invalid_force_spec_rejected(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained)); raw["force"] = {"name": "gravity"}
    with pytest.raises(CheckpointError, match="force"):
        load_checkpoint(_write(tmp_path, raw))
    with pytest.raises(CheckpointError, match="force"):
        _save(tmp_path, trained, name="f.pt", force={"name": "restoring", "k": -1.0})


def test_constant_force_roundtrip(tmp_path, trained):
    ck = load_checkpoint(_save(tmp_path, trained, force={"name": "constant", "fx": 0.0, "fy": -0.5}))
    assert ck.force == {"name": "constant", "fx": 0.0, "fy": -0.5}


def test_optional_fields_default_to_none(tmp_path, trained):
    ck = load_checkpoint(save_checkpoint(str(tmp_path / "min.pt"), trained["model"], trained["xn"], trained["yn"]))
    assert ck.force is None and ck.simulation is None and ck.train_config is None and ck.experiment_config is None


def test_non_serialisable_metadata_rejected(tmp_path, trained):
    with pytest.raises(CheckpointError, match="JSON"):
        _save(tmp_path, trained, training_metadata={"x": object()})


def test_only_mlp_models_can_be_saved(tmp_path, trained):
    with pytest.raises(CheckpointError, match="MLP"):
        save_checkpoint(str(tmp_path / "z.pt"), torch.nn.Linear(4, 2), trained["xn"], trained["yn"])


# ---- 5. incompatible architecture / tensors --------------------------------------------------
def test_state_dict_shape_mismatch_with_architecture(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained)); raw["architecture"]["hidden"] = 16
    with pytest.raises(CheckpointError, match="shape"):
        load_checkpoint(_write(tmp_path, raw))
    raw = _raw(_save(tmp_path, trained, name="m2.pt")); raw["architecture"]["in_dim"] = 5
    with pytest.raises(CheckpointError, match="shape|needs"):
        load_checkpoint(_write(tmp_path, raw))


def test_state_dict_missing_or_unexpected_keys(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained)); del raw["state_dict"]["net.2.bias"]
    with pytest.raises(CheckpointError, match="net.2.bias"):
        load_checkpoint(_write(tmp_path, raw))
    raw = _raw(_save(tmp_path, trained, name="m2.pt")); raw["state_dict"]["extra.weight"] = torch.zeros(1)
    with pytest.raises(CheckpointError, match="extra.weight"):
        load_checkpoint(_write(tmp_path, raw))


def test_non_tensor_nonfloat_and_nonfinite_weights(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained)); raw["state_dict"]["net.0.bias"] = [0.0] * 8
    with pytest.raises(CheckpointError, match="not a tensor"):
        load_checkpoint(_write(tmp_path, raw))
    raw = _raw(_save(tmp_path, trained, name="m2.pt")); raw["state_dict"]["net.0.bias"] = torch.zeros(8, dtype=torch.long)
    with pytest.raises(CheckpointError, match="floating-point"):
        load_checkpoint(_write(tmp_path, raw))
    raw = _raw(_save(tmp_path, trained, name="m3.pt")); raw["state_dict"]["net.0.bias"][0] = float("nan")
    with pytest.raises(CheckpointError, match="NaN"):
        load_checkpoint(_write(tmp_path, raw))


def test_tampered_weights_detected_by_fingerprint(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained)); raw["state_dict"]["net.0.bias"][0] += 1.0
    with pytest.raises(CheckpointError, match="fingerprint"):
        load_checkpoint(_write(tmp_path, raw))


def test_normalizer_dim_mismatch_with_architecture(tmp_path, trained):
    raw = _raw(_save(tmp_path, trained)); raw["normalizers"]["y"] = {"mean": [0.0] * 3, "std": [1.0] * 3}
    with pytest.raises(CheckpointError, match="needs"):
        load_checkpoint(_write(tmp_path, raw))


# ---- 6. CPU loading --------------------------------------------------------------------------
def test_loads_onto_cpu_and_ignores_saved_device(tmp_path, trained):
    ck = load_checkpoint(_save(tmp_path, trained))
    assert all(p.device.type == "cpu" for p in ck.model.parameters())
    raw = _raw(ck.path)
    assert all(v.device.type == "cpu" for v in raw["state_dict"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_checkpoint_from_cuda_model_loads_on_cpu(tmp_path, trained):
    import copy
    m = copy.deepcopy(trained["model"]).to("cuda")
    p = save_checkpoint(str(tmp_path / "cuda.pt"), m, trained["xn"], trained["yn"])
    ck = load_checkpoint(p)
    assert all(q.device.type == "cpu" for q in ck.model.parameters())
    assert np.allclose(ck.predict(trained["x"]), predict_mlp(trained["model"], trained["x"], trained["xn"],
                                                              trained["yn"]), rtol=1e-5, atol=1e-6)


def test_load_remaps_non_cpu_tagged_tensors(tmp_path, trained, monkeypatch):
    """map_location='cpu' must be requested so a GPU-saved file can load on a CPU-only machine."""
    p = _save(tmp_path, trained); seen = {}
    real = torch.load
    def spy(*a, **kw):
        seen.update(kw); return real(*a, **kw)
    monkeypatch.setattr(torch, "load", spy)
    load_checkpoint(p)
    assert seen.get("map_location") == "cpu" and seen.get("weights_only") is True


# ---- overwrite protection --------------------------------------------------------------------
def test_does_not_overwrite_silently(tmp_path, trained):
    p = _save(tmp_path, trained)
    before = open(p, "rb").read()
    with pytest.raises(FileExistsError, match="overwrite"):
        _save(tmp_path, trained)
    assert open(p, "rb").read() == before
    _save(tmp_path, trained, overwrite=True)
    assert [f.name for f in tmp_path.iterdir()] == ["m.pt"]       # no leftover temp files


def test_save_creates_missing_directories(tmp_path, trained):
    p = save_checkpoint(str(tmp_path / "a" / "b" / "m.pt"), trained["model"], trained["xn"], trained["yn"])
    assert os.path.isfile(p)


# ---- 7. existing functionality unaffected ----------------------------------------------------
def test_run_experiment_without_checkpoint_unchanged(tmp_path):
    sz = {"train": 6, "val": 3, "test": 3}
    r = ex.run_experiment(CFG, str(tmp_path), sizes=sz, n_steps=5)
    assert set(r) >= {"metrics", "mlp_config", "final_train_loss_normalised"}
    assert sorted(f.name for f in tmp_path.iterdir()) == ["config.json", "metrics.json", "prediction_comparison.png",
                                                          "training_curves.png"]


def test_run_experiment_saves_loadable_checkpoint(tmp_path):
    sz = {"train": 6, "val": 3, "test": 3}
    phys = {"k": 3.0, "center_x": 0.2, "center_y": 0.0}
    ck_path = str(tmp_path / "ck" / "baseline.pt")
    r = ex.run_experiment(CFG, str(tmp_path / "out"), sizes=sz, n_steps=5, phys=phys, checkpoint_path=ck_path)
    ck = load_checkpoint(ck_path)
    assert ck.force == {"name": "restoring", "k": 3.0, "center_x": 0.2, "center_y": 0.0}
    assert ck.simulation == {"dt": 0.01, "n_steps": 5} and ck.train_config["hidden"] == 8
    assert ck.training_metadata["final_val_loss_normalised"] == r["final_val_loss_normalised"]
    # independent evaluation: the loaded model reproduces the experiment's reported test metrics
    d = ex.make_splits(sizes=sz, n_steps=5, phys=phys)
    from evaluate import evaluate
    m = evaluate(d["test"]["cur"], d["test"]["nxt"], d["test"]["acc"], ck.predict(d["test"]["cur"]), 0.01)
    assert m == pytest.approx(r["metrics"]["mlp"]["test"], rel=1e-5, abs=1e-12)


def test_run_experiment_checkpoint_not_overwritten_and_fails_before_training(tmp_path):
    sz = {"train": 6, "val": 3, "test": 3}
    ck_path = str(tmp_path / "b.pt")
    ex.run_experiment(CFG, str(tmp_path / "o1"), sizes=sz, n_steps=5, checkpoint_path=ck_path)
    with pytest.raises(FileExistsError):
        ex.run_experiment(CFG, str(tmp_path / "o2"), sizes=sz, n_steps=5, checkpoint_path=ck_path)
    assert not (tmp_path / "o2").exists()                           # failed before any output was produced
    ex.run_experiment(CFG, str(tmp_path / "o2"), sizes=sz, n_steps=5, checkpoint_path=ck_path,
                      overwrite_checkpoint=True)


def test_cli_save_checkpoint_flags(tmp_path):
    out, ck = str(tmp_path / "o"), str(tmp_path / "c.pt")
    args = ["--out", out, "--epochs", "1", "--hidden", "4", "--n-train", "4", "--n-val", "2", "--n-test", "2",
            "--n-steps", "5", "--save-checkpoint", ck]
    ex.main(args)
    assert load_checkpoint(ck).architecture["hidden"] == 4
    with pytest.raises(SystemExit):                                  # existing file, no --overwrite-checkpoint
        ex.main(args + ["--overwrite"])
    ex.main(args + ["--overwrite", "--overwrite-checkpoint"])
