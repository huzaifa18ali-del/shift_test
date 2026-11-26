"""Stage 5 tests with REAL checkpoints (need torch)."""
import os, sys, json
import numpy as np
import pytest
torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import independent_eval as ie
import train
from checkpoint import save_checkpoint, load_checkpoint, CheckpointError
from config import ExperimentConfig
from simulator import restoring_force, generate_dataset, acceleration
from train import TrainConfig, Normalizer, train_mlp, params_fingerprint

CFG = TrainConfig(hidden=8, lr=1e-2, batch_size=32, epochs=2, seed=0)


@pytest.fixture(scope="module")
def ck_path(tmp_path_factory):
    f = restoring_force(1.0)
    _, (cur, _, acc) = generate_dataset(f, 6, 5, 0.01, 0)
    _, (cv, _, av) = generate_dataset(f, 3, 5, 0.01, 1)
    xn, yn = Normalizer().fit(cur), Normalizer().fit(acc)
    model, _ = train_mlp(cur, acc, cv, av, CFG, xn, yn)
    p = str(tmp_path_factory.mktemp("ck") / "m.pt")
    ec = ExperimentConfig()     # records dataset seeds 0/1/2
    save_checkpoint(p, model, xn, yn, force={"name": "restoring", "k": 1.0}, simulation={"dt": 0.01, "n_steps": 5},
                    train_config=CFG, experiment_config=ec)
    return p


def test_load_and_evaluate_generated(ck_path, tmp_path):
    r = ie.evaluate_checkpoint(ck_path, {"name": "restoring", "k": 1.0}, n_steps=10, n_particles=5, out_dir=str(tmp_path))
    ck = load_checkpoint(ck_path)
    assert r["checkpoint"]["fingerprint"] == ck.fingerprint and r["checkpoint"]["architecture"]["hidden"] == 8
    assert r["metrics"]["n_samples"] == 50 and np.isfinite(r["metrics"]["accel_mse"])
    # metrics agree with a manual computation from the loaded model
    s = ie.generate_test_states(restoring_force(1.0), 0.01, 10, 5, 42)
    manual = np.mean((ck.predict(s) - acceleration(s, restoring_force(1.0))) ** 2)
    assert r["metrics"]["accel_mse"] == pytest.approx(manual)


def test_csv_k_and_second_force(ck_path, tmp_path):
    csvp = tmp_path / "s.csv"; csvp.write_text("x,y,vx,vy\n0.5,0.2,0,0\n-0.3,0.9,1,1\n")
    e1 = ie.evaluate_checkpoint(ck_path, {"name": "restoring", "k": 1.0}, input_csv=str(csvp), out_dir=str(tmp_path))
    e5 = ie.evaluate_checkpoint(ck_path, {"name": "restoring", "k": 5.0}, input_csv=str(csvp), out_dir=str(tmp_path))
    assert e5["metrics"]["accel_mse"] > e1["metrics"]["accel_mse"] and e5["force"]["k"] == 5.0
    c = ie.evaluate_checkpoint(ck_path, {"name": "constant", "fx": 1.0, "fy": 0.0}, n_steps=3, n_particles=3,
                               out_dir=str(tmp_path))
    assert c["force"]["name"] == "constant"


def test_no_training_and_model_unchanged(ck_path, tmp_path, monkeypatch):
    def boom(*a, **k): raise AssertionError("training was called during evaluation")
    monkeypatch.setattr(train, "train_mlp", boom)
    monkeypatch.setattr(torch.optim.Adam, "step", boom)
    ck = load_checkpoint(ck_path)
    before = params_fingerprint(ck.model)
    r = ie.evaluate_checkpoint(ck, {"name": "restoring", "k": 9.0}, n_steps=5, n_particles=5, out_dir=str(tmp_path))
    assert r["model_unchanged"] is True and params_fingerprint(ck.model) == before and not ck.model.training


def test_training_seed_rejected_with_real_checkpoint(ck_path, tmp_path):
    with pytest.raises(ValueError, match="independent"):
        ie.evaluate_checkpoint(ck_path, {"name": "restoring", "k": 1.0}, seed=0, out_dir=str(tmp_path))


def test_invalid_and_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        ie.evaluate_checkpoint(str(tmp_path / "none.pt"), {"name": "restoring", "k": 1.0})
    bad = tmp_path / "bad.pt"; bad.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        ie.evaluate_checkpoint(str(bad), {"name": "restoring", "k": 1.0})


def test_cli_end_to_end(ck_path, tmp_path, capsys):
    out = str(tmp_path / "o")
    assert ie.main(["--checkpoint", ck_path, "--k", "5", "--n-steps", "5", "--n-particles", "4", "--out-dir", out]) == 0
    files = os.listdir(out); assert len(files) == 1
    assert json.load(open(os.path.join(out, files[0])))["force"]["k"] == 5.0
    assert "RMSE" in capsys.readouterr().out
