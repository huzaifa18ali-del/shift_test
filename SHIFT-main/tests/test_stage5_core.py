"""Stage 5 tests that need no torch (stub checkpoint with an exact-physics 'model')."""
import os, sys, json
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import independent_eval as ie
from independent_eval import evaluate_checkpoint, load_test_csv, compute_metrics, CSVError
from simulator import restoring_force, constant_force

RK5 = {"name": "restoring", "k": 5.0}


class Stub:
    """Stands in for a Checkpoint: predicts the physics of force `fn` (optionally with a fixed error)."""
    path, fingerprint, architecture = "stub.pt", "abc", {"type": "MLP", "in_dim": 4, "hidden": 8, "out_dim": 2}
    force, simulation = {"name": "restoring", "k": 1.0}, {"dt": 0.01, "n_steps": 200}
    experiment_config = {"dataset": {"seed_train": 0, "seed_val": 1, "seed_test": 2}}
    def __init__(self, fn, err=0.0): self.fn, self.err = fn, err
    def predict(self, x): return self.fn(x) + self.err


def csv_file(tmp_path, text, name="t.csv"):
    p = tmp_path / name; p.write_text(text); return str(p)


def test_metrics_known_values():
    true = np.zeros((4, 2)); pred = np.array([[1., -1.], [1., -1.], [2., 0.], [0., 0.]])
    m = compute_metrics(pred, true)
    assert m["accel_mse"] == pytest.approx(8 / 8) and m["accel_mae"] == pytest.approx(6 / 8)
    assert m["accel_rmse"] == pytest.approx(1.0) and m["n_samples"] == 4
    with pytest.raises(ValueError): compute_metrics(np.zeros((0, 2)), np.zeros((0, 2)))


def test_generated_eval_perfect_and_constant_error(tmp_path):
    r = evaluate_checkpoint(Stub(restoring_force(5.0)), RK5, n_steps=10, n_particles=7, out_dir=str(tmp_path))
    assert r["metrics"]["n_samples"] == 70 and r["metrics"]["accel_mse"] < 1e-24
    assert r["test_data"]["source"] == "generated" and r["seed"] == 42
    r = evaluate_checkpoint(Stub(restoring_force(5.0), err=0.5), RK5, n_steps=3, n_particles=4, out_dir=str(tmp_path))
    assert r["metrics"]["accel_rmse"] == pytest.approx(0.5) and r["metrics"]["accel_mae"] == pytest.approx(0.5)


def test_k_changes_independently(tmp_path):
    ck = Stub(restoring_force(1.0))               # model knows k=1
    errs = [evaluate_checkpoint(ck, {"name": "restoring", "k": k}, n_steps=5, n_particles=20,
                                out_dir=str(tmp_path))["metrics"]["accel_mse"] for k in (1.0, 5.0, 20.0)]
    assert errs[0] < 1e-24 and errs[0] < errs[1] < errs[2]     # truth follows the requested k


def test_multiple_force_types(tmp_path):
    r = evaluate_checkpoint(Stub(constant_force(1.0, -2.0)), {"name": "constant", "fx": 1.0, "fy": -2.0},
                            n_steps=4, n_particles=3, out_dir=str(tmp_path))
    assert r["metrics"]["accel_mse"] < 1e-24 and r["force"]["name"] == "constant"
    r2 = evaluate_checkpoint(Stub(restoring_force(1.0)), {"name": "constant", "fx": 1.0, "fy": -2.0},
                             n_steps=4, n_particles=3, out_dir=str(tmp_path))
    assert r2["metrics"]["accel_mse"] > 0.1 and any("differs" in s for s in r2["limitations"])


@pytest.mark.parametrize("spec", [{"name": "restoring", "k": -1}, {"name": "restoring"}, {"name": "gravity"},
                                  {"name": "restoring", "k": float("nan")}, {"name": "restoring", "k": 1, "z": 2},
                                  {"name": "constant", "fx": 1}])
def test_invalid_force(tmp_path, spec):
    with pytest.raises(ValueError):
        evaluate_checkpoint(Stub(restoring_force(1.0)), spec, out_dir=str(tmp_path))


@pytest.mark.parametrize("kw", [{"dt": 0}, {"dt": float("inf")}, {"n_steps": 0}, {"n_particles": -1}, {"seed": -3}])
def test_invalid_generation_params(tmp_path, kw):
    with pytest.raises(ValueError):
        evaluate_checkpoint(Stub(restoring_force(1.0)), RK5, out_dir=str(tmp_path), **kw)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_recorded_split_seeds_rejected(tmp_path, seed):
    with pytest.raises(ValueError, match="independent"):
        evaluate_checkpoint(Stub(restoring_force(1.0)), RK5, seed=seed, out_dir=str(tmp_path))


def test_missing_checkpoint(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(FileNotFoundError):
        evaluate_checkpoint(str(tmp_path / "nope.pt"), RK5, out_dir=str(tmp_path))


# ---------------- CSV ----------------
def test_csv_valid_and_extra_column(tmp_path):
    p = csv_file(tmp_path, "id,x,y,vx,vy\n7,0.5,0.1,0,1\n8,-1,2,3,4\n\n")
    s = load_test_csv(p)
    assert s.shape == (2, 4) and s[1].tolist() == [-1, 2, 3, 4]
    r = evaluate_checkpoint(Stub(restoring_force(5.0)), RK5, input_csv=p, out_dir=str(tmp_path))
    assert r["test_data"]["source"] == "external_csv" and r["test_data"]["n_samples"] == 2
    assert r["simulation"] is None and r["seed"] is None and r["metrics"]["accel_mse"] < 1e-24


@pytest.mark.parametrize("text,msg", [
    ("", "empty"), ("\n\n", "empty"), ("x,y,vx,vy\n", "no data"),
    ("x,y,vx\n1,2,3\n", "missing required"), ("x,y,vx,vx,vy\n1,2,3,4,5\n", "duplicate"),
    ("x,y,vx,vy\n1,2,3\n", "expected 4 fields"), ("x,y,vx,vy\n1,2,3,4,5\n", "expected 4 fields"),
    ("x,y,vx,vy\n1,,3,4\n", "missing value"), ("x,y,vx,vy\n1,abc,3,4\n", "non-numeric"),
    ("x,y,vx,vy\n1,nan,3,4\n", "NaN/infinite"), ("x,y,vx,vy\n1,2,inf,4\n", "NaN/infinite"),
    ("x,y,vx,vy\n1,2,3,-Infinity\n", "NaN/infinite")])
def test_csv_invalid(tmp_path, text, msg):
    with pytest.raises(CSVError, match=msg):
        load_test_csv(csv_file(tmp_path, text))


def test_csv_missing_file_and_generation_args_rejected(tmp_path):
    with pytest.raises(FileNotFoundError): load_test_csv(str(tmp_path / "none.csv"))
    p = csv_file(tmp_path, "x,y,vx,vy\n1,2,3,4\n")
    with pytest.raises(ValueError, match="only apply to generated"):
        evaluate_checkpoint(Stub(restoring_force(1.0)), RK5, input_csv=p, seed=3, out_dir=str(tmp_path))


# ---------------- results ----------------
def test_results_json_and_no_silent_overwrite(tmp_path):
    ck = Stub(restoring_force(1.0))
    a = evaluate_checkpoint(ck, RK5, n_steps=2, n_particles=2, out_dir=str(tmp_path))
    b = evaluate_checkpoint(ck, RK5, n_steps=2, n_particles=2, out_dir=str(tmp_path))
    assert a["results_path"] != b["results_path"] and len(os.listdir(tmp_path)) == 2
    saved = json.load(open(a["results_path"]))
    for k in ("checkpoint", "force", "simulation", "seed", "test_data", "metrics", "timestamp_utc", "limitations"):
        assert k in saved
    assert saved["checkpoint"]["architecture"]["hidden"] == 8
    out = str(tmp_path / "mine.json")
    evaluate_checkpoint(ck, RK5, n_steps=2, n_particles=2, out=out)
    before = open(out).read()
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(ck, RK5, n_steps=2, n_particles=2, out=out)
    assert open(out).read() == before
    evaluate_checkpoint(ck, RK5, n_steps=2, n_particles=2, out=out, overwrite=True)


# ---------------- CLI ----------------
def test_cli_help_and_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as e: ie.main(["--help"])
    assert e.value.code == 0 and "--input-csv" in capsys.readouterr().out
    pytest.importorskip("torch")
    assert ie.main(["--checkpoint", str(tmp_path / "x.pt"), "--force", "restoring", "--k", "5",
                    "--out-dir", str(tmp_path)]) == 2
    assert "not found" in capsys.readouterr().err


def test_force_from_args_defaults_to_checkpoint_force():
    a = ie.build_parser().parse_args(["--checkpoint", "c", "--k", "5"])
    assert ie.force_from_args(a, {"name": "restoring", "k": 1.0, "center_x": 0.2, "center_y": 0.0}) == \
        {"name": "restoring", "k": 5.0, "center_x": 0.2, "center_y": 0.0}
    with pytest.raises(ValueError, match="--force is required"): ie.force_from_args(a, None)
