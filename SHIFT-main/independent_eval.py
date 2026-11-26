"""Stage 5: independent evaluation of a trained SHIFT checkpoint (no training, no fitting).

Python:
    from independent_eval import evaluate_checkpoint
    res = evaluate_checkpoint("ck.pt", {"name": "restoring", "k": 5.0}, dt=0.01, n_steps=200,
                              n_particles=100, seed=42)                    # fresh simulator data
    res = evaluate_checkpoint("ck.pt", {"name": "restoring", "k": 5.0}, input_csv="states.csv")
CLI:  python evaluate.py --checkpoint ck.pt --force restoring --k 5 --seed 42   (see --help)

Ground truth is always simulator.acceleration(state, make_force(spec)). The model, its weights and
the normalisers stored in the checkpoint are used as-is.
"""
import argparse, csv, json, math, os, sys
from datetime import datetime, timezone

import numpy as np

from evaluate import mse, mae
from simulator import FORCES, make_force, validate_force_spec, acceleration, generate_dataset

COLUMNS = ("x", "y", "vx", "vy")
DEFAULTS = {"dt": 0.01, "n_steps": 200, "n_particles": 100, "seed": 42}
DEFAULT_OUT_DIR = os.path.join("outputs", "stage5")


class CSVError(ValueError):
    """Raised when an externally supplied CSV file is invalid (nothing is silently dropped)."""


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_metrics(acc_pred, acc_true):
    """MSE/MAE/RMSE over all samples and both acceleration components (same mse/mae as evaluate.py)."""
    acc_pred, acc_true = np.asarray(acc_pred, float), np.asarray(acc_true, float)
    if acc_pred.shape != acc_true.shape or acc_pred.ndim != 2 or acc_pred.shape[0] == 0:
        raise ValueError(f"need equal non-empty (N, 2) arrays, got {acc_pred.shape} and {acc_true.shape}")
    return {"accel_mse": mse(acc_pred, acc_true), "accel_mae": mae(acc_pred, acc_true),
            "accel_rmse": rmse(acc_pred, acc_true), "n_samples": int(acc_pred.shape[0])}


# ---------------------------------------------------------------- test data
def load_test_csv(path):
    """Read a CSV with header columns x, y, vx, vy (extra columns ignored) -> float64 array (N, 4).

    Any bad row (wrong field count, empty cell, non-numeric, NaN/inf) raises CSVError naming the row.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"test CSV not found: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    rows = [(i, r) for i, r in enumerate(rows, 1) if r and any(c.strip() for c in r)]   # blank lines only
    if not rows:
        raise CSVError(f"{path} is empty")
    header = [c.strip() for c in rows[0][1]]
    if len(set(header)) != len(header):
        raise CSVError(f"{path}: duplicate column names in header {header}")
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise CSVError(f"{path}: missing required column(s) {missing}; header is {header}")
    if len(rows) == 1:
        raise CSVError(f"{path} has a header but no data rows")
    idx = [header.index(c) for c in COLUMNS]
    data = []
    for line, r in rows[1:]:
        if len(r) != len(header):
            raise CSVError(f"{path} line {line}: expected {len(header)} fields, got {len(r)}")
        vals = []
        for c, j in zip(COLUMNS, idx):
            cell = r[j].strip()
            if cell == "":
                raise CSVError(f"{path} line {line}: missing value in column '{c}'")
            try:
                v = float(cell)
            except ValueError:
                raise CSVError(f"{path} line {line}: non-numeric value {cell!r} in column '{c}'") from None
            if not math.isfinite(v):
                raise CSVError(f"{path} line {line}: NaN/infinite value in column '{c}'")
            vals.append(v)
        data.append(vals)
    return np.array(data, dtype=np.float64)


def _check_gen_params(dt, n_steps, n_particles, seed):
    def isint(v): return isinstance(v, (int, np.integer)) and not isinstance(v, bool)
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be a finite number > 0, got {dt!r}")
    for name, v, lo in (("n_steps", n_steps, 1), ("n_particles", n_particles, 1), ("seed", seed, 0)):
        if not isint(v) or v < lo:
            raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")


def _forbidden_seeds(ck):
    """Data seeds the checkpoint's training pipeline used for train/val data (must not be reused)."""
    ds = ((getattr(ck, "experiment_config", None) or {}).get("dataset")) or {}
    return {n: ds[n] for n in ("seed_train", "seed_val", "seed_test") if isinstance(ds.get(n), int)}


def generate_test_states(force_fn, dt, n_steps, n_particles, seed):
    """Fresh random particles simulated for n_steps; returns the n_steps*n_particles visited states (N, 4)."""
    _, (cur, _, _) = generate_dataset(force_fn, n_particles, n_steps, dt, seed)
    return cur


# ---------------------------------------------------------------- results file
def write_results(res, out_dir=DEFAULT_OUT_DIR, out=None, overwrite=False):
    """Write res as JSON. Never replaces an existing file silently; returns the path written."""
    text = json.dumps(res, indent=2)
    if out is not None:
        path = os.fspath(out)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"results file {path} already exists; choose another --out or pass --overwrite")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for n in range(1000):                                   # exclusive create: unique name, no overwrite
        path = os.path.join(out_dir, f"eval_{stamp}{'' if n == 0 else f'_{n}'}.json")
        try:
            with open(path, "x") as fh:
                fh.write(text)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"could not find a free results filename in {out_dir}")


def _limitations(source, same_force):
    lim = ["Metrics measure acceleration error only, on this one force configuration; they do not show "
           "that the model generalises to other physical environments.",
           "The model only sees (x, y, vx, vy) and was trained on a limited state range; states far outside "
           "that range are extrapolation."]
    if not same_force:
        lim.append("The evaluation force differs from the force the checkpoint was trained on (or the "
                   "checkpoint does not record it): this measures out-of-distribution behaviour.")
    if source == "external_csv":
        lim.append("External CSV states are used as given; the code cannot verify they are disjoint from "
                   "the training data or physically meaningful.")
    else:
        lim.append("Generated states come from random initial conditions; the seed is checked against all "
                   "recorded train/validation/test split seeds.")
    return lim


# ---------------------------------------------------------------- main API
def evaluate_checkpoint(checkpoint, force, dt=None, n_steps=None, n_particles=None, seed=None,
                        input_csv=None, out_dir=DEFAULT_OUT_DIR, out=None, overwrite=False, save=True):
    """Evaluate a checkpoint (path or loaded Checkpoint) on independent data; return the results dict.

    Generated mode (input_csv=None): dt/n_steps/n_particles/seed default to DEFAULTS.
    CSV mode: states come from the file; dt/n_steps/n_particles/seed must not be given.
    The result has "results_path" when save=True.
    """
    if isinstance(checkpoint, (str, os.PathLike)):
        from checkpoint import load_checkpoint          # Stage 4 loader (validates the file)
        ck = load_checkpoint(checkpoint)
    else:
        ck = checkpoint
    spec = validate_force_spec(force)                    # ValueError on bad name/parameters
    force_fn = make_force(spec)

    if input_csv is not None:
        given = {k: v for k, v in dict(dt=dt, n_steps=n_steps, n_particles=n_particles, seed=seed).items()
                 if v is not None}
        if given:
            raise ValueError(f"{sorted(given)} only apply to generated data, not to --input-csv")
        states = load_test_csv(input_csv)
        source, sim, used_seed = "external_csv", None, None
        data = {"source": source, "path": os.fspath(input_csv)}
    else:
        dt = DEFAULTS["dt"] if dt is None else dt
        n_steps = DEFAULTS["n_steps"] if n_steps is None else n_steps
        n_particles = DEFAULTS["n_particles"] if n_particles is None else n_particles
        seed = DEFAULTS["seed"] if seed is None else seed
        _check_gen_params(dt, n_steps, n_particles, seed)
        for name, s in _forbidden_seeds(ck).items():
            if seed == s:
                raise ValueError(f"seed {seed} equals the checkpoint's {name}; choose a different seed so the "
                                 f"test data is independent of training/validation data")
        states = generate_test_states(force_fn, dt, n_steps, n_particles, seed)
        source, used_seed = "generated", seed
        sim = {"dt": float(dt), "n_steps": int(n_steps), "n_particles": int(n_particles)}
        data = {"source": source, "path": None}

    # Evaluation must not change the model: compare weights before/after (no training, no fitting).
    model = getattr(ck, "model", None)
    if model is not None:
        from train import params_fingerprint
        fp_before = params_fingerprint(model)
        norm_before = [np.copy(a) for n in (ck.x_norm, ck.y_norm) for a in (n.mean, n.std)]
    acc_pred = ck.predict(states)
    acc_true = acceleration(states, force_fn)
    if not np.all(np.isfinite(acc_pred)):
        raise ValueError("model produced non-finite predictions for the test states")
    metrics = compute_metrics(acc_pred, acc_true)
    unchanged = None
    if model is not None:
        norm_after = [a for n in (ck.x_norm, ck.y_norm) for a in (n.mean, n.std)]
        unchanged = bool(params_fingerprint(model) == fp_before
                         and all(np.array_equal(a, b) for a, b in zip(norm_before, norm_after)))

    train_force = getattr(ck, "force", None)
    data["n_samples"] = metrics["n_samples"]
    res = {
        "stage": 5,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": getattr(ck, "path", None), "fingerprint": getattr(ck, "fingerprint", None),
                       "architecture": getattr(ck, "architecture", None), "training_force": train_force,
                       "training_simulation": getattr(ck, "simulation", None)},
        "force": spec, "simulation": sim, "seed": used_seed, "test_data": data,
        "metrics": metrics, "model_unchanged": unchanged,
        "limitations": _limitations(source, train_force == spec),
    }
    if save:
        res["results_path"] = write_results(res, out_dir, out, overwrite)
        with open(res["results_path"], "w") as fh:          # same file we just created exclusively
            json.dump(res, fh, indent=2)
    return res


# ---------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(
        prog="evaluate.py", description="Independently evaluate a trained SHIFT checkpoint (no retraining).",
        epilog="Example: python evaluate.py --checkpoint checkpoints/baseline.pt --force restoring --k 5 "
               "--dt 0.01 --n-steps 200 --n-particles 100 --seed 42")
    p.add_argument("--checkpoint", required=True, help="Stage 4 checkpoint file (.pt)")
    p.add_argument("--input-csv", help="CSV with columns x,y,vx,vy; if omitted, fresh data is simulated")
    p.add_argument("--force", choices=sorted(FORCES),
                   help="force type (default: the checkpoint's recorded training force)")
    p.add_argument("--k", type=float, help="restoring: spring constant (> 0)")
    p.add_argument("--center-x", type=float, help="restoring: centre x")
    p.add_argument("--center-y", type=float, help="restoring: centre y")
    p.add_argument("--fx", type=float, help="constant: force x")
    p.add_argument("--fy", type=float, help="constant: force y")
    p.add_argument("--dt", type=float, help=f"generated only; default {DEFAULTS['dt']}")
    p.add_argument("--n-steps", type=int, help=f"generated only; default {DEFAULTS['n_steps']}")
    p.add_argument("--n-particles", type=int, help=f"generated only; default {DEFAULTS['n_particles']}")
    p.add_argument("--seed", type=int, help=f"generated only; default {DEFAULTS['seed']}")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="results folder (default %(default)s)")
    p.add_argument("--out", help="exact results file path (refuses to overwrite unless --overwrite)")
    p.add_argument("--overwrite", action="store_true", help="allow replacing --out")
    return p


def force_from_args(a, ck_force=None):
    """Build a force spec from CLI args; unspecified parameters of the same force type come from the checkpoint."""
    name = a.force or (ck_force or {}).get("name")
    if name is None:
        raise ValueError("--force is required (the checkpoint does not record its training force)")
    spec = {"name": name}
    if ck_force and ck_force.get("name") == name and a.force is None:
        spec.update(ck_force)
    given = {"k": a.k, "center_x": a.center_x, "center_y": a.center_y, "fx": a.fx, "fy": a.fy}
    spec.update({k: v for k, v in given.items() if v is not None})
    return spec


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        from checkpoint import load_checkpoint
        ck = load_checkpoint(a.checkpoint)
        res = evaluate_checkpoint(ck, force_from_args(a, ck.force), a.dt, a.n_steps, a.n_particles, a.seed,
                                  a.input_csv, a.out_dir, a.out, a.overwrite)
    except (ValueError, OSError) as e:        # CheckpointError, CSVError, FileNotFoundError, FileExistsError
        print(f"error: {e}", file=sys.stderr)
        return 2
    m = res["metrics"]
    print(f"source={res['test_data']['source']}  force={res['force']}  n={m['n_samples']}")
    print(f"accel MSE={m['accel_mse']:.6g}  MAE={m['accel_mae']:.6g}  RMSE={m['accel_rmse']:.6g}  "
          f"model_unchanged={res['model_unchanged']}")
    print(f"results: {res['results_path']}")
    for s in res["limitations"]:
        print(f"  note: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
