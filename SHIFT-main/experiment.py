"""End-to-end baseline experiment: linear regression vs MLP on the restoring force.

Usage: python experiment.py [--out outputs/baseline] [--epochs 100] ...
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, sklearn
from sklearn.linear_model import LinearRegression

from simulator import restoring_force, generate_dataset, make_force, validate_force_spec
from train import TrainConfig, Normalizer, train_mlp, predict_mlp, set_seed
from evaluate import evaluate
from config import ExperimentConfig

SEEDS = {"train": 0, "val": 1, "test": 2}          # data-generation seeds (distinct)
SIZES = {"train": 128, "val": 32, "test": 32}      # whole trajectories per split
N_STEPS, DT = 200, 0.01
PHYS = {"k": 1.0, "center_x": 0.0, "center_y": 0.0}


def resolve_force(phys=PHYS, force=None):
    """-> (phys, force_spec). A restoring spec is folded into the legacy phys dict; None keeps the default law."""
    if force is None:
        return phys, None
    force = validate_force_spec(force)
    if force["name"] == "restoring":
        return {"k": force["k"], "center_x": force.get("center_x", 0.0), "center_y": force.get("center_y", 0.0)}, None
    return phys, force


def validate_params(phys=PHYS, dt=DT, n_steps=N_STEPS, sizes=SIZES, seeds=SEEDS, cfg=None, force=None):
    """Raise ValueError on invalid baseline-experiment parameters."""
    import adaptation   # reuse the physics/size checks of Stage 1A (budget check is trivially satisfied)
    phys, force = resolve_force(phys, force)
    adaptation.validate_params(phys["k"], phys["k"], (phys["center_x"], phys["center_y"]), dt, n_steps,
                               sizes, (1,), None, force, force)
    vals = [seeds[s] for s in ("train", "val", "test")]
    if any(int(v) != v or v < 0 for v in vals) or len(set(vals)) != 3:
        raise ValueError(f"data seeds must be three distinct integers >= 0, got {seeds}")
    if cfg is not None:
        if int(cfg.hidden) != cfg.hidden or cfg.hidden < 1: raise ValueError(f"hidden must be an integer >= 1, got {cfg.hidden}")
        if not (cfg.lr > 0 and cfg.lr < float("inf")): raise ValueError(f"lr must be finite and > 0, got {cfg.lr}")
        if int(cfg.batch_size) != cfg.batch_size or cfg.batch_size < 1: raise ValueError(f"batch_size must be an integer >= 1, got {cfg.batch_size}")
        if int(cfg.epochs) != cfg.epochs or cfg.epochs < 1: raise ValueError(f"epochs must be an integer >= 1, got {cfg.epochs}")
        if int(cfg.seed) != cfg.seed or cfg.seed < 0: raise ValueError(f"model seed must be an integer >= 0, got {cfg.seed}")


def check_disjoint(d):
    """No two splits may share an initial state (trajectories are never shared across splits)."""
    rows = {k: {tuple(r) for r in v["traj"][0]} for k, v in d.items()}
    names = list(rows)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (rows[names[i]] & rows[names[j]]), f"split overlap: {names[i]} vs {names[j]}"


def check_overwrite(out_dir, is_default, overwrite):
    """Non-default runs may not silently replace an existing metrics.json (default runs keep old behaviour)."""
    if not is_default and not overwrite and os.path.exists(os.path.join(out_dir, "metrics.json")):
        raise FileExistsError(f"{out_dir} already contains metrics.json; choose another --out or pass --overwrite")


def make_splits(seeds=SEEDS, sizes=SIZES, n_steps=N_STEPS, dt=DT, phys=PHYS, force=None):
    phys, force = resolve_force(phys, force)
    f = restoring_force(**phys) if force is None else make_force(force)
    out = {}
    for s in seeds:
        traj, (cur, nxt, acc) = generate_dataset(f, sizes[s], n_steps, dt, seeds[s])
        out[s] = {"traj": traj, "cur": cur, "nxt": nxt, "acc": acc}
    return out


def run_experiment(cfg, out_dir, seeds=SEEDS, sizes=SIZES, n_steps=N_STEPS, dt=DT, phys=PHYS, force=None):
    validate_params(phys, dt, n_steps, sizes, seeds, cfg, force)
    phys, force = resolve_force(phys, force)
    set_seed(cfg.seed)
    d = make_splits(seeds, sizes, n_steps, dt, phys, force)
    # trajectories never shared across splits: seeds differ => initial states differ
    check_disjoint(d)

    x_norm = Normalizer().fit(d["train"]["cur"])       # train statistics only
    y_norm = Normalizer().fit(d["train"]["acc"])

    lin = LinearRegression().fit(d["train"]["cur"], d["train"]["acc"])   # raw units, train only
    model, hist = train_mlp(d["train"]["cur"], d["train"]["acc"],
                            d["val"]["cur"], d["val"]["acc"], cfg, x_norm, y_norm)

    te = d["test"]; f = restoring_force(**phys) if force is None else make_force(force)
    preds = {
        "linear_regression": lin.predict(te["cur"]),
        "mlp": predict_mlp(model, te["cur"], x_norm, y_norm, cfg.device),
        "analytical_oracle (simulator reference, not a learned model)": f(te["cur"]),
    }
    metrics = {}
    for name, p in preds.items():
        metrics[name] = {"test": evaluate(te["cur"], te["nxt"], te["acc"], p, dt)}
    for name, p in (("linear_regression", lin.predict(d["val"]["cur"])),
                    ("mlp", predict_mlp(model, d["val"]["cur"], x_norm, y_norm, cfg.device))):
        metrics[name]["val"] = evaluate(d["val"]["cur"], d["val"]["nxt"], d["val"]["acc"], p, dt)
    metrics["linear_coefficients"] = {"coef": lin.coef_.tolist(), "intercept": lin.intercept_.tolist()}

    record = {
        "data_seeds": seeds, "trajectories_per_split": sizes, "n_steps": n_steps, "dt": dt,
        "physics": phys if force is None else force, "mlp_config": cfg.to_dict(),
        "versions": {"torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "final_train_loss_normalised": hist["train_loss"][-1],
        "final_val_loss_normalised": hist["val_loss"][-1],
        "metrics": metrics,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    ExperimentConfig.from_baseline(cfg, seeds, sizes, n_steps, dt, phys, force).save(out_dir)   # effective config

    law = (f"k={phys['k']:g}, centre=({phys['center_x']:g}, {phys['center_y']:g})" if force is None
           else f"constant force fx={force['fx']:g}, fy={force['fy']:g}")
    label = f"{law}, dt={dt:g}, n_steps={n_steps}"
    plt.figure(figsize=(6, 4))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    plt.plot(ep, hist["train_loss"], label="train"); plt.plot(ep, hist["val_loss"], label="validation")
    plt.title(label, fontsize=9); plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE (normalised acceleration)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=120); plt.close()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(te["acc"]), size=min(2000, len(te["acc"])), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for j, comp in enumerate(["ax", "ay"]):
        ax = axes[j]
        ax.scatter(te["acc"][idx, j], preds["linear_regression"][idx, j], s=6, alpha=0.5, label="linear regression")
        ax.scatter(te["acc"][idx, j], preds["mlp"][idx, j], s=6, alpha=0.5, label="MLP")
        lim = [te["acc"][:, j].min(), te["acc"][:, j].max()]
        ax.plot(lim, lim, "k--", lw=1, label="ideal (y = x)")
        ax.set_xlabel(f"true {comp}"); ax.set_ylabel(f"predicted {comp}"); ax.legend()
    fig.suptitle(label, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "prediction_comparison.png"), dpi=120); plt.close()
    return record


def run_from_config(ec, out_dir):
    """Run the baseline experiment described by an ExperimentConfig (validated first)."""
    ec.validate("baseline")
    phys, force = ec.baseline_args()
    return run_experiment(ec.train_config(), out_dir, ec.seeds(), ec.sizes(), ec.simulation.n_steps,
                          ec.simulation.dt, phys, force)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("outputs", "baseline"))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="model seed")
    p.add_argument("--device", default="cpu")
    p.add_argument("--k", type=float, default=PHYS["k"])
    p.add_argument("--center-x", type=float, default=PHYS["center_x"])
    p.add_argument("--center-y", type=float, default=PHYS["center_y"])
    p.add_argument("--force", choices=["restoring", "constant"], default="restoring")
    p.add_argument("--fx", type=float, default=None, help="constant force x-component")
    p.add_argument("--fy", type=float, default=None, help="constant force y-component")
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--n-train", type=int, default=SIZES["train"])
    p.add_argument("--n-val", type=int, default=SIZES["val"])
    p.add_argument("--n-test", type=int, default=SIZES["test"])
    p.add_argument("--seed-train", type=int, default=SEEDS["train"], help="data seed")
    p.add_argument("--seed-val", type=int, default=SEEDS["val"])
    p.add_argument("--seed-test", type=int, default=SEEDS["test"])
    p.add_argument("--overwrite", action="store_true", help="allow a non-default run to replace an existing metrics.json")
    a = p.parse_args(argv)
    cfg = TrainConfig(a.hidden, a.lr, a.batch_size, a.epochs, a.seed, a.device)
    phys = {"k": a.k, "center_x": a.center_x, "center_y": a.center_y}
    sizes = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    seeds = {"train": a.seed_train, "val": a.seed_val, "test": a.seed_test}
    force = None
    if a.force == "constant":
        if a.fx is None or a.fy is None:
            p.error("--force constant requires --fx and --fy")
        if (a.k, a.center_x, a.center_y) != (PHYS["k"], PHYS["center_x"], PHYS["center_y"]):
            p.error("--k/--center-x/--center-y are not used with --force constant")
        force = {"name": "constant", "fx": a.fx, "fy": a.fy}
    elif a.fx is not None or a.fy is not None:
        p.error("--fx/--fy require --force constant")
    is_default = all(getattr(a, k) == p.get_default(k) for k in vars(a) if k not in ("out", "overwrite"))
    try:
        ExperimentConfig.from_baseline(cfg, seeds, sizes, a.n_steps, a.dt, phys, force).validate("baseline")
        check_overwrite(a.out, is_default, a.overwrite)
    except (ValueError, FileExistsError) as e:
        p.error(str(e))
    rec = run_experiment(cfg, a.out, seeds, sizes, a.n_steps, a.dt, phys, force)
    print(json.dumps(rec["metrics"], indent=2))
    return rec


if __name__ == "__main__":
    main()