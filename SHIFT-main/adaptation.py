"""Task 4: adaptation after a force-law change (k_old=1.0 -> k_new=9.0).

Design (see README): old-world pretraining; new-world data from separate seeds; budgets are nested
prefixes of one 128-trajectory new-world pool; every learned MLP keeps the ORIGINAL (old-world,
train-fitted) normalisers so fine-tuned and fresh MLPs are comparable. Validation = new-world val
set only (checkpoint selection); new-world test is used for final evaluation only.
Usage: python adaptation.py [--seeds 0 1 2] [--epochs 100] [--smoke] [--out outputs/adaptation]
"""
import argparse, json, math, os, platform
import numpy as np
import torch, sklearn
from sklearn.linear_model import LinearRegression

from simulator import restoring_force, generate_dataset, export_transitions, make_force, validate_force_spec
from train import TrainConfig, Normalizer, train_mlp, predict_mlp, params_fingerprint
from evaluate import evaluate, relative_improvement

K_OLD, K_NEW = 1.0, 9.0
CENTER = (0.0, 0.0)
DT, N_STEPS = 0.01, 200
BUDGETS = (8, 16, 32, 64, 128)
SIZES = {"train": 128, "val": 32, "test": 32}
METRICS = ("accel_mse", "accel_mae", "next_state_mse")


# ---------- data ----------
def seed_plan(s):
    """Six distinct data seeds per experiment seed s (s=0 -> old world 0/1/2 as in the baseline)."""
    b = 100 * s
    return {"old": {"train": b, "val": b + 1, "test": b + 2},
            "new": {"train": b + 10, "val": b + 11, "test": b + 12}}


def force_for(k, center=CENTER, spec=None):
    """Restoring force from (k, center) by default; a registry spec (non-restoring) overrides it."""
    return restoring_force(k, *center) if spec is None else make_force(spec)


def validate_params(k_old=K_OLD, k_new=K_NEW, center=CENTER, dt=DT, n_steps=N_STEPS,
                    sizes=SIZES, budgets=BUDGETS, seeds=None, force_old=None, force_new=None):
    """Raise ValueError with a clear message on invalid experiment parameters."""
    for name, k, spec in (("k_old", k_old, force_old), ("k_new", k_new, force_new)):
        if spec is not None:      # world defined by a registry force: k is not used
            if validate_force_spec(spec)["name"] == "restoring":
                raise ValueError("restoring worlds are configured with k/center, not a force spec")
        elif not math.isfinite(k) or k <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {k}")
    if len(center) != 2 or not all(math.isfinite(c) for c in center):
        raise ValueError(f"center must be two finite numbers, got {center}")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and > 0, got {dt}")
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps}")
    for split in ("train", "val", "test"):
        if int(sizes[split]) != sizes[split] or sizes[split] < 1:
            raise ValueError(f"size of '{split}' split must be an integer >= 1, got {sizes[split]}")
    budgets = list(budgets)
    if not budgets or any(int(b) != b or b < 1 for b in budgets):
        raise ValueError(f"budgets must be a non-empty list of integers >= 1, got {budgets}")
    if len(set(budgets)) != len(budgets):
        raise ValueError(f"budgets must be unique, got {budgets}")
    if max(budgets) > sizes["train"]:
        raise ValueError(f"budget {max(budgets)} exceeds available new-world training trajectories ({sizes['train']})")
    if seeds is not None:
        seeds = list(seeds)
        if not seeds or any(int(x) != x or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"seeds must be a non-empty list of unique integers >= 0, got {seeds}")


def make_world(k, seeds, sizes=SIZES, n_steps=N_STEPS, dt=DT, center=CENTER, force=None):
    """{split: {traj (T+1,N,4), cur, nxt, acc}}; each split is whole trajectories from its own seed."""
    f, out = force_for(k, center, force), {}
    for split, seed in seeds.items():
        traj, (cur, nxt, acc) = generate_dataset(f, sizes[split], n_steps, dt, seed)
        out[split] = {"traj": traj, "cur": cur, "nxt": nxt, "acc": acc}
    return out


def share_initial_states(traj_a, traj_b):
    a = {tuple(r) for r in traj_a[0]}
    return any(tuple(r) in a for r in traj_b[0])


def assert_disjoint(old, new):
    named = [(f"old_{k}", v["traj"]) for k, v in old.items()] + [(f"new_{k}", v["traj"]) for k, v in new.items()]
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            if share_initial_states(named[i][1], named[j][1]):
                raise AssertionError(f"trajectories overlap: {named[i][0]} vs {named[j][0]}")


def budget_transitions(train_split, b, k=K_NEW, center=CENTER, force=None):
    """First b whole trajectories of the new-world pool -> transitions (nested budgets)."""
    if b > train_split["traj"].shape[1]:
        raise ValueError("budget exceeds available trajectories")
    traj = train_split["traj"][:, :b]
    cur, nxt, acc = export_transitions(traj, force_for(k, center, force))
    return {"traj": traj, "cur": cur, "nxt": nxt, "acc": acc}


# ---------- evaluation ----------
def score(split, acc_pred, dt=DT):
    return evaluate(split["cur"], split["nxt"], split["acc"], acc_pred, dt)


def mlp_scores(model, split, xn, yn, dt=DT):
    return score(split, predict_mlp(model, split["cur"], xn, yn), dt)


def improvement(unchanged, current):
    return {m: relative_improvement(unchanged[m], current[m]) for m in METRICS}


# ---------- one experiment seed ----------
def run_seed(seed, budgets=BUDGETS, epochs=100, pretrain_epochs=None, hidden=64, lr=1e-3,
             batch_size=256, sizes=SIZES, n_steps=N_STEPS, k_old=K_OLD, k_new=K_NEW, center=CENTER, dt=DT,
             force_old=None, force_new=None):
    validate_params(k_old, k_new, center, dt, n_steps, sizes, budgets, None, force_old, force_new)
    plan = seed_plan(seed)
    old = make_world(k_old, plan["old"], sizes, n_steps, dt, center, force_old)
    new = make_world(k_new, plan["new"], sizes, n_steps, dt, center, force_new)
    if old["train"]["acc"].std(axis=0).min() < 1e-8:
        raise ValueError("old-world acceleration has zero variance (e.g. a constant force): the old-world-fitted "
                         "normalisers would be degenerate. Use a restoring force for the old world.")
    assert_disjoint(old, new)

    xn, yn = Normalizer().fit(old["train"]["cur"]), Normalizer().fit(old["train"]["acc"])  # old-world train only
    cfg_pre = TrainConfig(hidden, lr, batch_size, pretrain_epochs or epochs, seed)
    cfg_ad = TrainConfig(hidden, lr, batch_size, epochs, seed)
    pre, pre_hist = train_mlp(old["train"]["cur"], old["train"]["acc"], old["val"]["cur"], old["val"]["acc"],
                              cfg_pre, xn, yn)  # baseline policy: final-epoch weights
    pre_fp = params_fingerprint(pre)
    lin_old = LinearRegression().fit(old["train"]["cur"], old["train"]["acc"])

    unch = {"mlp": {"new_test": mlp_scores(pre, new["test"], xn, yn, dt), "old_test": mlp_scores(pre, old["test"], xn, yn, dt)},
            "linear": {"new_test": score(new["test"], lin_old.predict(new["test"]["cur"]), dt),
                       "old_test": score(old["test"], lin_old.predict(old["test"]["cur"]), dt)}}
    rec = {"seed": seed, "data_seeds": plan, "k_old": k_old if force_old is None else None, "k_new": k_new if force_new is None else None,
           "pretrain": {"epochs": cfg_pre.epochs, "steps": pre_hist["steps"], "seed": seed,
                        "n_traj": sizes["train"], "n_trans": old["train"]["cur"].shape[0],
                        "checkpoint_fingerprint": pre_fp},
           "unchanged": unch,
           "oracle": {"new_test": score(new["test"], force_for(k_new, center, force_new)(new["test"]["cur"]), dt)},
           "budgets": {}}
    for key, spec in (("force_old", force_old), ("force_new", force_new)):
        if spec is not None:
            rec[key] = validate_force_spec(spec)

    for b in budgets:
        d = budget_transitions(new["train"], b, k_new, center, force_new)
        entry = {"n_traj": b, "n_trans": d["cur"].shape[0]}
        for name, init in (("finetune", pre), ("fresh", None)):
            m, h = train_mlp(d["cur"], d["acc"], new["val"]["cur"], new["val"]["acc"], cfg_ad, xn, yn,
                             init_model=init, select_best_val=True)
            nt, ot = mlp_scores(m, new["test"], xn, yn, dt), mlp_scores(m, old["test"], xn, yn, dt)
            before, after = unch["mlp"]["old_test"]["accel_mse"], ot["accel_mse"]
            entry[name] = {"new_test": nt, "old_test": ot, "improvement_vs_unchanged_mlp": improvement(unch["mlp"]["new_test"], nt),
                           "old_world_forgetting": {"before_mse": before, "after_mse": after, "abs_increase": after - before,
                                                    "ratio": None if before <= 1e-12 else after / before},
                           "seed": seed, "epochs": cfg_ad.epochs, "steps": h["steps"], "best_epoch": h["best_epoch"],
                           "best_val_loss_norm": min(h["val_loss"]), "init_fingerprint": h["init_fingerprint"]}
        lin = LinearRegression().fit(d["cur"], d["acc"])
        lnt = score(new["test"], lin.predict(new["test"]["cur"]), dt)
        entry["linear_refit"] = {"new_test": lnt, "old_test": score(old["test"], lin.predict(old["test"]["cur"]), dt),
                                 "improvement_vs_unchanged_linear": improvement(unch["linear"]["new_test"], lnt),
                                 "coef": lin.coef_.tolist(), "intercept": lin.intercept_.tolist()}
        rec["budgets"][str(b)] = entry
    return rec


# ---------- aggregation / io ----------
def aggregate(runs):
    """Recursive mean / sample-std (ddof=1) over seeds for numeric leaves; None if any value is None."""
    first = runs[0]
    if isinstance(first, dict):
        return {k: aggregate([r[k] for r in runs]) for k in first}
    if isinstance(first, (list, str)):
        return first if all(r == first for r in runs) else None
    if any(r is None for r in runs):
        return None
    a = np.array(runs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None}


def aggregate_runs(runs):
    keys = ("pretrain", "unchanged", "oracle", "budgets")
    return {k: aggregate([r[k] for r in runs]) for k in keys}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="experiment seeds (default 0 1 2)")
    p.add_argument("--repetitions", type=int, default=None, help="shorthand for --seeds 0..N-1")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--k-old", type=float, default=K_OLD)
    p.add_argument("--k-new", type=float, default=K_NEW)
    p.add_argument("--center-x", type=float, default=CENTER[0])
    p.add_argument("--center-y", type=float, default=CENTER[1])
    p.add_argument("--force-old", choices=["restoring", "constant"], default="restoring")
    p.add_argument("--force-new", choices=["restoring", "constant"], default="restoring")
    for side in ("old", "new"):
        p.add_argument(f"--fx-{side}", type=float, default=None, help=f"constant force x-component ({side} world)")
        p.add_argument(f"--fy-{side}", type=float, default=None, help=f"constant force y-component ({side} world)")
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--n-train", type=int, default=SIZES["train"], help="trajectories in each train pool")
    p.add_argument("--n-val", type=int, default=SIZES["val"])
    p.add_argument("--n-test", type=int, default=SIZES["test"])
    p.add_argument("--out", default=os.path.join("outputs", "adaptation"))
    p.add_argument("--smoke", action="store_true", help="1 seed, 3 epochs, separate output dir")
    a = p.parse_args(argv)
    if a.seeds is not None and a.repetitions is not None:
        p.error("use either --seeds or --repetitions, not both")
    if a.repetitions is not None:
        if a.repetitions < 1:
            p.error("--repetitions must be >= 1")
        a.seeds = list(range(a.repetitions))
    if a.seeds is None:
        a.seeds = [0, 1, 2]
    if a.smoke:
        a.seeds, a.epochs = [0], 3
        if a.out == os.path.join("outputs", "adaptation"):
            a.out = os.path.join("outputs", "adaptation_smoke")
    center = (a.center_x, a.center_y)
    specs = {}
    for side, kdef in (("old", K_OLD), ("new", K_NEW)):
        fx, fy = getattr(a, f"fx_{side}"), getattr(a, f"fy_{side}")
        if getattr(a, f"force_{side}") == "restoring":
            if fx is not None or fy is not None:
                p.error(f"--fx-{side}/--fy-{side} require --force-{side} constant")
            specs[side] = None
        else:
            if getattr(a, f"k_{side}") != kdef:
                p.error(f"--k-{side} is not used with --force-{side} constant")
            if fx is None or fy is None:
                p.error(f"--force-{side} constant requires --fx-{side} and --fy-{side}")
            specs[side] = {"name": "constant", "fx": fx, "fy": fy}
    if all(specs.values()) and center != CENTER:
        p.error("--center-x/--center-y are not used when both worlds use constant forces")
    sizes = {"train": a.n_train, "val": a.n_val, "test": a.n_test}
    try:
        validate_params(a.k_old, a.k_new, center, a.dt, a.n_steps, sizes, a.budgets, a.seeds,
                        specs["old"], specs["new"])
    except ValueError as e:
        p.error(str(e))
    runs = []
    for s in a.seeds:
        print(f"seed {s} ...", flush=True)
        try:
            runs.append(run_seed(s, tuple(a.budgets), a.epochs, None, a.hidden, a.lr, a.batch_size, sizes, a.n_steps,
                                 a.k_old, a.k_new, center, a.dt, specs["old"], specs["new"]))
        except ValueError as e:
            p.error(str(e))
    agg = aggregate_runs(runs)
    os.makedirs(a.out, exist_ok=True)
    meta = {"config": {"k_old": a.k_old, "k_new": a.k_new, "center": center, "dt": a.dt, "n_steps": a.n_steps,
                       "budgets": a.budgets, "split_sizes": sizes, "epochs": a.epochs, "hidden": a.hidden,
                       "lr": a.lr, "batch_size": a.batch_size, "seeds": a.seeds, "smoke": a.smoke,
                       "normalizer_policy": "original old-world train-fitted normalisers for ALL MLPs",
                       "checkpoint_selection": "lowest new-world VALIDATION loss (adapted/fresh MLPs); pretrain = final epoch",
                       "std": "sample std, ddof=1 across seeds",
                       "improvement": "(unchanged_err - method_err)/unchanged_err; null if unchanged_err<=1e-12"},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "sklearn": sklearn.__version__, "numpy": np.__version__},
            "per_seed": runs, "aggregate": agg}
    for side in ("old", "new"):
        if specs[side] is not None:
            meta["config"][f"force_{side}"] = specs[side]
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    from plot_adaptation import plot_all
    plot_all(agg, a.budgets, a.out, n_seeds=len(a.seeds),
             info={"k_old": a.k_old, "k_new": a.k_new, "n_steps": a.n_steps,
                   **{f"label_{sd}": f"constant(fx={specs[sd]['fx']:g}, fy={specs[sd]['fy']:g})"
                      for sd in ("old", "new") if specs[sd] is not None}})
    print("saved to", a.out)
    return meta


if __name__ == "__main__":
    main()