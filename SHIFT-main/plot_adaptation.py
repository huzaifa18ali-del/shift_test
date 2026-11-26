"""Plots for the Task 4 adaptation experiment (inputs: aggregated metrics dict)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB = {"finetune": "fine-tuned MLP (from old-world checkpoint)", "fresh": "fresh MLP (random init)",
       "linear_refit": "linear regression refit on new data"}
TRAJ_X = "adaptation data: new-world trajectories (200 transitions each)"
DEFAULT_INFO = {"k_old": 1.0, "k_new": 9.0, "n_steps": 200}


def world_label(info, side):
    return info.get(f"label_{side}") or f"k={info[f'k_{side}']:g}"


def traj_x(info):
    return f"adaptation data: new-world trajectories ({info['n_steps']} transitions each)"


def _ms(node):
    return node["mean"], (node["std"] or 0.0)


def _series(agg, budgets, method, split, metric):
    m, s = zip(*[_ms(agg["budgets"][str(b)][method][split][metric]) for b in budgets])
    return np.array(m), np.array(s)


def _line(ax, x, m, s, label, **kw):
    ax.plot(x, m, marker="o", label=label, **kw)
    ax.fill_between(x, np.maximum(m - s, m * 0.01), m + s, alpha=0.2)


def _hline(ax, node, label, **kw):
    m, s = _ms(node)
    ax.axhline(m, label=label, **kw)


def adaptation_curve(agg, budgets, path, n_seeds, info=DEFAULT_INFO):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for meth in LAB:
        m, s = _series(agg, budgets, meth, "new_test", "accel_mse")
        _line(ax, budgets, m, s, LAB[meth])
    _hline(ax, agg["unchanged"]["mlp"]["new_test"]["accel_mse"], "unchanged MLP (no update)", color="k", ls="--")
    _hline(ax, agg["unchanged"]["linear"]["new_test"]["accel_mse"], "unchanged linear (no refit)", color="gray", ls=":")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(budgets); ax.set_xticklabels(budgets)
    ax.set_xlabel(traj_x(info)); ax.set_ylabel("new-world test acceleration MSE (log scale)")
    orc = agg["oracle"]["new_test"]["accel_mse"]["mean"]
    ax.set_title(f"Adaptation to {world_label(info, 'old')} -> {world_label(info, 'new')} (mean ± std, {n_seeds} seeds)\nanalytical oracle MSE = {orc:.1e} (reference, not shown)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def strategy_comparison(agg, budgets, path, n_seeds, info=DEFAULT_INFO):
    methods = [("unchanged MLP", None, "mlp"), ("fine-tuned MLP", "finetune", None), ("fresh MLP", "fresh", None),
               ("unchanged linear", None, "linear"), ("refit linear", "linear_refit", None)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    w = 0.16
    for ax, metric, ylab in ((axes[0], "accel_mse", "new-world test acceleration MSE"),
                             (axes[1], "next_state_mse", "new-world test one-step next-state MSE")):
        for i, (lab, meth, unch) in enumerate(methods):
            if meth is None:
                m0, s0 = _ms(agg["unchanged"][unch]["new_test"][metric]); m = np.full(len(budgets), m0); s = np.full(len(budgets), s0)
            else:
                m, s = _series(agg, budgets, meth, "new_test", metric)
            ax.bar(np.arange(len(budgets)) + (i - 2) * w, m, w, yerr=np.minimum(s, m * 0.99), capsize=2, label=lab)
        ax.set_yscale("log"); ax.set_xticks(range(len(budgets))); ax.set_xticklabels(budgets)
        ax.set_xlabel("adaptation trajectories"); ax.set_ylabel(ylab + " (log)"); ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].legend(fontsize=8)
    orc = agg["oracle"]["new_test"]
    fig.suptitle(f"Strategy comparison (mean ± std, {n_seeds} seeds). Analytical oracle (reference only): "
                 f"accel MSE={orc['accel_mse']['mean']:.1e}, next-state MSE={orc['next_state_mse']['mean']:.1e}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def forgetting_curve(agg, budgets, path, n_seeds, info=DEFAULT_INFO):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for meth in LAB:
        m, s = _series(agg, budgets, meth, "old_test", "accel_mse")
        _line(ax, budgets, m, s, LAB[meth] + " [after]")
    _hline(ax, agg["unchanged"]["mlp"]["old_test"]["accel_mse"], "pretrained MLP [before fine-tuning]", color="k", ls="--")
    _hline(ax, agg["unchanged"]["linear"]["old_test"]["accel_mse"], "original linear regression", color="gray", ls=":")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(budgets); ax.set_xticklabels(budgets)
    ax.set_xlabel(traj_x(info)); ax.set_ylabel(f"OLD-world ({world_label(info, 'old')}) test acceleration MSE (log scale)")
    ax.set_title(f"Forgetting: old-world error before vs after adaptation (mean ± std, {n_seeds} seeds)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_all(agg, budgets, out_dir, n_seeds, info=DEFAULT_INFO):
    os.makedirs(out_dir, exist_ok=True)
    adaptation_curve(agg, budgets, os.path.join(out_dir, "adaptation_curve.png"), n_seeds, info)
    strategy_comparison(agg, budgets, os.path.join(out_dir, "strategy_comparison.png"), n_seeds, info)
    forgetting_curve(agg, budgets, os.path.join(out_dir, "forgetting_curve.png"), n_seeds, info)