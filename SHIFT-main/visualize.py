"""Matplotlib visualisation: static trajectories or frame-by-frame animation."""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_trajectories(traj, path=None):
    fig, ax = plt.subplots()
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.8)
        ax.scatter(traj[-1, i, 0], traj[-1, i, 1], s=12)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    if path:
        fig.savefig(path, dpi=120)
    return fig


def animate(traj, path=None, interval=40):
    fig, ax = plt.subplots()
    lo, hi = traj[..., :2].min() - 0.1, traj[..., :2].max() + 0.1
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    pts = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=12)
    lines = [ax.plot([], [], lw=0.6)[0] for _ in range(traj.shape[1])]

    def update(t):
        pts.set_offsets(traj[t, :, :2])
        for i, ln in enumerate(lines):
            ln.set_data(traj[: t + 1, i, 0], traj[: t + 1, i, 1])
        return [pts, *lines]

    anim = FuncAnimation(fig, update, frames=len(traj), interval=interval, blit=False)
    if path:
        anim.save(path, writer="pillow")
    return anim


def compare_k_plot(state0, k_values, dt, n_steps, center=(0.0, 0.0), path=None):
    """Same initial states, one panel per k, to show how trajectories differ."""
    from simulator import restoring_force, run
    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 5), sharex=True, sharey=True)
    for ax, k in zip(axes, k_values):
        traj = run(state0, restoring_force(k, *center), dt, n_steps)
        for i in range(traj.shape[1]):
            ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.8)
            ax.scatter(traj[0, i, 0], traj[0, i, 1], s=14, marker="o")
        ax.scatter(*center, c="k", marker="x")
        ax.set_title(f"k = {k}"); ax.set_aspect("equal"); ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    if path:
        fig.savefig(path, dpi=120)
    return fig