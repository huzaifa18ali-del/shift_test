import matplotlib
from simulator import init_state, constant_force, run
from visualize import plot_trajectories

if __name__ == "__main__":
    matplotlib.use("Agg")  # remove this line to open an interactive window
    f = constant_force(0.0, -0.5)
    traj = run(init_state(10, seed=0), f, dt=0.05, n_steps=100)
    plot_trajectories(traj, "trajectories.png")
    print("saved trajectories.png", traj.shape)