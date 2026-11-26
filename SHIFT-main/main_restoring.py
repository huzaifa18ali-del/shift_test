import matplotlib
matplotlib.use("Agg")
from simulator import init_state
from visualize import compare_k_plot

if __name__ == "__main__":
    compare_k_plot(init_state(5, seed=0), [1.0, 9.0], dt=0.02, n_steps=400, path="compare_k.png")
    print("saved compare_k.png")