import os, sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from simulator import (init_state, restoring_force, constant_force, step, run,
                       export_transitions, generate_dataset)
from visualize import animate, compare_k_plot

C = (1.0, -2.0)


def test_points_toward_center():
    f = restoring_force(2.0, *C)
    s = np.array([[3.0, -2.0, 0, 0], [1.0, 1.0, 0, 0], [-1.0, -5.0, 0, 0]])
    a = f(s)
    to_center = np.array(C) - s[:, :2]
    assert np.all(np.sum(a * to_center, axis=1) > 0)
    assert np.allclose(a, 2.0 * to_center)


def test_zero_at_center():
    a = restoring_force(5.0, *C)(np.array([[C[0], C[1], 3.0, -4.0]]))
    assert np.allclose(a, 0.0)  # velocity does not matter


def test_k_scales_acceleration():
    s = np.array([[2.0, 1.0, 0, 0]])
    a1, a3 = restoring_force(1.0)(s), restoring_force(3.0)(s)
    assert np.allclose(a3, 3 * a1)


def test_depends_on_position_only():
    f = restoring_force(1.0)
    a = f(np.array([[1.0, 1.0, 0.0, 0.0]]))
    b = f(np.array([[1.0, 1.0, 9.0, 9.0]]))
    assert np.allclose(a, b) and not np.allclose(a, f(np.array([[2.0, 1.0, 0, 0]])))


def test_integration_hand_calculation():
    # p=(2,0), v=(0,1), k=4, center=0, dt=0.1: a=(-8,0); v'=(-0.8,1); p'=(2-0.08, 0.1)
    out = step(np.array([[2.0, 0.0, 0.0, 1.0]]), restoring_force(4.0), 0.1)
    assert np.allclose(out, [[1.92, 0.1, -0.8, 1.0]])


def test_constant_force_still_works():
    out = run(np.zeros((1, 4)), constant_force(2, 0), 0.1, 5)
    assert np.isclose(out[-1, 0, 2], 1.0)


def test_dataset_shapes_finite_and_targets():
    f = restoring_force(2.0, *C)
    traj, (cur, nxt, acc) = generate_dataset(f, n_traj=20, n_steps=50, dt=0.05, seed=3)
    assert traj.shape == (51, 20, 4)
    assert cur.shape == nxt.shape == (1000, 4) and acc.shape == (1000, 2)
    assert all(np.isfinite(x).all() for x in (traj, cur, nxt, acc))
    assert np.allclose(acc, -2.0 * (cur[:, :2] - np.array(C)))
    # transition consistency with the integrator
    assert np.allclose(nxt[:, 2:], cur[:, 2:] + acc * 0.05)


def test_reproducible_and_seed_sensitive():
    f = restoring_force(1.5)
    a = generate_dataset(f, 10, 20, 0.05, seed=7)[0]
    b = generate_dataset(f, 10, 20, 0.05, seed=7)[0]
    c = generate_dataset(f, 10, 20, 0.05, seed=8)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_oscillation_bounded_for_stable_dt():
    # semi-implicit Euler is stable for dt*sqrt(k) < 2; amplitude must not blow up
    k, dt = 4.0, 0.05
    traj = run(init_state(5, 0), restoring_force(k), dt, 2000)
    assert np.abs(traj).max() < 10.0


def test_animate_and_compare_plot_run(tmp_path):
    traj = run(init_state(3, 0), restoring_force(1.0), 0.05, 10)
    anim = animate(traj, path=str(tmp_path / "a.gif"))  # renders every frame
    assert (tmp_path / "a.gif").stat().st_size > 0
    fig = compare_k_plot(init_state(3, 0), [1.0, 4.0], 0.05, 20)
    assert len(fig.axes) == 2