import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from simulator import init_state, constant_force, step, run, export_transitions


def test_shape_and_finite():
    s = init_state(7, seed=1)
    assert s.shape == (7, 4)
    traj = run(s, constant_force(1, 1), 0.1, 20)
    assert traj.shape == (21, 7, 4) and np.isfinite(traj).all()


def test_zero_velocity_zero_force():
    s = np.array([[1.0, 2.0, 0.0, 0.0]])
    out = run(s, constant_force(0, 0), 0.1, 10)
    assert np.allclose(out, s)


def test_constant_velocity_zero_force():
    s = np.array([[0.0, 0.0, 2.0, -1.0]])
    out = run(s, constant_force(0, 0), 0.5, 4)
    assert np.allclose(out[-1], [4.0, -2.0, 2.0, -1.0])


def test_constant_force_from_rest():
    # v_n = n*a*dt ; x_n = a*dt^2 * n(n+1)/2  (semi-implicit Euler)
    a, dt, n = 2.0, 0.1, 5
    out = run(np.zeros((1, 4)), constant_force(a, 0), dt, n)
    assert np.isclose(out[-1, 0, 2], n * a * dt)
    assert np.isclose(out[-1, 0, 0], a * dt**2 * n * (n + 1) / 2)


def test_seed_reproducible():
    assert np.array_equal(init_state(5, 42), init_state(5, 42))
    assert not np.array_equal(init_state(5, 42), init_state(5, 43))


def test_hand_calculated_step():
    # x=1,y=2,vx=3,vy=4; F=(2,-4); dt=0.5 -> a=(2,-4); v'=(4,2); p'=(1+2, 2+1)=(3,3)
    out = step(np.array([[1.0, 2.0, 3.0, 4.0]]), constant_force(2, -4), 0.5)
    assert np.allclose(out, [[3.0, 3.0, 4.0, 2.0]])


def test_export_transitions():
    f = constant_force(1, 2)
    traj = run(init_state(3, 0), f, 0.1, 4)
    cur, nxt, acc = export_transitions(traj, f)
    assert cur.shape == nxt.shape == (12, 4) and acc.shape == (12, 2)
    assert np.allclose(acc, [1, 2]) and np.allclose(nxt[:3], traj[1])