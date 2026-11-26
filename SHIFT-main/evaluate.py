"""Metrics and one-step state prediction (existing semi-implicit Euler)."""
import numpy as np


def mse(a, b): return float(np.mean((a - b) ** 2))
def mae(a, b): return float(np.mean(np.abs(a - b)))


def predict_next_state(cur, acc_pred, dt):
    """v' = v + a*dt ; p' = p + v'*dt. cur (B,4), acc_pred (B,2) -> (B,4)."""
    v = cur[:, 2:4] + acc_pred * dt
    p = cur[:, 0:2] + v * dt
    return np.hstack([p, v])


def evaluate(cur, nxt, acc_true, acc_pred, dt):
    return {
        "accel_mse": mse(acc_pred, acc_true),
        "accel_mae": mae(acc_pred, acc_true),
        "next_state_mse": mse(predict_next_state(cur, acc_pred, dt), nxt),
    }


def relative_improvement(baseline_err, new_err, eps=1e-12):
    """(baseline_err - new_err) / baseline_err; 1 = error eliminated, <0 = worse.

    Returns None (not inf/NaN) when baseline_err <= eps, where the ratio is undefined.
    """
    if baseline_err <= eps:
        return None
    return float((baseline_err - new_err) / baseline_err)