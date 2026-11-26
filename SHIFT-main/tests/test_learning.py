import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import MLP
from train import Normalizer, TrainConfig, train_mlp, predict_mlp
from evaluate import mse, mae, predict_next_state, evaluate
from simulator import restoring_force, generate_dataset, step, init_state
from experiment import SEEDS


def test_mlp_output_shape():
    assert MLP(4, 16, 2)(torch.zeros(5, 4)).shape == (5, 2)


def test_normalizer_fit_on_train_only():
    tr = np.random.default_rng(0).normal(5, 2, (100, 4))
    other = tr + 100
    n = Normalizer().fit(tr)
    assert np.allclose(n.mean, tr.mean(0))
    assert np.allclose(n.transform(tr).mean(0), 0, atol=1e-9)
    assert not np.allclose(n.transform(other).mean(0), 0)      # stats not refit on other data
    assert np.allclose(n.inverse(n.transform(other)), other)


def test_normalizer_constant_column_safe():
    a = np.ones((10, 2)); assert np.isfinite(Normalizer().fit(a).transform(a)).all()


def test_split_seeds_distinct_initial_states():
    f = restoring_force(1.0)
    s = [generate_dataset(f, 8, 3, 0.01, SEEDS[k])[0][0] for k in ("train", "val", "test")]
    assert not np.array_equal(s[0], s[1]) and not np.array_equal(s[0], s[2]) and not np.array_equal(s[1], s[2])
    assert len(set(SEEDS.values())) == 3


def test_metrics_hand_values():
    a, b = np.array([[1., 2.]]), np.array([[2., 0.]])
    assert np.isclose(mse(a, b), 2.5) and np.isclose(mae(a, b), 1.5)


def test_next_state_matches_simulator_and_oracle_zero_error():
    f = restoring_force(1.0)
    cur = init_state(6, 0)
    nxt = step(cur, f, 0.01)
    assert np.allclose(predict_next_state(cur, f(cur), 0.01), nxt)
    m = evaluate(cur, nxt, f(cur), f(cur), 0.01)
    assert m["accel_mse"] == 0 and m["next_state_mse"] < 1e-24


def test_training_reproducible_and_runs():
    f = restoring_force(1.0)
    _, (c, _, a) = generate_dataset(f, 8, 10, 0.01, 0)
    xn, yn = Normalizer().fit(c), Normalizer().fit(a)
    cfg = TrainConfig(hidden=8, batch_size=16, epochs=2, seed=5)
    m1, h1 = train_mlp(c, a, c, a, cfg, xn, yn)
    m2, h2 = train_mlp(c, a, c, a, cfg, xn, yn)
    assert h1 == h2 and len(h1["train_loss"]) == 2
    assert np.array_equal(predict_mlp(m1, c, xn, yn), predict_mlp(m2, c, xn, yn))
    assert predict_mlp(m1, c, xn, yn).shape == a.shape