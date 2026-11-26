"""Normalization, seeding, config and MLP training loop."""
import copy
import hashlib
import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch

from models import MLP


@dataclass
class TrainConfig:
    hidden: int = 64
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    seed: int = 0
    device: str = "cpu"

    def to_dict(self):
        return asdict(self)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class Normalizer:
    """Per-column standardisation. fit() on TRAIN data only."""
    def __init__(self, eps=1e-8):
        self.eps = eps; self.mean = None; self.std = None

    def fit(self, a):
        self.mean = a.mean(axis=0)
        self.std = np.maximum(a.std(axis=0), self.eps)
        return self

    def transform(self, a):
        return (a - self.mean) / self.std

    def inverse(self, a):
        return a * self.std + self.mean


def params_fingerprint(model):
    """SHA-256 of all parameters/buffers; identifies an exact checkpoint."""
    h = hashlib.sha256()
    for p in model.state_dict().values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_mlp(x_tr, y_tr, x_va, y_va, cfg, x_norm, y_norm, init_model=None, select_best_val=False):
    """Train on normalised data (MSE, Adam, shuffled mini-batches).

    Default (init_model=None, select_best_val=False) is the original behaviour: fresh
    model, final-epoch weights returned. Task 4 options:
      init_model      : start from a DEEP COPY of this model (the argument is never mutated).
      select_best_val : return the epoch with the lowest VALIDATION loss (never test data).
    Returns (model, history). Losses are in NORMALISED target units.
    """
    set_seed(cfg.seed)
    dev = torch.device(cfg.device)
    xt = torch.tensor(x_norm.transform(x_tr), dtype=torch.float32, device=dev)
    yt = torch.tensor(y_norm.transform(y_tr), dtype=torch.float32, device=dev)
    xv = torch.tensor(x_norm.transform(x_va), dtype=torch.float32, device=dev)
    yv = torch.tensor(y_norm.transform(y_va), dtype=torch.float32, device=dev)

    if init_model is None:
        model = MLP(x_tr.shape[1], cfg.hidden, y_tr.shape[1]).to(dev)
    else:
        model = copy.deepcopy(init_model).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(cfg.seed)
    n = xt.shape[0]
    hist = {"train_loss": [], "val_loss": [], "init_fingerprint": params_fingerprint(model),
            "steps": cfg.epochs * math.ceil(n / cfg.batch_size)}
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward(); opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(xv), yv).item()
        hist["val_loss"].append(v)
        hist["train_loss"].append(total / n)
        if select_best_val and v < best_val:
            best_val, best_epoch = v, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    if select_best_val:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch if select_best_val else cfg.epochs
    return model, hist


def predict_mlp(model, x, x_norm, y_norm, device="cpu"):
    """Raw states (B,4) -> acceleration (B,2) in ORIGINAL units."""
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_norm.transform(x), dtype=torch.float32, device=device)
        out = model(xt).cpu().numpy().astype(np.float64)
    return y_norm.inverse(out)