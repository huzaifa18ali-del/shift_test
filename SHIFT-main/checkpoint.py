"""Stage 4: save / load trained SHIFT MLPs without retraining.

A checkpoint is ONE file written with ``torch.save`` that holds only plain data (no pickled Python
objects): a ``state_dict`` of CPU tensors plus JSON-compatible dicts / lists / numbers / strings.
It is therefore loaded with ``torch.load(..., weights_only=True)`` and always onto the CPU.

Public API
----------
    save_checkpoint(path, model, x_norm, y_norm, *, force=None, simulation=None, train_config=None,
                    experiment_config=None, training_metadata=None, overwrite=False) -> path
    load_checkpoint(path) -> Checkpoint          (``.model``, ``.x_norm``, ``.y_norm``, ``.predict(x)`` ...)
    CheckpointError                              (ValueError subclass raised for any invalid checkpoint)

Example
-------
    save_checkpoint("run1.pt", model, x_norm, y_norm, force={"name": "restoring", "k": 1.0},
                    simulation={"dt": 0.01, "n_steps": 200}, train_config=cfg)
    ck = load_checkpoint("run1.pt")
    acc = ck.predict(states)          # raw states (B, 4) -> acceleration (B, 2), original units
"""
import json
import math
import os
import platform
from dataclasses import dataclass, field

import numpy as np
import torch

from models import MLP
from train import Normalizer, params_fingerprint, predict_mlp
from simulator import validate_force_spec

FORMAT_NAME = "shift-checkpoint"
FORMAT_VERSION = 1
MODEL_TYPE = "MLP"
_REQUIRED_KEYS = ("format", "format_version", "architecture", "state_dict", "normalizers")
_OPTIONAL_KEYS = ("fingerprint", "force", "simulation", "train_config", "experiment_config",
                  "training_metadata", "versions")


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be saved, or is invalid / incomplete / incompatible."""


@dataclass
class Checkpoint:
    """Everything restored from a checkpoint file. ``model`` is on the CPU and in eval mode."""
    model: torch.nn.Module
    x_norm: Normalizer
    y_norm: Normalizer
    architecture: dict
    fingerprint: str
    force: dict = None
    simulation: dict = None
    train_config: dict = None
    experiment_config: dict = None
    training_metadata: dict = None
    versions: dict = field(default_factory=dict)
    path: str = None

    def predict(self, x, device="cpu"):
        """Raw states (B, 4) -> acceleration (B, 2) in ORIGINAL units (same as train.predict_mlp)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.architecture["in_dim"]:
            raise ValueError(f"expected states of shape (B, {self.architecture['in_dim']}), got {x.shape}")
        if str(device) != "cpu":
            self.model.to(device)
        try:
            return predict_mlp(self.model, x, self.x_norm, self.y_norm, device)
        finally:
            if str(device) != "cpu":
                self.model.to("cpu")


# ---------------------------------------------------------------- helpers
def _plain(obj, name):
    """Return a JSON round-tripped copy (tuples -> lists, numpy scalars rejected); error if not plain."""
    if obj is None:
        return None
    try:
        return json.loads(json.dumps(obj))
    except (TypeError, ValueError) as e:
        raise CheckpointError(f"'{name}' must be JSON-serialisable plain data: {e}") from e


def _as_dict(obj, name):
    """Accept a dict, or an object with to_dict() (TrainConfig, ExperimentConfig)."""
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    if not isinstance(obj, dict):
        raise CheckpointError(f"'{name}' must be a dict or provide to_dict(), got {type(obj).__name__}")
    return _plain(obj, name)


def check_can_write(path, overwrite=False):
    """Raise FileExistsError if ``path`` exists and overwrite is False (never replace silently)."""
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"checkpoint {path} already exists; choose another path or pass overwrite=True")


def model_architecture(model):
    """Read the MLP architecture off a model instance: {'type','in_dim','hidden','out_dim'}."""
    if not isinstance(model, MLP):
        raise CheckpointError(f"only models.MLP can be checkpointed, got {type(model).__name__}")
    lins = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    if len(lins) != 3 or lins[0].out_features != lins[1].in_features or lins[1].out_features != lins[2].in_features:
        raise CheckpointError("model does not have the expected 3-layer MLP structure")
    return {"type": MODEL_TYPE, "in_dim": lins[0].in_features, "hidden": lins[0].out_features,
            "out_dim": lins[2].out_features}


def _normalizer_to_dict(norm, name, dim):
    if not isinstance(norm, Normalizer) or norm.mean is None or norm.std is None:
        raise CheckpointError(f"{name} must be a fitted Normalizer")
    mean, std = np.asarray(norm.mean, dtype=np.float64), np.asarray(norm.std, dtype=np.float64)
    if mean.shape != (dim,) or std.shape != (dim,):
        raise CheckpointError(f"{name} has shape mean{mean.shape}/std{std.shape} but the model needs ({dim},)")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        raise CheckpointError(f"{name} must have finite mean and finite std > 0")
    return {"mean": mean.tolist(), "std": std.tolist(), "eps": float(norm.eps)}


# ---------------------------------------------------------------- save
def save_checkpoint(path, model, x_norm, y_norm, *, force=None, simulation=None, train_config=None,
                    experiment_config=None, training_metadata=None, overwrite=False):
    """Write a checkpoint for a trained MLP and its normalisers; return ``path``.

    force             : force spec dict, e.g. {"name": "restoring", "k": 1.0} (validated via the registry).
    simulation        : e.g. {"dt": 0.01, "n_steps": 200}.
    train_config      : TrainConfig (or dict).     experiment_config: ExperimentConfig (or dict).
    training_metadata : free-form dict of plain values (final losses, best epoch, ...).
    Refuses to replace an existing file unless ``overwrite=True``; the write is atomic (temp file + rename).
    """
    path = os.fspath(path)
    check_can_write(path, overwrite)
    arch = model_architecture(model)
    if any(p.dtype != torch.float32 for p in model.state_dict().values()):
        raise CheckpointError("only float32 models can be checkpointed (SHIFT trains float32 MLPs)")
    if force is not None:
        try:
            force = validate_force_spec(force)
        except ValueError as e:
            raise CheckpointError(f"invalid force spec: {e}") from e
    sim = _plain(simulation, "simulation")
    if sim is not None and not isinstance(sim, dict):
        raise CheckpointError("'simulation' must be a dict")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {
        "format": FORMAT_NAME, "format_version": FORMAT_VERSION,
        "architecture": arch,
        "state_dict": state,
        "fingerprint": params_fingerprint(model),
        "normalizers": {"x": _normalizer_to_dict(x_norm, "x_norm", arch["in_dim"]),
                        "y": _normalizer_to_dict(y_norm, "y_norm", arch["out_dim"])},
        "force": force, "simulation": sim,
        "train_config": _as_dict(train_config, "train_config"),
        "experiment_config": _as_dict(experiment_config, "experiment_config"),
        "training_metadata": _plain(training_metadata, "training_metadata"),
        "versions": {"torch": str(torch.__version__), "numpy": str(np.__version__), "python": platform.python_version()},
    }
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        torch.save(payload, tmp)
        if os.path.exists(path) and not overwrite:      # created by someone else while we were writing
            raise FileExistsError(f"checkpoint {path} already exists; choose another path or pass overwrite=True")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


# ---------------------------------------------------------------- load / validate
def _pos_int(v, name):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise CheckpointError(f"architecture field '{name}' must be an integer >= 1, got {v!r}")
    return v


def _load_normalizer(d, name, dim):
    if not isinstance(d, dict):
        raise CheckpointError(f"normalizers['{name}'] must be a dict, got {type(d).__name__}")
    missing = [k for k in ("mean", "std") if k not in d]
    if missing:
        raise CheckpointError(f"normalizers['{name}'] is missing field(s) {missing}")
    try:
        mean, std = np.asarray(d["mean"], dtype=np.float64), np.asarray(d["std"], dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise CheckpointError(f"normalizers['{name}'] mean/std are not numeric: {e}") from e
    if mean.shape != (dim,) or std.shape != (dim,):
        raise CheckpointError(f"normalizers['{name}'] has mean{mean.shape}/std{std.shape} but the architecture "
                              f"needs ({dim},)")
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        raise CheckpointError(f"normalizers['{name}'] must have finite mean and finite std > 0")
    eps = d.get("eps", 1e-8)
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise CheckpointError(f"normalizers['{name}'] eps must be a finite number > 0, got {eps!r}")
    n = Normalizer(eps=float(eps))
    n.mean, n.std = mean, std
    return n


def load_checkpoint(path):
    """Load and validate a checkpoint; return a :class:`Checkpoint` (CPU model in eval mode).

    Raises FileNotFoundError if the file is missing and CheckpointError (a ValueError) for anything
    unreadable, incomplete, malformed or internally inconsistent. No retraining is involved.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise CheckpointError(f"{path} is not a readable SHIFT checkpoint ({type(e).__name__}: {e})") from e
    if not isinstance(raw, dict):
        raise CheckpointError(f"{path}: expected a checkpoint dict, got {type(raw).__name__}")
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise CheckpointError(f"{path}: checkpoint is missing required field(s) {missing}")
    if raw["format"] != FORMAT_NAME:
        raise CheckpointError(f"{path}: not a SHIFT checkpoint (format={raw['format']!r})")
    if raw["format_version"] != FORMAT_VERSION:
        raise CheckpointError(f"{path}: unsupported checkpoint format_version {raw['format_version']!r}; "
                              f"this code reads version {FORMAT_VERSION}")
    unknown = sorted(set(raw) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unknown:
        raise CheckpointError(f"{path}: unknown checkpoint field(s) {unknown}")

    arch = raw["architecture"]
    if not isinstance(arch, dict):
        raise CheckpointError("'architecture' must be a dict")
    miss = [k for k in ("type", "in_dim", "hidden", "out_dim") if k not in arch]
    if miss:
        raise CheckpointError(f"'architecture' is missing field(s) {miss}")
    if arch["type"] != MODEL_TYPE:
        raise CheckpointError(f"unsupported model type {arch['type']!r}; supported: {MODEL_TYPE!r}")
    dims = {k: _pos_int(arch[k], k) for k in ("in_dim", "hidden", "out_dim")}

    model = MLP(dims["in_dim"], dims["hidden"], dims["out_dim"])
    expected = model.state_dict()
    state = raw["state_dict"]
    if not isinstance(state, dict):
        raise CheckpointError("'state_dict' must be a dict of tensors")
    if set(state) != set(expected):
        raise CheckpointError(f"state_dict keys do not match the MLP architecture: missing "
                              f"{sorted(set(expected) - set(state))}, unexpected {sorted(set(state) - set(expected))}")
    for k, ref in expected.items():
        t = state[k]
        if not torch.is_tensor(t):
            raise CheckpointError(f"state_dict['{k}'] is not a tensor ({type(t).__name__})")
        if tuple(t.shape) != tuple(ref.shape):
            raise CheckpointError(f"state_dict['{k}'] has shape {tuple(t.shape)} but architecture {dims} "
                                  f"requires {tuple(ref.shape)}")
        if not t.is_floating_point():
            raise CheckpointError(f"state_dict['{k}'] must be a floating-point tensor, got {t.dtype}")
        if not bool(torch.isfinite(t).all()):
            raise CheckpointError(f"state_dict['{k}'] contains NaN or infinite values")
    model.load_state_dict({k: state[k].to("cpu") for k in expected}, strict=True)
    model.eval()

    norms = raw["normalizers"]
    if not isinstance(norms, dict) or any(k not in norms for k in ("x", "y")):
        raise CheckpointError("'normalizers' must be a dict with 'x' and 'y' entries")
    x_norm = _load_normalizer(norms["x"], "x", dims["in_dim"])
    y_norm = _load_normalizer(norms["y"], "y", dims["out_dim"])

    force = raw.get("force")
    if force is not None:
        try:
            force = validate_force_spec(force)
        except ValueError as e:
            raise CheckpointError(f"checkpoint contains an invalid force spec: {e}") from e
    for name in ("simulation", "train_config", "experiment_config", "training_metadata", "versions"):
        if raw.get(name) is not None and not isinstance(raw[name], dict):
            raise CheckpointError(f"'{name}' must be a dict or None, got {type(raw[name]).__name__}")

    fp = params_fingerprint(model)
    if raw.get("fingerprint") is not None and raw["fingerprint"] != fp:
        raise CheckpointError("weights do not match the stored fingerprint; the checkpoint is corrupted or was "
                              "modified after saving")
    return Checkpoint(model=model, x_norm=x_norm, y_norm=y_norm, architecture=dict(arch), fingerprint=fp,
                      force=force, simulation=raw.get("simulation"), train_config=raw.get("train_config"),
                      experiment_config=raw.get("experiment_config"),
                      training_metadata=raw.get("training_metadata"), versions=raw.get("versions") or {},
                      path=path)
