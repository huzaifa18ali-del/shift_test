"""Stage 3: unified, JSON-serialisable experiment configuration (stdlib dataclasses only).

Defaults equal the existing project defaults. Semantic validation is delegated to the existing
`experiment.validate_params` / `adaptation.validate_params` (imported lazily, so this module itself
does not need torch for construction / JSON I/O).
"""
import copy, json, math, os
from dataclasses import dataclass, field, asdict, fields

from simulator import validate_force_spec

CONFIG_FILENAME = "config.json"


def _restoring(k, cx=0.0, cy=0.0):
    return {"name": "restoring", "k": k, "center_x": cx, "center_y": cy}


@dataclass
class SimulationConfig:
    dt: float = 0.01
    n_steps: int = 200


@dataclass
class DatasetConfig:
    n_train: int = 128          # whole trajectories per split
    n_val: int = 32
    n_test: int = 32
    seed_train: int = 0         # data-generation seeds (baseline pipeline)
    seed_val: int = 1
    seed_test: int = 2


@dataclass
class ModelConfig:
    hidden: int = 64


@dataclass
class TrainingConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 256
    seed: int = 0               # model seed (baseline; adaptation uses its per-experiment seeds)
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    force: dict = field(default_factory=lambda: _restoring(1.0))        # baseline pipeline
    force_old: dict = field(default_factory=lambda: _restoring(1.0))    # adaptation: old world
    force_new: dict = field(default_factory=lambda: _restoring(9.0))    # adaptation: new world
    budgets: tuple = (8, 16, 32, 64, 128)                               # adaptation
    adaptation_seeds: tuple = (0, 1, 2)                                 # adaptation experiment seeds

    # ---------- serialisation ----------
    def to_dict(self):
        d = asdict(self)
        d["budgets"], d["adaptation_seeds"] = list(self.budgets), list(self.adaptation_seeds)
        return d

    @classmethod
    def from_dict(cls, d):
        """Missing keys take defaults; unknown keys / wrong container types raise ValueError."""
        if not isinstance(d, dict):
            raise ValueError(f"config must be a JSON object, got {type(d).__name__}")
        sections = {"simulation": SimulationConfig, "dataset": DatasetConfig,
                    "model": ModelConfig, "training": TrainingConfig}
        plain = {"force", "force_old", "force_new", "budgets", "adaptation_seeds"}
        unknown = sorted(set(d) - set(sections) - plain)
        if unknown:
            raise ValueError(f"unknown config key(s) {unknown}; allowed: {sorted(set(sections) | plain)}")
        kw = {}
        for name, klass in sections.items():
            if name in d:
                sub = d[name]
                if not isinstance(sub, dict):
                    raise ValueError(f"config section '{name}' must be an object, got {sub!r}")
                bad = sorted(set(sub) - {f.name for f in fields(klass)})
                if bad:
                    raise ValueError(f"unknown field(s) {bad} in '{name}'; allowed: {[f.name for f in fields(klass)]}")
                kw[name] = klass(**sub)
        for name in ("force", "force_old", "force_new"):
            if name in d:
                kw[name] = copy.deepcopy(d[name])
        for name in ("budgets", "adaptation_seeds"):
            if name in d:
                if not isinstance(d[name], (list, tuple)):
                    raise ValueError(f"'{name}' must be a list, got {d[name]!r}")
                kw[name] = tuple(d[name])
        return cls(**kw)

    def to_json(self, path=None):
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_json(cls, path_or_text):
        text = open(path_or_text).read() if os.path.exists(str(path_or_text)) else path_or_text
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid config JSON: {e}") from e

    def save(self, out_dir):
        """Write config.json into an experiment output directory; return its path."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, CONFIG_FILENAME)
        self.to_json(path)
        return path

    # ---------- bridges to the existing (unchanged) APIs ----------
    def sizes(self):
        d = self.dataset
        return {"train": d.n_train, "val": d.n_val, "test": d.n_test}

    def seeds(self):
        d = self.dataset
        return {"train": d.seed_train, "val": d.seed_val, "test": d.seed_test}

    def train_config(self):
        from train import TrainConfig
        t = self.training
        return TrainConfig(self.model.hidden, t.lr, t.batch_size, t.epochs, t.seed, t.device)

    def baseline_args(self):
        """-> (phys, force) as accepted by experiment.run_experiment (restoring -> phys, else spec)."""
        spec = validate_force_spec(self.force)
        if spec["name"] == "restoring":
            return {"k": spec["k"], "center_x": spec.get("center_x", 0.0), "center_y": spec.get("center_y", 0.0)}, None
        return {"k": 1.0, "center_x": 0.0, "center_y": 0.0}, spec

    def adaptation_args(self):
        """-> dict(k_old, k_new, center, force_old, force_new) as accepted by adaptation.run_seed."""
        out, centers = {}, []
        for side, default_k in (("old", 1.0), ("new", 9.0)):
            spec = validate_force_spec(getattr(self, f"force_{side}"))
            if spec["name"] == "restoring":
                out[f"k_{side}"], out[f"force_{side}"] = spec["k"], None
                centers.append((spec.get("center_x", 0.0), spec.get("center_y", 0.0)))
            else:
                out[f"k_{side}"], out[f"force_{side}"] = default_k, spec
        if len(set(centers)) > 1:
            raise ValueError(f"old and new restoring forces must share one centre in the adaptation pipeline, got {centers}")
        out["center"] = centers[0] if centers else (0.0, 0.0)
        return out

    # ---------- validation ----------
    def validate(self, pipeline="both"):
        """Raise ValueError on invalid values; semantic checks reuse the existing validators."""
        if pipeline not in ("baseline", "adaptation", "both"):
            raise ValueError(f"pipeline must be 'baseline', 'adaptation' or 'both', got {pipeline!r}")
        if not isinstance(self.training.device, str) or not self.training.device:
            raise ValueError(f"device must be a non-empty string, got {self.training.device!r}")
        for v in (self.simulation.dt, self.training.lr):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"dt and lr must be finite numbers, got {v!r}")
        sim = self.simulation
        if pipeline in ("baseline", "both"):
            import experiment
            phys, force = self.baseline_args()
            experiment.validate_params(phys, sim.dt, sim.n_steps, self.sizes(), self.seeds(), self.train_config(), force)
        if pipeline in ("adaptation", "both"):
            import adaptation, experiment
            a = self.adaptation_args()
            # model/training checks (hidden, lr, batch size, epochs, seed) live in experiment.validate_params
            experiment.validate_params(dt=sim.dt, n_steps=sim.n_steps, sizes=self.sizes(), seeds=self.seeds(),
                                       cfg=self.train_config())
            adaptation.validate_params(a["k_old"], a["k_new"], a["center"], sim.dt, sim.n_steps, self.sizes(),
                                       self.budgets, self.adaptation_seeds, a["force_old"], a["force_new"])
        return self

    # ---------- constructors from legacy-style arguments ----------
    @classmethod
    def from_baseline(cls, cfg, seeds, sizes, n_steps, dt, phys, force=None):
        return cls(
            SimulationConfig(dt, n_steps),
            DatasetConfig(sizes["train"], sizes["val"], sizes["test"], seeds["train"], seeds["val"], seeds["test"]),
            ModelConfig(cfg.hidden),
            TrainingConfig(cfg.epochs, cfg.lr, cfg.batch_size, cfg.seed, cfg.device),
            force=copy.deepcopy(force) if force is not None else _restoring(phys["k"], phys["center_x"], phys["center_y"]))

    @classmethod
    def from_adaptation(cls, seeds, budgets, epochs, hidden, lr, batch_size, sizes, n_steps, k_old, k_new,
                        center, dt, force_old=None, force_new=None):
        spec = lambda f, k: copy.deepcopy(f) if f is not None else _restoring(k, center[0], center[1])
        return cls(
            SimulationConfig(dt, n_steps),
            DatasetConfig(sizes["train"], sizes["val"], sizes["test"]),
            ModelConfig(hidden),
            TrainingConfig(epochs, lr, batch_size),
            force_old=spec(force_old, k_old), force_new=spec(force_new, k_new),
            budgets=tuple(budgets), adaptation_seeds=tuple(seeds))