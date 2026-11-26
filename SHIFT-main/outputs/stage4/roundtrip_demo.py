"""Stage 4 round-trip demo. Run from the project root:  python outputs/stage4/roundtrip_demo.py
Trains a small MLP, saves a checkpoint to a temp dir, reloads it, and compares predictions."""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import experiment as ex
from checkpoint import load_checkpoint
from train import TrainConfig

OUT = os.path.dirname(os.path.abspath(__file__))
cfg = TrainConfig(hidden=64, lr=1e-3, batch_size=256, epochs=20, seed=0)
with tempfile.TemporaryDirectory() as tmp:
    ck_path = os.path.join(tmp, "demo.pt")
    rec = ex.run_experiment(cfg, os.path.join(tmp, "run"), checkpoint_path=ck_path)
    ck = load_checkpoint(ck_path)
    d = ex.make_splits()
    x = d["test"]["cur"]
    # original model's predictions are not kept by run_experiment, so retrain identically (same seed) for reference
    from train import Normalizer, train_mlp, predict_mlp
    xn, yn = Normalizer().fit(d["train"]["cur"]), Normalizer().fit(d["train"]["acc"])
    model, _ = train_mlp(d["train"]["cur"], d["train"]["acc"], d["val"]["cur"], d["val"]["acc"], cfg, xn, yn)
    orig, loaded = predict_mlp(model, x, xn, yn), ck.predict(x)
    res = {"n_points": int(len(x)), "max_abs_diff": float(np.abs(orig - loaded).max()),
           "allclose_rtol1e-6_atol1e-8": bool(np.allclose(orig, loaded, rtol=1e-6, atol=1e-8)),
           "normalizer_x_equal": bool(np.array_equal(xn.mean, ck.x_norm.mean) and np.array_equal(xn.std, ck.x_norm.std)),
           "normalizer_y_equal": bool(np.array_equal(yn.mean, ck.y_norm.mean) and np.array_equal(yn.std, ck.y_norm.std)),
           "force": ck.force, "simulation": ck.simulation, "architecture": ck.architecture}
json.dump(res, open(os.path.join(OUT, "roundtrip_result.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
