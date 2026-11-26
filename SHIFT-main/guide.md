# SHIFT — Professor Testing Guide

This guide is for running and testing SHIFT locally from the project source folder. All commands below are run from the directory containing `experiment.py`, `adaptation.py`, and `evaluate.py` (the inner `SHIFT-main` directory in the supplied repository archive).

SHIFT is a 2D particle simulator with configurable force laws and a learned MLP dynamics model. The main experimental workflow compares baseline learning, adaptation by fine-tuning, training from scratch, and independent evaluation of saved checkpoints.

## 1. Open the project and prepare the environment

Open a terminal in the SHIFT source directory:

```bash
cd path/to/SHIFT-main
```

Create and activate a virtual environment (Ubuntu/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python3 -m venv` is unavailable, install the operating-system package that provides Python's `venv` module, then repeat the command.

Check the available command-line options:

```bash
python experiment.py --help
python adaptation.py --help
python evaluate.py --help
```

## 2. Run the automated tests

```bash
python -m pytest -q
```

The most recently verified local run reported **207 passed, 1 skipped**. Test counts can change if the code or dependencies change.

## 3. Run the simulator demonstrations

Constant-force particle simulation:

```bash
python main.py
```

This writes `trajectories.png` in the current directory.

Restoring-force comparison:

```bash
python main_restoring.py
```

This writes `compare_k.png`, comparing the same initial particles under two restoring-force strengths (`k=1` and `k=9`).

## 4. Run a baseline training experiment and save a checkpoint

This trains the baseline MLP, evaluates it against the baseline test split, writes metrics/plots/configuration, and saves a reusable checkpoint.

```bash
python experiment.py \
  --out outputs/professor/baseline_k1 \
  --save-checkpoint checkpoints/professor_baseline_k1.pt
```

Important:
- This command **trains** the MLP.
- The checkpoint contains the model weights, input/output normalisers, architecture, and recorded experiment metadata.
- Checkpoint files are intentionally excluded from Git by `.gitignore`.
- Saving refuses to replace an existing checkpoint unless `--overwrite-checkpoint` is supplied. Prefer a new filename for each distinct trained model.
- The output directory contains `metrics.json`, `config.json`, `training_curves.png`, and `prediction_comparison.png`.

To train on a different restoring-force strength, for example `k=5`, use a separate output directory and checkpoint filename:

```bash
python experiment.py \
  --k 5 \
  --out outputs/professor/baseline_k5 \
  --save-checkpoint checkpoints/professor_baseline_k5.pt
```

For a constant-force baseline, both force components are required:

```bash
python experiment.py \
  --force constant \
  --fx 2 \
  --fy -1 \
  --out outputs/professor/baseline_constant \
  --save-checkpoint checkpoints/professor_baseline_constant.pt
```

## 5. Independently evaluate a saved checkpoint (no retraining)

This is the Stage 5 evaluation interface. It loads an existing checkpoint, generates fresh test states by default, evaluates predictions against the selected force law, and saves a JSON result.

Example: evaluate a model trained in the `k=1` world in a `k=5` world:

```bash
python evaluate.py \
  --checkpoint checkpoints/professor_baseline_k1.pt \
  --force restoring \
  --k 5 \
  --dt 0.01 \
  --n-steps 200 \
  --n-particles 100 \
  --seed 42 \
  --out-dir outputs/professor/evaluation_k5
```

The command prints acceleration MSE, MAE, RMSE, the sample count, and whether the model remained unchanged. A successful independent evaluation reports `model_unchanged=True`. Evaluation does **not** call the training loop or update model weights.

Other restoring-force examples:

```bash
# Stronger restoring force
python evaluate.py --checkpoint checkpoints/professor_baseline_k1.pt \
  --force restoring --k 90 --out-dir outputs/professor/evaluation_k90

# Shift the restoring-force centre
python evaluate.py --checkpoint checkpoints/professor_baseline_k1.pt \
  --force restoring --k 5 --center-x 2 --center-y -1 \
  --out-dir outputs/professor/evaluation_k5_shifted
```

Constant-force evaluation:

```bash
python evaluate.py \
  --checkpoint checkpoints/professor_baseline_k1.pt \
  --force constant \
  --fx 5 \
  --fy -3 \
  --out-dir outputs/professor/evaluation_constant
```

For constant force, use `--fx` and `--fy`; `--k` and restoring-force centre options do not apply.

### Evaluate professor-provided CSV states

The CSV must contain these columns:

```text
x,y,vx,vy
0.5,0.2,0.0,0.0
-0.3,0.9,1.0,1.0
```

Save the file, for example as `data/professor_states.csv`, then run:

```bash
python evaluate.py \
  --checkpoint checkpoints/professor_baseline_k1.pt \
  --force restoring \
  --k 5 \
  --input-csv data/professor_states.csv \
  --out-dir outputs/professor/evaluation_csv
```

When `--input-csv` is supplied, the evaluator uses those rows as states. The generated-data options (`--dt`, `--n-steps`, `--n-particles`, `--seed`) are not used to create the test states.

Each evaluation writes a timestamped JSON file under the selected output directory. Use a new output directory for each run, or use `--out` with a new filename. The exact output path is printed in the terminal.

## 6. Run adaptation experiments

The adaptation experiment compares:
- an MLP pretrained in the old environment and left unchanged;
- fine-tuning that pretrained MLP using limited new-environment data;
- training a fresh MLP using the same new-environment data budgets;
- linear-regression reference results;
- performance on the new world and forgetting on the old world.

Default old/new restoring-force strengths are `k=1` and `k=9`. The default budgets are 8, 16, 32, 64, and 128 trajectories, and the default experiment seeds are 0, 1, and 2.

Run the full default adaptation experiment:

```bash
python adaptation.py --out outputs/professor/adaptation_k1_to_k9
```

For a smaller smoke test:

```bash
python adaptation.py --smoke --out outputs/professor/adaptation_smoke
```

The smoke test uses one seed and three epochs; it is for checking that the pipeline runs, not for drawing final research conclusions.

Change the old and new restoring-force strengths:

```bash
python adaptation.py \
  --k-old 1 \
  --k-new 5 \
  --budgets 8 16 32 64 \
  --seeds 0 1 2 \
  --out outputs/professor/adaptation_k1_to_k5
```

Adaptation supports separate old/new force selections. Constant-force components use `--fx-old`, `--fy-old`, `--fx-new`, and `--fy-new` with the corresponding `--force-old constant` or `--force-new constant` options. The old-world training data must have nonzero acceleration variance; a constant-force old world is therefore rejected by the current adaptation implementation. A constant-force new world can be used with a suitable restoring-force old world.

The adaptation output directory contains `metrics.json`, `config.json`, and plots generated by `plot_adaptation.py`.

## 7. Where results are saved

Unless a command specifies another location:

- Baseline experiment: `outputs/baseline/`
- Adaptation experiment: `outputs/adaptation/`
- Independent evaluation: `outputs/stage5/`

Prefer distinct output directories for separate experiments so earlier results are preserved. Baseline and adaptation runs write JSON metrics/configuration and plots. Independent evaluation writes a timestamped JSON file.

## 8. Important interpretation notes

- The restoring-force law is linear in position. Linear regression is therefore a strong baseline and can recover the ideal mapping almost exactly on this simulator. This is an expected property of the chosen force law, not evidence that the MLP is superior.
- Independent evaluation under a changed force law measures out-of-distribution performance. It is not itself an adaptation experiment.
- Adaptation experiments measure whether fine-tuning on limited new-world observations improves new-world performance compared with leaving the model unchanged or training a fresh model.
- Fine-tuning can increase error on the original world; this is measured as old-world forgetting.
- Evaluation metrics in `evaluate.py` are acceleration metrics on the selected test states. They do not by themselves establish long-horizon trajectory accuracy or generalisation to every possible environment.
- Generated evaluation states use a seed that is checked against the checkpoint's recorded train, validation, and test split seeds.

## 9. Useful command reference

```bash
python -m pytest -q
python main.py
python main_restoring.py
python experiment.py --help
python adaptation.py --help
python evaluate.py --help
```