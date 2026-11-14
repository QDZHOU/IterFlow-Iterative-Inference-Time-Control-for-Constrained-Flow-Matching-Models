# IterFlow

Iterative Inference-Time Control for Constrained Flow-Matching Models — Maze2D.

The repository follows the original HardFlow layout. It contains native
**HardFlow** and the **IterFlow** history-ball extension. Independent seed
experiments, Projection, Gradient/OC-Flow baselines, No-ball experiments and
plotting utilities are excluded.

## Structure

```text
hardflow/
  config/                 model and dataset configuration
  datasets/               dataset loading and normalization
  envs/                   native MuJoCo Maze2D
  models_flow/
    unet.py               pretrained flow architecture
    flow_matcher.py       flow-matching training objective
    flow_policy.py        native HardFlow surrogate and sampler
    value_model.py        original task/constraint value model
    iterflow/             history balls, noise search and tail optimization
  utils/                  tensor utilities
run/
  train.py                flow model training
  eval.py                 HardFlow / IterFlow evaluation entry
  fit_dynamics.py         linear dynamics fitting
  prepare_maze2d_native.py compact-data dynamics fitting
configs/
  maze2d.json              one runnable default scene
  search.json             IterFlow search settings
run_scripts/              minimal training/evaluation launchers
docs/method.md            algorithm equations
```

The `hardflow` package name is retained for upstream compatibility, including
saved normalizers. The new method is named **IterFlow**.

## Setup

Use Python 3.10+ and an editable installation from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional training / dynamics fitting dependencies:
pip install -e '.[train]'
```

Required local assets (excluded from Git):

- `logs/maze2d-large-v1/flow/H384_1e6steps/model_ema_20.pth`
- `logs/maze2d-large-v1/dynamics/linear_model.npz`
- `datasets/maze2d-large-sparse-v1.hdf5`

Place trusted assets at the paths above, or create local symbolic links to an
existing HardFlow checkout. Then verify their availability:

```bash
python -m run.eval doctor
```

On this machine the links already exist; `../HardFlow/.venv/bin/python` can be
used without installing another environment. The new repository's code is
imported, with only weights/data reused. The dynamics archive contains a pickled
normalizer; use a trusted archive.

## Run

```bash
# Native HardFlow baseline.
python -m run.eval run --method hardflow --output outputs/hardflow

# IterFlow, initialized by HardFlow.
python -m run.eval run --method iterflow --output outputs/iterflow --rounds 300

# Inspect progress or extend the requested total.
python -m run.eval status outputs/iterflow
python -m run.eval resume outputs/iterflow --rounds 500
```

Runs are foreground processes. Use a fresh output directory for every new run.
The `--seed` and `--search-seed` parameters are retained solely for reproducible
algorithm randomness; no seed sweep or seed-selection experiments are included.
`--threads` controls CPU threads. `--dry-run` prints configuration without loading
models. Results are numerical `.npz` / `.json` records; no figures are generated.

Default scene: start `(0.94875744, 2.93648809)`, goal `(7,9)`, fixed initial
velocity `(-0.01347715, 0.06358764)`. Edit `configs/maze2d.json` to change it.
