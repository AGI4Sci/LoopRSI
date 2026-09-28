# VCC25 CRPM prototype

This directory contains a minimal implementation of Calibrated Residual
Perturbation Modeling (CRPM) for the supplied real-data compact artifact. It
predicts log-expression as a training-only batch control mean plus a low-rank
perturbation residual. Smooth group shrinkage suppresses weak gene programs.

The default artifact split is a random row-level split and therefore produces
**engineering evidence only**. `--split-strategy heldout_target` reconstructs a
deterministic whole-target holdout from the compact rows; this is a scientific
candidate split, not an official VCC25 score.

## Methods and ablations

- `--method global_mean`: required unconditional fallback baseline.
- `--method batch_control_mean`: batch-specific non-targeting-control baseline.
- `--method crpm`: batch control baseline plus target and optional batch weights
  over shared low-rank gene programs.
- `--shrinkage 0` ablates hierarchical program shrinkage.
- `--no-batch-calibration` ablates learned batch residual calibration.

Every run reports the selected method and both baselines in one JSON object,
including MSE, residual/delta MSE, mean-profile and delta Pearson correlations,
delta energy, and top-k delta overlap. `beats_global_mean_mse` makes a loss to
the mean baseline explicit.

## GPU smoke

From this directory on a one-H200 worker:

```bash
bash scripts/run_smoke.sh
```

The smoke uses 10 epochs and an 80-step cap. `scripts/run_full.sh` uses at most
20 epochs and a held-out-target split. The user-facing `submit_rjob.sh` mounts
the required GPFS tree; do not call it from a controller-managed rjob worker.

Direct equivalent:

```bash
python3 train.py --data-root data --dataset vcc25_smoke.npz \
  --method crpm --device cuda --epochs 10 --max-steps 80 \
  --metrics-out artifacts/smoke/crpm.json
```

No checkpoint is loaded implicitly. `--save-checkpoint` is output-only.

## Static validation

```bash
python3 -m py_compile train.py crpm/*.py tests/*.py
bash -n scripts/*.sh
python3 -m pytest -q tests/test_contract.py
```

These checks do not import Torch through the experiment entrypoint or execute
training. Runtime validation belongs on the GPU worker.
