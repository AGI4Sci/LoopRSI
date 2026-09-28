# Prototype report

## Implemented

- Strict compact-NPZ loading and deterministic artifact/held-out-target splits.
- Global-mean and batch-control baselines.
- CRPM control residual prediction with low-rank gene programs, target-level
  group shrinkage, and optional learned batch calibration.
- JSON metrics covering MSE and delta-sensitive behavior, with explicit
  comparison against the global mean.
- Bounded one-GPU smoke/full payloads and a separate user-facing rjob launcher.

## Validation status

Passed on the control node:

- `python3 -m py_compile train.py crpm/*.py tests/*.py`
- `bash -n scripts/*.sh`
- all three source-only contract functions via a dependency-free runner

`python3 -m pytest` was unavailable because pytest is not installed in the
control environment; the same test functions passed when called directly.
Training, Torch-importing runtime checks, smoke/full scripts, and rjob commands
were deliberately not run because prototype rjob permission is 0.

## Next runtime command

On a controller-provisioned one-H200 worker:

```bash
bash /data/zhangzhicheng/omni-ar/case03_vcc25/runs/e2e_vcc25_20260806/implementation/scripts/run_smoke.sh
```

Treat its numbers as engineering evidence only. Report honestly if CRPM's MSE
does not beat `global_mean_baseline_metrics.mse`.
