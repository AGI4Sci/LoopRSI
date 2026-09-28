# Round 0 suggestion report

## Outcome

No implementation code was changed. The validated suggestion's only priority is
the first GPU smoke execution, and `codex_tasks` is empty. Running that workload
on the CPU/control node would violate the controller and loop-engineering
guardrails, while submitting a GPU job is outside this apply step.

Static inspection found the existing payload ready for controller-managed
runtime validation:

- `scripts/run_smoke.sh` runs the experiment directly and does not submit an
  rjob or invoke proxy helpers.
- The smoke data defaults to the existing implementation-local
  `data/vcc25_smoke.npz` artifact.
- Metrics parent directories are created before output, and the two outputs are
  machine-readable JSON files.
- Checkpoint output uses the distinct `--save-checkpoint` flag; the smoke does
  not pass a checkpoint argument.

## Failure targeted

The next trial addresses the absence of runtime evidence for the baseline and
CRPM payload. It should establish whether the code trains on one H200 and emits
`artifacts/smoke/crpm.json` with `status: ok`, a non-empty `training_loss`, and a
boolean `beats_global_mean_mse`.

## Static checks run

- `bash -n` on `run_smoke.sh`, `submit_rjob.sh`, and `run_full.sh`: passed.
- `python3 -m py_compile` on the Python implementation and contract test:
  passed (without importing or executing the training payload).
- `python3 -m json.tool data/vcc25_smoke.json`: passed.
- `unzip -t data/vcc25_smoke.npz`: all array members passed integrity checks.
- Guardrail scan of `run_smoke.sh` and `train.py` for nested rjob submission,
  proxy helpers, and ambiguous `--checkpoint`: no matches.

No runtime, unit, smoke, torch-importing, training, or GPU checks were run on the
control node.

## Exact next controller-managed worker payload

```bash
set -euo pipefail
IMPLEMENTATION_DIR=/data/zhangzhicheng/omni-ar/case03_vcc25/runs/e2e_vcc25_20260806/implementation
nvidia-smi -L
python3 --version
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
bash "$IMPLEMENTATION_DIR/scripts/run_smoke.sh"
```

The controller/rjob submission must mount the GPFS tree containing the absolute
implementation path:
`gpfs://gpfs2/beam-gpfs02:/mnt/shared-storage-gpfs2/beam-gpfs02`.
