# Round 1 suggestion report

## What changed

- Added `--loss-type {mse,huber}` to `train.py`, defaulting to `mse` for backward compatibility.
- Added `--huber-delta`, defaulting to `1.0` and validated as positive.
- Applied the existing optional per-gene delta weights to either elementwise loss before reduction.
- Recorded the selected loss configuration in metrics JSON.

## Failure targeted

Huber loss limits the influence of large reconstruction errors, addressing the suggestion's highest-priority evidence gap around outlier sensitivity and unstable perturbation-recovery metrics. Evaluation MSE remains computed independently by `regression_metrics`, so changing the optimization loss does not redefine the reported MSE.

## Static validation

- Parsed and byte-compiled the changed Python source without importing or executing `torch`.
- Checked shell syntax for the implementation launch scripts.
- Confirmed by source inspection that the default MSE expression remains `(prediction - y) ** 2` and that loss weighting still occurs before the mean reduction.

No runtime, unit, smoke, training, or GPU jobs were run on the control node, per the loop-engineering guardrail.

## Next controller-managed rjob payload

From the implementation directory, compare the new robust-loss variant with the current MSE configuration using the existing mounted dataset path:

```bash
python3 train.py --data-root data --dataset vcc25_full_512g.npz --method guide_crpm --split-strategy heldout_guide --seed 20250805 --epochs 15 --rank 16 --shrinkage 0.001 --guide-shrinkage 0.0005 --delta-loss-weight 0.3 --metrics-out artifacts/round_1_robust_loss.json --loss-type huber --huber-delta 1.0
```

Run this payload directly inside the controller-managed rjob; do not invoke `scripts/submit_rjob.sh` from an rjob worker.
