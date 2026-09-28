# Implementation plan

1. Load and validate the compact NPZ without assuming access to the source H5.
2. Estimate each batch baseline only from training non-targeting controls, with
   a global-control fallback for batches with too few controls.
3. Fit target and batch coefficients over shared low-rank gene programs to the
   control-referenced residual, with a group shrinkage penalty.
4. Compare every run to global-mean and batch-control baselines using absolute
   and perturbation-sensitive metrics written as JSON.
5. Falsify components with zero-shrinkage and no-batch-calibration ablations;
   use held-out targets before making scientific claims.

## Claim-to-code traceability

| Idea-card claim | Implementation | Switch / diagnostic |
|---|---|---|
| SC1: batch control residualization | `crpm/data.py:estimate_control_baselines` | `batch_control_mean`; fallback counts |
| SC2: shared low-rank programs | `crpm/model.py:CRPM` | `--rank` |
| SC3: hierarchical shrinkage | `CRPM.shrinkage_penalty` | `--shrinkage`; set 0 for ablation |
| SC4: beat simple baselines on held-out targets | `train.py`, `crpm/metrics.py` | `--split-strategy heldout_target`; `beats_global_mean_mse` |

The next scale-up should add a sparse H5 loader, guide labels, and the official
evaluation protocol only after this compact vertical slice runs successfully.
