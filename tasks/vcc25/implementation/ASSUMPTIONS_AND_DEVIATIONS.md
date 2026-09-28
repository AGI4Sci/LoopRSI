# Assumptions and deviations

- The compact NPZ is the authoritative prototype input. It contains 1,024
  training rows, 256 validation rows, 512 genes, target codes, and batch codes.
- The artifact has no guide codes. Guide-level hierarchy and held-out-guide
  evaluation are therefore not implemented or claimed. Target-level weights
  are the only perturbation hierarchy available.
- No paired pre-perturbation cells are assumed. Baselines use aggregate
  non-targeting controls from the training partition only.
- A batch with fewer than `--min-controls-per-batch` training controls falls
  back to the global training-control mean. Counts are emitted in metrics.
- The idea card specifies half-Cauchy priors and variational inference. For the
  smallest dependency-light prototype, these are approximated by deterministic
  low-rank factors and smooth group shrinkage. No posterior uncertainty claim
  is made.
- Learned batch program weights are a lightweight calibration correction on top
  of the fixed control mean; they can be disabled independently.
- The supplied row split is engineering-only. The deterministic held-out-target
  option pools the compact train/validation rows and repartitions whole targets,
  but remains a small candidate evaluation rather than competition evidence.
- Unseen held-out targets retain the zero-initialized residual embedding because
  there are no target features from which to infer an effect. This deliberately
  exposes the method's cold-start limitation rather than leaking held-out rows.
- The full 221,273-by-18,080 H5 loader is deferred. The human requirement asks
  for the smallest runnable prototype and specifically prioritizes the compact
  NPZ for the first H200 trial.
