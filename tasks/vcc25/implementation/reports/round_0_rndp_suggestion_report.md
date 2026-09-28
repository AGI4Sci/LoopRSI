# Round 0 RNDP suggestion report

## Changes

- Added `crpm/rndp.py`, a training-only co-expression graph with sparse
  personalized PageRank, rank-64 diffusion calibration, soft-thresholded sparse
  residual correction, and a degree-matched randomized-topology control.
- Added all four requested activation diagnostics to the result JSON.
- Added `scripts/run_rndp.sh` as the direct experiment payload and static
  contract coverage in `tests/test_contract.py`.
- Exported the module from `crpm` without changing the established training or
  evaluation entrypoints.

## Scope and limitation

The configured task adapter is outside the allowed edit directory and currently
accepts only `candidate_g_promoted_delta_model`. Consequently, RNDP is exposed
as a direct implementation payload rather than silently changing official H1
adapter semantics. Its NPZ validation result uses the implementation's existing
`mean_delta_pearson` proxy; canonical official `pearson_delta` still requires
Cell-Eval integration in an authorized adapter change.

## Validation

Only static checks were run on the control node: Python AST/bytecode syntax,
shell syntax, path/interface inspection, and the text-only contract tests. No
Torch import, model execution, training, smoke test, or GPU job was run.

## Next worker payload

```bash
DATA_ROOT=/path/to/prepared/vcc25 OUTPUT_ROOT=/path/to/new/output \
  bash scripts/run_rndp.sh
```

The output is `candidate_rndp.json`, including the structured-topology and
random-graph metrics. Run it on a controller-managed worker with a fresh output
directory; do not invoke `scripts/submit_rjob.sh` from inside the worker.
