# Round 0 CC-GAT suggestion report

## Changes

- Added `crpm/cc_gat.py`: sparse context-conditioned GRN attention, a train-only
  ridge decoder, and pre-decoder signal ranking for unseen perturbation targets.
- Context combines target identity, upstream-neighbor pooling, and a deterministic
  pathway-like symbol bucket fallback. The fallback is explicitly not curated GO
  evidence.
- Added an automatically evaluated shuffled-context negative control and all four
  requested activation diagnostics.
- Added `scripts/run_cc_gat.sh` as a direct worker payload and text-only contract
  coverage. The established task entrypoints and metric source paths are unchanged.

## Scope and limitations

The configured official adapter is outside the allowed edit directory, so CC-GAT
is exposed as a direct implementation payload. The existing prepared NPZ metric
is `mean_delta_pearson`; canonical official `metrics.pearson_delta` evaluation
still requires the unchanged adapter/Cell-Eval workflow. The pathway component is
a deterministic no-label fallback because no curated GO-BP asset is configured.

## Static validation

Python compilation, shell syntax, text-only contract tests, and path inspection
were run on the control node. No Torch import, model execution, smoke test,
training, or GPU job was run.

## Next worker payload

```bash
DATA_ROOT=/path/to/prepared/vcc25 OUTPUT_ROOT=/path/to/new/output \
  bash scripts/run_cc_gat.sh
```

The output is `candidate_cc_gat.json`, including ordinary and shuffled-context
metrics plus pre- and post-decoder ranking diagnostics.
