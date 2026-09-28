#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXTERNAL_ARTIFACT=${EXTERNAL_ARTIFACT:?quantitative GWPS artifact is required}
EXTERNAL_SHA256=${EXTERNAL_SHA256:?quantitative GWPS artifact SHA256 is required}
PY312_SITE=/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1/runs/lingshu_vcc_reproduction_b77f980_20260908/py312_infer_env/lib/python3.12/site-packages
export PYTHONPATH="$PY312_SITE${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/official_h1_candidate_r_decomposed_gwps.py" \
  --external-artifact "$EXTERNAL_ARTIFACT" \
  --external-sha256 "$EXTERNAL_SHA256" \
  --fold-salt candidate-r-decomposed-gwps-folds-20260914 \
  --output "${1:?output path required}"
