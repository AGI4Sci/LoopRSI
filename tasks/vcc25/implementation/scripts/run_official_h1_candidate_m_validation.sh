#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXTERNAL_ROOT=${EXTERNAL_ROOT:-/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/datasets/replogle-2022/derived}
PY312_SITE=/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1/runs/lingshu_vcc_reproduction_b77f980_20260908/py312_infer_env/lib/python3.12/site-packages
export PYTHONPATH="$PY312_SITE${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/official_h1_candidate_m_six_direction_gate.py" \
  --external-artifact "$EXTERNAL_ROOT/h1_exclusion_safe_pseudobulk_delta.npz" \
  --external-sha256 c38e7d6e12daf414bf516df0081b7ca4f5de709e7807bdf07854911cdf3dac48 \
  --fold-salt candidate-m-six-direction-folds-20260913 \
  --output "${1:?output path required}"
