#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR=${1:?run directory required}
CONFIG=${2:-$ROOT/configs/replogle_gwps_quantitative_target_complete_20260914.json}
PY312_SITE=/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1/runs/lingshu_vcc_reproduction_b77f980_20260908/py312_infer_env/lib/python3.12/site-packages
export PYTHONPATH="$PY312_SITE${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_DIR"
ARTIFACT="$RUN_DIR/h1_official_replogle_gwps_quantitative_delta.npz"
MANIFEST="$RUN_DIR/h1_official_replogle_gwps_quantitative_manifest.json"
RESULT="$RUN_DIR/result.json"

python "$ROOT/prepare_multicontext_corpus.py" \
  --sources "$CONFIG" \
  --information-policy vcc_official \
  --chunk-rows 3886 \
  --output "$ARTIFACT" \
  --manifest "$MANIFEST"

EXTERNAL_ARTIFACT="$ARTIFACT" \
EXTERNAL_SHA256=$(sha256sum "$ARTIFACT" | awk '{print $1}') \
  bash "$ROOT/scripts/run_official_h1_candidate_q_validation.sh" "$RESULT"

cat "$RESULT"
