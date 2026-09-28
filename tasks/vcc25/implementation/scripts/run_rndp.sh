#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${IMPLEMENTATION_DIR}/artifacts/rndp}"
DATASET="${DATASET:-vcc25_full_512g.npz}"
mkdir -p "${OUTPUT_ROOT}"
cd "${IMPLEMENTATION_DIR}"

python3 -m crpm.rndp \
  --data-npz "${DATA_ROOT}/${DATASET}" \
  --output "${OUTPUT_ROOT}/candidate_rndp.json" \
  --pagerank-alpha "${PAGERANK_ALPHA:-0.15}" \
  --calibration-rank "${CALIBRATION_RANK:-64}" \
  --lasso-alpha "${LASSO_ALPHA:-0.001}" \
  --coexpression-threshold "${COEXPRESSION_THRESHOLD:-0.3}" \
  --neighbors "${GRAPH_NEIGHBORS:-32}" \
  --seed "${SEED:-20260912}"
