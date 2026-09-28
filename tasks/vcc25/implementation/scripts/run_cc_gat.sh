#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${IMPLEMENTATION_DIR}/artifacts/cc_gat}"
DATASET="${DATASET:-vcc25_full_512g.npz}"
mkdir -p "${OUTPUT_ROOT}"
cd "${IMPLEMENTATION_DIR}"

python3 -m crpm.cc_gat \
  --data-npz "${DATA_ROOT}/${DATASET}" \
  --output "${OUTPUT_ROOT}/candidate_cc_gat.json" \
  --context-dim "${CONTEXT_DIM:-128}" \
  --gat-heads "${GAT_HEADS:-4}" \
  --gat-layers "${GAT_LAYERS:-2}" \
  --grn-neighbors "${GRN_NEIGHBORS:-32}" \
  --grn-threshold "${GRN_THRESHOLD:-0.3}" \
  --ridge "${RIDGE:-0.001}" \
  --seed "${SEED:-20260913}"
