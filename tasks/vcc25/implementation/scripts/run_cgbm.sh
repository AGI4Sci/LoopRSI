#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "${IMPLEMENTATION_DIR}/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
GO_BP_ROOT="${GO_BP_ROOT:-${REPOSITORY_ROOT}/datasets/vcc25/annotations/official-h1-go-bp-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${IMPLEMENTATION_DIR}/artifacts/cgbm}"
DATASET="${DATASET:-vcc25_full_512g.npz}"
mkdir -p "${OUTPUT_ROOT}"
cd "${IMPLEMENTATION_DIR}"

python3 -m crpm.cgbm \
  --data-npz "${DATA_ROOT}/${DATASET}" \
  --output "${OUTPUT_ROOT}/candidate_cgbm.json" \
  --go-node-table "${GO_BP_ROOT}/node_table.csv" \
  --go-edge-table "${GO_BP_ROOT}/edge_table.csv" \
  --pretext-weight "${PRETEXT_WEIGHT:-0.1}" \
  --gat-heads "${GAT_HEADS:-4}" \
  --gat-layers "${GAT_LAYERS:-2}" \
  --neighbor-k "${NEIGHBOR_K:-16}" \
  --rank "${RANK:-24}" \
  --feature-dim "${FEATURE_DIM:-128}" \
  --calibration-folds "${CALIBRATION_FOLDS:-3}" \
  --seed "${SEED:-20260913}"
