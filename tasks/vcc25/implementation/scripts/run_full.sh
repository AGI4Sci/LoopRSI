#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "${IMPLEMENTATION_DIR}/artifacts/full"
cd "${IMPLEMENTATION_DIR}"

# The compact artifact remains a prototype. This split holds whole targets out,
# but is not an official competition evaluation.
python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_smoke.npz \
  --method crpm --split-strategy heldout_target --heldout-target-fraction 0.2 \
  --device "${DEVICE}" --seed 20250805 --epochs 20 --rank 32 \
  --shrinkage 0.001 --batch-size 128 \
  --metrics-out artifacts/full/crpm_heldout_target.json \
  --save-checkpoint artifacts/full/crpm.pt
