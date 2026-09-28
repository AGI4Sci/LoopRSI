#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "${IMPLEMENTATION_DIR}/artifacts/smoke"
cd "${IMPLEMENTATION_DIR}"

python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_smoke.npz \
  --method global_mean --device "${DEVICE}" --seed 20250805 \
  --train-limit 1024 --eval-limit 256 \
  --metrics-out artifacts/smoke/global_mean.json

python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_smoke.npz \
  --method crpm --device "${DEVICE}" --seed 20250805 \
  --train-limit 1024 --eval-limit 256 --epochs 10 --max-steps 80 \
  --rank 16 --shrinkage 0.001 --batch-size 128 \
  --metrics-out artifacts/smoke/crpm.json
