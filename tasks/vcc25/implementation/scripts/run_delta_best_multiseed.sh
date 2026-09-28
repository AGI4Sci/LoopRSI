#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/delta_best_multiseed"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for seed in 20250805 20250806 20250807; do
  python3 train.py \
    --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method guide_crpm --split-strategy heldout_guide \
    --device cuda --seed "${seed}" --epochs 15 --batch-size 1024 \
    --rank 16 --learning-rate 0.01 --shrinkage 0.001 \
    --guide-shrinkage 0.0005 --delta-loss-weight 0.3 \
    --metrics-out "${OUT_DIR}/seed${seed}.json"
done

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, statistics, sys
out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("seed*.json")):
    result = json.loads(path.read_text()); metrics = result["metrics"]
    rows.append({"seed": result["seed"], "mse": metrics["mse"],
                 "mean_delta_pearson": metrics["mean_delta_pearson"],
                 "top_k_delta_overlap": metrics["top_k_delta_overlap"],
                 "beats_global_mean_mse": result["beats_global_mean_mse"]})
keys = ("mse", "mean_delta_pearson", "top_k_delta_overlap")
summary = {"protocol": "vcc25_full_paired_heldout_guide", "rank": 16,
           "guide_shrinkage": 0.0005, "delta_loss_weight": 0.3, "rows": rows,
           "mean": {key: statistics.mean(row[key] for row in rows) for key in keys},
           "std": {key: statistics.pstdev(row[key] for row in rows) for key in keys}}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo VCC25_DELTA_BEST_MULTISEED_DONE
