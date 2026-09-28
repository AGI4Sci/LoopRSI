#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/delta_balance_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for delta_weight in 0.1 0.3 1.0; do
  for balanced in no yes; do
    extra=()
    if [[ "${balanced}" == yes ]]; then
      extra+=(--target-balanced-sampling)
    fi
    python3 train.py \
      --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
      --method guide_crpm --split-strategy heldout_guide \
      --device cuda --seed 20250805 --epochs 15 --batch-size 1024 \
      --rank 16 --learning-rate 0.01 --shrinkage 0.001 \
      --guide-shrinkage 0.0005 --delta-loss-weight "${delta_weight}" \
      "${extra[@]}" \
      --metrics-out "${OUT_DIR}/dw${delta_weight}_balanced${balanced}.json"
  done
done

python3 - "${OUT_DIR}" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("dw*_balanced*.json")):
    result = json.loads(path.read_text())
    metrics = result["metrics"]
    row = {"file": path.name, "delta_loss_weight": result["delta_loss_weight"],
           "target_balanced_sampling": result["target_balanced_sampling"],
           "mse": metrics["mse"], "mean_delta_pearson": metrics["mean_delta_pearson"],
           "top_k_delta_overlap": metrics["top_k_delta_overlap"],
           "delta_energy_ratio": metrics["delta_energy_ratio"]}
    row["selection_score"] = (row["mean_delta_pearson"]
                              + 0.25 * row["top_k_delta_overlap"]
                              - 10.0 * max(0.0, row["mse"] - 0.46536242961883545))
    rows.append(row)
rows.sort(key=lambda row: row["selection_score"], reverse=True)
summary = {"protocol": "vcc25_full_paired_heldout_guide", "rows": rows, "best": rows[0]}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo VCC25_DELTA_BALANCE_GRID_DONE
