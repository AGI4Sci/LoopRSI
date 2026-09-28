#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/guide_dropout_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for dropout in 0.0 0.1 0.25 0.5 0.75 1.0; do
  python3 train.py \
    --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method guide_crpm --split-strategy heldout_guide \
    --device cuda --seed 20250805 --epochs 15 --batch-size 1024 \
    --rank 16 --learning-rate 0.01 --shrinkage 0.001 \
    --guide-shrinkage 0.0005 --guide-dropout "${dropout}" \
    --delta-loss-weight 0.3 --loss-type mse \
    --metrics-out "${OUT_DIR}/dropout${dropout}.json"
done

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1]); rows = []
for path in sorted(out_dir.glob("dropout*.json")):
    result = json.loads(path.read_text()); metrics = result["metrics"]
    row = {"guide_dropout": result["guide_dropout"], "mse": metrics["mse"],
           "mean_delta_pearson": metrics["mean_delta_pearson"],
           "top_k_delta_overlap": metrics["top_k_delta_overlap"],
           "delta_energy_ratio": metrics["delta_energy_ratio"],
           "runtime_seconds": result["runtime_seconds"]}
    row["selection_score"] = (row["mean_delta_pearson"]
                              + 0.25 * row["top_k_delta_overlap"]
                              - 10.0 * max(0.0, row["mse"] - 0.46536242961883545))
    rows.append(row)
rows.sort(key=lambda row: row["selection_score"], reverse=True)
summary = {"protocol": "vcc25_full_paired_heldout_guide", "rows": rows, "best": rows[0]}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("GUIDE_DROPOUT_GRID_SUMMARY=" + json.dumps(summary, sort_keys=True))
print("VCC25_GUIDE_DROPOUT_GRID_DONE")
PY
