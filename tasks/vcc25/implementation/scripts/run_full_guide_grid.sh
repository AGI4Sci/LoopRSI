#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/full_guide_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

# Fixed scientific protocol: every evaluation guide belongs to a target that
# remains represented by a different guide in training. Only model capacity
# and guide-residual regularization vary in this first optimization sweep.
for rank in 8 16 32 64; do
  for guide_shrinkage in 0.0005 0.002 0.01; do
    tag="rank${rank}_gs${guide_shrinkage}"
    python3 train.py \
      --data-root "${DATA_ROOT}" \
      --dataset vcc25_full_512g.npz \
      --method guide_crpm \
      --split-strategy heldout_guide \
      --device cuda \
      --seed 20250805 \
      --epochs 15 \
      --batch-size 1024 \
      --rank "${rank}" \
      --learning-rate 0.01 \
      --shrinkage 0.001 \
      --guide-shrinkage "${guide_shrinkage}" \
      --metrics-out "${OUT_DIR}/${tag}.json"
  done
done

python3 - "${OUT_DIR}" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("rank*_gs*.json")):
    result = json.loads(path.read_text())
    metrics = result["metrics"]
    rows.append({
        "file": path.name,
        "rank": result["rank"],
        "guide_shrinkage": result["guide_shrinkage"],
        "mse": metrics["mse"],
        "mean_delta_pearson": metrics["mean_delta_pearson"],
        "top_k_delta_overlap": metrics["top_k_delta_overlap"],
        "delta_energy_ratio": metrics["delta_energy_ratio"],
        "runtime_seconds": result["runtime_seconds"],
    })

# Composite is used only for ranking candidates: prioritize perturbation
# structure, then top-k recovery, with a mild penalty for MSE above global mean.
global_mse = 0.46536242961883545
for row in rows:
    row["selection_score"] = (
        row["mean_delta_pearson"]
        + 0.25 * row["top_k_delta_overlap"]
        - 10.0 * max(0.0, row["mse"] - global_mse)
    )
rows.sort(key=lambda row: row["selection_score"], reverse=True)
summary = {"protocol": "vcc25_full_paired_heldout_guide", "rows": rows, "best": rows[0]}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo VCC25_FULL_GUIDE_GRID_DONE
