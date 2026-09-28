#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/target_batch_interaction_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

# The disabled row is the exact current-best control.
python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
  --method guide_crpm --split-strategy heldout_guide --device cuda \
  --seed 20250805 --epochs 15 --batch-size 1024 --rank 16 \
  --learning-rate 0.01 --shrinkage 0.001 --guide-shrinkage 0.0005 \
  --delta-loss-weight 0.3 --loss-type mse \
  --metrics-out "${OUT_DIR}/disabled.json"

for interaction_shrinkage in 0.0 0.0001 0.001 0.01 0.1; do
  python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method guide_crpm --split-strategy heldout_guide --device cuda \
    --seed 20250805 --epochs 15 --batch-size 1024 --rank 16 \
    --learning-rate 0.01 --shrinkage 0.001 --guide-shrinkage 0.0005 \
    --delta-loss-weight 0.3 --loss-type mse --target-batch-interaction \
    --interaction-shrinkage "${interaction_shrinkage}" \
    --metrics-out "${OUT_DIR}/is${interaction_shrinkage}.json"
done

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1]); rows = []
for path in sorted(out_dir.glob("*.json")):
    if path.name == "summary.json": continue
    result = json.loads(path.read_text()); metrics = result["metrics"]
    row = {"target_batch_interaction": result["target_batch_interaction"],
           "interaction_shrinkage": result["interaction_shrinkage"],
           "mse": metrics["mse"], "mean_delta_pearson": metrics["mean_delta_pearson"],
           "top_k_delta_overlap": metrics["top_k_delta_overlap"],
           "delta_energy_ratio": metrics["delta_energy_ratio"],
           "runtime_seconds": result["runtime_seconds"]}
    row["selection_score"] = (row["mean_delta_pearson"]
                              + .25 * row["top_k_delta_overlap"]
                              - 10 * max(0., row["mse"] - .46536242961883545))
    rows.append(row)
rows.sort(key=lambda x: x["selection_score"], reverse=True)
summary = {"protocol": "vcc25_full_paired_heldout_guide", "rows": rows, "best": rows[0]}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("TARGET_BATCH_INTERACTION_SUMMARY=" + json.dumps(summary, sort_keys=True))
print("VCC25_TARGET_BATCH_INTERACTION_DONE")
PY
