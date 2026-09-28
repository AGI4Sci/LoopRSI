#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/gentle_target_sampling_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

run_one() {
  local tag=$1
  shift
  python3 train.py \
    --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method guide_crpm --split-strategy heldout_guide \
    --device cuda --seed 20250805 --epochs 15 --batch-size 1024 \
    --rank 16 --learning-rate 0.01 --shrinkage 0.001 \
    --guide-shrinkage 0.0005 --delta-loss-weight 0.3 --loss-type mse \
    "$@" --metrics-out "${OUT_DIR}/${tag}.json"
}

run_one control --target-sampling-power 0.0
run_one sqrt_uncapped --target-sampling-power 0.5
run_one sqrt_cap3 --target-sampling-power 0.5 --target-sampling-max-weight 3.0
run_one inverse_cap3 --target-sampling-power 1.0 --target-sampling-max-weight 3.0

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.json")):
    if path.name == "summary.json":
        continue
    result = json.loads(path.read_text()); metrics = result["metrics"]
    row = {"name": path.stem, "power": result["target_sampling_power"],
           "max_weight": result["target_sampling_max_weight"],
           "observed_weight_min": result["target_sample_weight_min"],
           "observed_weight_max": result["target_sample_weight_max"],
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

echo VCC25_GENTLE_TARGET_SAMPLING_GRID_DONE
