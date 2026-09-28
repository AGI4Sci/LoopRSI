#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/gentle_target_sampling_multiseed"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for seed in 20250805 20250806 20250807; do
  python3 train.py \
    --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method guide_crpm --split-strategy heldout_guide \
    --device cuda --seed "${seed}" --epochs 15 --batch-size 1024 \
    --rank 16 --learning-rate 0.01 --shrinkage 0.001 \
    --guide-shrinkage 0.0005 --delta-loss-weight 0.3 --loss-type mse \
    --target-sampling-power 0.5 --target-sampling-max-weight 3.0 \
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
                 "beats_global_mean_mse": result["beats_global_mean_mse"],
                 "observed_weight_max": result["target_sample_weight_max"]})
keys = ("mse", "mean_delta_pearson", "top_k_delta_overlap")
summary = {"protocol": "vcc25_full_paired_heldout_guide",
           "target_sampling_power": 0.5, "target_sampling_max_weight": 3.0,
           "rows": rows,
           "mean": {key: statistics.mean(row[key] for row in rows) for key in keys},
           "std": {key: statistics.pstdev(row[key] for row in rows) for key in keys}}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo VCC25_GENTLE_TARGET_SAMPLING_MULTISEED_DONE
