#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/pseudobulk_best_multiseed"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for seed in 20250805 20250806 20250807; do
  python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method guide_crpm --split-strategy heldout_guide --device cuda \
    --seed "${seed}" --epochs 250 --batch-size 256 --rank 16 \
    --learning-rate 0.01 --shrinkage 0.001 --guide-shrinkage 0.0005 \
    --delta-loss-weight 0.3 --loss-type mse --aggregate-training \
    --aggregate-count-power 1.0 --group-metrics \
    --metrics-out "${OUT_DIR}/seed${seed}.json"
done

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, statistics, sys
out_dir=pathlib.Path(sys.argv[1]); rows=[]
for path in sorted(out_dir.glob("seed*.json")):
    r=json.loads(path.read_text()); m=r["metrics"]
    rows.append({"seed":r["seed"],"mse":m["mse"],
                 "mean_delta_pearson":m["mean_delta_pearson"],
                 "top_k_delta_overlap":m["top_k_delta_overlap"],
                 "delta_energy_ratio":m["delta_energy_ratio"]})
keys=("mse","mean_delta_pearson","top_k_delta_overlap","delta_energy_ratio")
aggregate={k:{"mean":statistics.mean(row[k] for row in rows),
              "stdev":statistics.stdev(row[k] for row in rows)} for k in keys}
summary={"protocol":"vcc25_full_paired_heldout_guide","configuration":{
    "rank":16,"epochs":250,"aggregate_training":True,"aggregate_count_power":1.0},
    "rows":rows,"aggregate":aggregate}
(out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("PSEUDOBULK_BEST_MULTISEED_SUMMARY="+json.dumps(summary,sort_keys=True))
print("VCC25_PSEUDOBULK_BEST_MULTISEED_DONE")
PY
