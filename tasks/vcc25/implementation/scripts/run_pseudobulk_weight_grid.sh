#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/pseudobulk_weight_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for power in 0.0 0.25 0.5 0.75 1.0; do
  python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method guide_crpm --split-strategy heldout_guide --device cuda \
    --seed 20250805 --epochs 250 --batch-size 256 --rank 16 \
    --learning-rate 0.01 --shrinkage 0.001 --guide-shrinkage 0.0005 \
    --delta-loss-weight 0.3 --loss-type mse --aggregate-training \
    --aggregate-count-power "${power}" \
    --metrics-out "${OUT_DIR}/power${power}.json"
done

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
out_dir=pathlib.Path(sys.argv[1]); rows=[]
for path in sorted(out_dir.glob("power*.json")):
    r=json.loads(path.read_text()); m=r["metrics"]
    row={"aggregate_count_power":r["aggregate_count_power"],"mse":m["mse"],
         "mean_delta_pearson":m["mean_delta_pearson"],
         "top_k_delta_overlap":m["top_k_delta_overlap"],
         "delta_energy_ratio":m["delta_energy_ratio"],"runtime_seconds":r["runtime_seconds"]}
    row["selection_score"]=(row["mean_delta_pearson"]+.25*row["top_k_delta_overlap"]
                            -10*max(0.,row["mse"]-.46536242961883545))
    rows.append(row)
rows.sort(key=lambda x:x["selection_score"],reverse=True)
summary={"protocol":"vcc25_full_paired_heldout_guide","rows":rows,"best":rows[0]}
(out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("PSEUDOBULK_WEIGHT_SUMMARY="+json.dumps(summary,sort_keys=True))
print("VCC25_PSEUDOBULK_WEIGHT_DONE")
PY
