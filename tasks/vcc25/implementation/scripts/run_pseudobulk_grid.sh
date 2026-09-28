#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/pseudobulk_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for rank in 8 16 32; do
  for epochs in 100 250; do
    python3 train.py --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
      --method guide_crpm --split-strategy heldout_guide --device cuda \
      --seed 20250805 --epochs "${epochs}" --batch-size 256 --rank "${rank}" \
      --learning-rate 0.01 --shrinkage 0.001 --guide-shrinkage 0.0005 \
      --delta-loss-weight 0.3 --loss-type mse --aggregate-training \
      --metrics-out "${OUT_DIR}/rank${rank}_e${epochs}.json"
  done
done

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1]); rows=[]
for path in sorted(out_dir.glob("rank*.json")):
    result=json.loads(path.read_text()); m=result["metrics"]
    row={"rank":result["rank"], "epochs":result["epochs_requested"],
         "n_fit_samples":result["n_fit_samples"], "mse":m["mse"],
         "mean_delta_pearson":m["mean_delta_pearson"],
         "top_k_delta_overlap":m["top_k_delta_overlap"],
         "delta_energy_ratio":m["delta_energy_ratio"],
         "runtime_seconds":result["runtime_seconds"]}
    row["selection_score"]=(row["mean_delta_pearson"]+.25*row["top_k_delta_overlap"]
                            -10*max(0.,row["mse"]-.46536242961883545))
    rows.append(row)
rows.sort(key=lambda x:x["selection_score"],reverse=True)
summary={"protocol":"vcc25_full_paired_heldout_guide","rows":rows,"best":rows[0]}
(out_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("PSEUDOBULK_GRID_SUMMARY="+json.dumps(summary,sort_keys=True))
print("VCC25_PSEUDOBULK_GRID_DONE")
PY
