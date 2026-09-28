#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/prototype_architecture_grid"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

run_one() {
  local tag=$1 rank=$2 blend=$3 target_pc=$4 batch_pc=$5
  python3 train.py \
    --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz \
    --method target_prototype --split-strategy heldout_guide \
    --device cuda --seed 20250805 --batch-size 1024 \
    --prototype-rank "${rank}" --prototype-blend "${blend}" \
    --prototype-iterations 8 --prototype-target-pseudocount "${target_pc}" \
    --prototype-batch-pseudocount "${batch_pc}" \
    --metrics-out "${OUT_DIR}/${tag}.json"
}

# Direct two-way prototypes and low-rank denoising strengths.
run_one direct_pc0 0 0.0 0 100
run_one direct_pc100 0 0.0 100 100
run_one direct_pc500 0 0.0 500 100
for rank in 4 8 16 32 64; do
  run_one "rank${rank}_full" "${rank}" 1.0 100 100
done
for blend in 0.25 0.5 0.75; do
  run_one "rank16_blend${blend}" 16 "${blend}" 100 100
done
run_one rank16_nobatchshrink 16 0.5 100 0
run_one rank16_batchpc500 16 0.5 100 500

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, sys
out_dir = pathlib.Path(sys.argv[1]); rows = []
for path in sorted(out_dir.glob("*.json")):
    if path.name == "summary.json": continue
    result = json.loads(path.read_text()); metrics = result["metrics"]
    row = {"name": path.stem, "rank": result["prototype_rank_requested"],
           "blend": result["prototype_blend"],
           "target_pseudocount": result["prototype_target_pseudocount"],
           "batch_pseudocount": result["prototype_batch_pseudocount"],
           "mse": metrics["mse"], "mean_delta_pearson": metrics["mean_delta_pearson"],
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
print("PROTOTYPE_GRID_SUMMARY=" + json.dumps(summary, sort_keys=True))
print("VCC25_PROTOTYPE_ARCHITECTURE_GRID_DONE")
PY
