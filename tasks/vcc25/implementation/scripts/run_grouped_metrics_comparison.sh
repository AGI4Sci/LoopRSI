#!/usr/bin/env bash
set -euo pipefail

IMPLEMENTATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${IMPLEMENTATION_DIR}/data}"
OUT_DIR="${IMPLEMENTATION_DIR}/artifacts/grouped_metrics_comparison"
mkdir -p "${OUT_DIR}"
cd "${IMPLEMENTATION_DIR}"

for seed in 20250805 20250806 20250807; do
  common=(
    --data-root "${DATA_ROOT}" --dataset vcc25_full_512g.npz
    --method guide_crpm --split-strategy heldout_guide
    --device cuda --seed "${seed}" --epochs 15 --batch-size 1024
    --rank 16 --learning-rate 0.01 --shrinkage 0.001
    --guide-shrinkage 0.0005 --delta-loss-weight 0.3 --loss-type mse
    --group-metrics
  )
  python3 train.py "${common[@]}" \
    --target-sampling-power 0.0 \
    --metrics-out "${OUT_DIR}/control_seed${seed}.json"
  python3 train.py "${common[@]}" \
    --target-sampling-power 0.5 --target-sampling-max-weight 3.0 \
    --metrics-out "${OUT_DIR}/sqrt_cap3_seed${seed}.json"
done

python3 - "${OUT_DIR}" <<'PY'
import json, pathlib, statistics, sys

out_dir = pathlib.Path(sys.argv[1])
seeds = (20250805, 20250806, 20250807)
dimensions = ("target", "guide", "batch")
summary = {"protocol": "vcc25_full_paired_heldout_guide", "seeds": {}, "aggregate": {}}

for seed in seeds:
    control = json.loads((out_dir / f"control_seed{seed}.json").read_text())
    sampled = json.loads((out_dir / f"sqrt_cap3_seed{seed}.json").read_text())
    seed_result = {}
    for dimension in dimensions:
        left = {row["group_id"]: row for row in control["group_metrics"][dimension]}
        right = {row["group_id"]: row for row in sampled["group_metrics"][dimension]}
        comparisons = []
        for group_id in sorted(left.keys() & right.keys()):
            a, b = left[group_id], right[group_id]
            delta_a, delta_b = a["mean_delta_pearson"], b["mean_delta_pearson"]
            comparisons.append({
                "group_id": group_id, "group_name": a["group_name"],
                "n_cells": a["n_cells"],
                "mse_control": a["mse"], "mse_sampled": b["mse"],
                "mse_improvement": a["mse"] - b["mse"],
                "delta_pearson_control": delta_a, "delta_pearson_sampled": delta_b,
                "delta_pearson_change": None if delta_a is None or delta_b is None else delta_b - delta_a,
                "top50_control": a["top_k_delta_overlap"],
                "top50_sampled": b["top_k_delta_overlap"],
                "top50_change": b["top_k_delta_overlap"] - a["top_k_delta_overlap"],
            })
        valid_delta = [row for row in comparisons if row["delta_pearson_change"] is not None]
        seed_result[dimension] = {
            "n_groups": len(comparisons),
            "delta_pearson_improved_groups": sum(row["delta_pearson_change"] > 0 for row in valid_delta),
            "delta_pearson_degraded_groups": sum(row["delta_pearson_change"] < 0 for row in valid_delta),
            "top50_improved_groups": sum(row["top50_change"] > 0 for row in comparisons),
            "top50_degraded_groups": sum(row["top50_change"] < 0 for row in comparisons),
            "groups": comparisons,
        }
    summary["seeds"][str(seed)] = seed_result

for dimension in dimensions:
    by_name = {}
    for seed in seeds:
        for row in summary["seeds"][str(seed)][dimension]["groups"]:
            by_name.setdefault(row["group_name"], []).append(row)
    aggregate_rows = []
    for name, rows in by_name.items():
        delta_changes = [r["delta_pearson_change"] for r in rows if r["delta_pearson_change"] is not None]
        aggregate_rows.append({
            "group_name": name, "seeds_present": len(rows),
            "mean_cells": statistics.mean(r["n_cells"] for r in rows),
            "mean_mse_improvement": statistics.mean(r["mse_improvement"] for r in rows),
            "mean_delta_pearson_change": statistics.mean(delta_changes) if delta_changes else None,
            "mean_top50_change": statistics.mean(r["top50_change"] for r in rows),
        })
    delta_ranked = sorted(
        (r for r in aggregate_rows if r["mean_delta_pearson_change"] is not None),
        key=lambda r: r["mean_delta_pearson_change"], reverse=True)
    summary["aggregate"][dimension] = {
        "n_unique_groups": len(aggregate_rows),
        "top_delta_gains": delta_ranked[:10],
        "top_delta_losses": list(reversed(delta_ranked[-10:])),
        "all_groups": aggregate_rows,
    }

(out_dir / "grouped_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
compact = {
    "protocol": summary["protocol"],
    "seed_group_counts": {
        seed: {dim: {k: v for k, v in payload.items() if k != "groups"}
               for dim, payload in dims.items()}
        for seed, dims in summary["seeds"].items()
    },
    "aggregate": {
        dim: {"n_unique_groups": payload["n_unique_groups"],
              "top_delta_gains": payload["top_delta_gains"],
              "top_delta_losses": payload["top_delta_losses"]}
        for dim, payload in summary["aggregate"].items()
    },
}
print("GROUPED_COMPARISON_SUMMARY=" + json.dumps(compact, sort_keys=True))
print("VCC25_GROUPED_METRICS_COMPARISON_DONE")
PY
