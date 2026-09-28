#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: $0 SEED OUTPUT_ROOT [ASSET_ROOT] [SKIP_EVAL]" >&2
  exit 2
fi

SEED=$1
OUT=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${3:-${VCC_H1_ROOT:-/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1}}
SKIP_EVAL=${4:-${VCC_SKIP_EVAL:-0}}
ASSETS=$ROOT/assets/official_2025
CELL_EVAL_ENV=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval-env
CELL_EVAL_SRC=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval
REAL_DE=$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/cell_eval_raw_vcc_test_full/real_de.csv
PRED=$OUT/official_h1_prediction.h5ad
EXPORT_MANIFEST=$OUT/official_h1_export_manifest.json
CONTRACT=$OUT/official_h1_contract.json
EVAL_OUT=$OUT/cell_eval

mkdir -p "$OUT"
test ! -e "$PRED"
test ! -e "$EVAL_OUT"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/runtime_py312:${PYTHONPATH:-}"

echo "[official-h1] root=$ROOT out=$OUT skip_eval=$SKIP_EVAL"
echo "[official-h1] seed=$SEED start=$(date -Is)"

python candidate_f_neighborhood_direction_prior.py \
  --train-h5ad "$ASSETS/train/adata_Training.h5ad" \
  --train-targets "$ASSETS/train/pert_counts_Training.csv" \
  --validation-targets "$ASSETS/validation/pert_counts_Validation.csv" \
  --test-targets "$ASSETS/test/pert_counts_Test.csv" \
  --gene-names "$ASSETS/gene_names.csv" \
  --gene-embedding-npz "$ROOT/assets/lingshu_hf_b77f980/gene_embeddings.npz" \
  --reference-h5ad "$ASSETS/test/adata_Test.h5ad" \
  --condition-csv "$ASSETS/test/pert_counts_Test.csv" \
  --output-h5ad "$PRED" \
  --manifest "$EXPORT_MANIFEST" \
  --expected-targets 100 \
  --seed "$SEED"

python official_h1_contract.py \
  --prediction-h5ad "$PRED" \
  --reference-test-h5ad "$ASSETS/test/adata_Test.h5ad" \
  --condition-csv "$ASSETS/test/pert_counts_Test.csv" \
  --gene-names "$ASSETS/gene_names.csv" \
  --output "$CONTRACT"

if [ "$SKIP_EVAL" = "1" ]; then
  echo "[official-h1] seed=$SEED prediction-ready=$(date -Is)"
  exit 0
fi

"$CELL_EVAL_ENV/bin/python3.11" -m cell_eval run \
  -ap "$PRED" \
  -ar "$ASSETS/test/adata_Test.h5ad" \
  -dr "$REAL_DE" \
  --pert-col target_gene \
  --control-pert non-targeting \
  --profile full \
  --num-threads 16 \
  -o "$EVAL_OUT"

python - "$OUT" "$SEED" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
seed = int(sys.argv[2])
rows = list(csv.DictReader((out / "cell_eval/agg_results.csv").open()))
means = next(row for row in rows if row["statistic"] == "mean")
counts = next(row for row in rows if row["statistic"] == "count")
nulls = next(row for row in rows if row["statistic"] == "null_count")
metrics = {key: float(value) for key, value in means.items() if key != "statistic"}
required = ["overlap_at_N", "de_spearman_sig", "de_spearman_lfc_sig", "pr_auc", "pearson_delta", "mae", "discrimination_score_l1"]
assert all(float(counts[key]) == 100 for key in required)
assert all(float(nulls[key]) == 0 for key in required)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

summary = {
    "protocol_id": "vcc-h1-cell-eval-v1",
    "candidate_id": "candidate_f_neighborhood_direction_prior",
    "seed": seed,
    "status": "pass",
    "targets": 100,
    "genes": 18080,
    "metrics": metrics,
    "references": {
        "lingshu_local_pearson_delta": 0.24808131580837134,
        "lingshu_paper_pearson_delta": 0.306,
    },
    "artifacts": {
        "prediction_h5ad": str((out / "official_h1_prediction.h5ad").resolve()),
        "prediction_sha256": digest(out / "official_h1_prediction.h5ad"),
        "contract_sha256": digest(out / "official_h1_contract.json"),
        "aggregate_sha256": digest(out / "cell_eval/agg_results.csv"),
    },
}
(out / "official_h1_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[official-h1] seed=$SEED done=$(date -Is)"
