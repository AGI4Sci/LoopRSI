#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "usage: $0 SEED OUTPUT_ROOT EXTERNAL_ARTIFACT" >&2; exit 2; fi
SEED=$1; OUT=$2; EXTERNAL_ARTIFACT=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=${VCC_H1_ROOT:-/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1}
ASSETS=$ROOT/assets/official_2025
CELL_EVAL_ENV=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval-env
CELL_EVAL_SRC=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval/src
REAL_DE=$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/cell_eval_raw_vcc_test_full/real_de.csv
PRED=$OUT/official_h1_prediction.h5ad
SHA=d3202c7f7a35c701cb5c131be6afaa1c7cebe5b8cc29df163ef8b4b7b7d48812
mkdir -p "$OUT"; cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/runtime_py312:${PYTHONPATH:-}"
PY312_SITE=$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/py312_infer_env/lib/python3.12/site-packages
export PYTHONPATH="$PY312_SITE:$PYTHONPATH"
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/vcc-numba-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/vcc-matplotlib-cache}
if [ ! -e "$PRED" ]; then
python candidate_r_promoted_transport_model.py --external-artifact "$EXTERNAL_ARTIFACT" --external-sha256 "$SHA" \
  --train-h5ad "$ASSETS/train/adata_Training.h5ad" --train-targets "$ASSETS/train/pert_counts_Training.csv" \
  --validation-targets "$ASSETS/validation/pert_counts_Validation.csv" --test-targets "$ASSETS/test/pert_counts_Test.csv" \
  --gene-names "$ASSETS/gene_names.csv" --gene-embedding-npz "$ROOT/assets/lingshu_hf_b77f980/gene_embeddings.npz" \
  --reference-h5ad "$ASSETS/test/adata_Test.h5ad" --condition-csv "$ASSETS/test/pert_counts_Test.csv" \
  --output-h5ad "$PRED" --manifest "$OUT/official_h1_export_manifest.json" --expected-targets 100 --seed "$SEED"
fi
test -f "$PRED"
python official_h1_contract.py --prediction-h5ad "$PRED" --reference-test-h5ad "$ASSETS/test/adata_Test.h5ad" \
  --condition-csv "$ASSETS/test/pert_counts_Test.csv" --gene-names "$ASSETS/gene_names.csv" --output "$OUT/official_h1_contract.json"
EVAL_OUT=$OUT/cell_eval_target_metrics
SKIP_METRICS=mse,mse_delta,mae_delta,discrimination_score_l2,discrimination_score_cosine,pearson_edistance,overlap_at_50,overlap_at_100,overlap_at_200,overlap_at_500,precision_at_N,precision_at_50,precision_at_100,precision_at_200,precision_at_500,de_direction_match,de_sig_genes_recall,de_nsig_counts,roc_auc,clustering_agreement
PRED_DE_ARGS=()
if [ -f "$OUT/cell_eval/pred_de.csv" ]; then PRED_DE_ARGS=(-dp "$OUT/cell_eval/pred_de.csv"); fi
if [ ! -e "$EVAL_OUT" ]; then
PYTHONPATH="$CELL_EVAL_SRC" "$CELL_EVAL_ENV/bin/python3.11" -m cell_eval run \
  -ap "$PRED" -ar "$ASSETS/test/adata_Test.h5ad" "${PRED_DE_ARGS[@]}" -dr "$REAL_DE" --pert-col target_gene \
  --control-pert non-targeting --profile full --skip-metrics "$SKIP_METRICS" --num-threads 16 -o "$EVAL_OUT"
fi
python - "$OUT" "$SEED" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
out,seed=Path(sys.argv[1]),int(sys.argv[2]); rows=list(csv.DictReader((out/'cell_eval_target_metrics/agg_results.csv').open()))
means=next(r for r in rows if r['statistic']=='mean'); counts=next(r for r in rows if r['statistic']=='count'); nulls=next(r for r in rows if r['statistic']=='null_count')
required=['overlap_at_N','de_spearman_sig','de_spearman_lfc_sig','pr_auc','pearson_delta','mae','discrimination_score_l1']
assert all(float(counts[k])==100 and float(nulls[k])==0 for k in required)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
result={'protocol_id':'vcc-h1-candidate-r-cell-eval-v1','candidate_id':'candidate_r_promoted_decomposed_gwps','seed':seed,'status':'pass','targets':100,'genes':18080,'metrics':{k:float(means[k]) for k in required},'artifacts':{'prediction_h5ad':str((out/'official_h1_prediction.h5ad').resolve()),'prediction_sha256':digest(out/'official_h1_prediction.h5ad'),'aggregate_sha256':digest(out/'cell_eval_target_metrics/agg_results.csv')}}
(out/'official_h1_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps(result,indent=2,sort_keys=True))
PY
