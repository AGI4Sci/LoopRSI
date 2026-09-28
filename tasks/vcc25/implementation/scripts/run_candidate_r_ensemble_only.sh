#!/usr/bin/env bash
set -euo pipefail
ROOT=${VCC_H1_ROOT:-/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1}
IMPL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=$ROOT/runs/vcc25_candidate_r_official_20260914
ASSETS=$ROOT/assets/official_2025
PY312_SITE=$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/py312_infer_env/lib/python3.12/site-packages
CELL_EVAL_ENV=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval-env
CELL_EVAL_SRC=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval/src
REAL_DE=$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/cell_eval_raw_vcc_test_full/real_de.csv
OUT=$RUN_ROOT/ensemble
PRED=$OUT/official_h1_prediction.h5ad
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/vcc-numba-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/vcc-matplotlib-cache}
if [ ! -e "$PRED" ]; then
  PYTHONPATH="$PY312_SITE" python "$IMPL/official_h1_seed_ensemble.py" \
    --inputs "$RUN_ROOT"/seed_*/official_h1_prediction.h5ad --output "$PRED"
fi
SKIP_METRICS=mse,mse_delta,mae_delta,discrimination_score_l2,discrimination_score_cosine,pearson_edistance,overlap_at_50,overlap_at_100,overlap_at_200,overlap_at_500,precision_at_N,precision_at_50,precision_at_100,precision_at_200,precision_at_500,de_direction_match,de_sig_genes_recall,de_nsig_counts,roc_auc,clustering_agreement
if [ ! -e "$OUT/cell_eval_target_metrics" ]; then
  PYTHONPATH="$CELL_EVAL_SRC" "$CELL_EVAL_ENV/bin/python3.11" -m cell_eval run \
    -ap "$PRED" -ar "$ASSETS/test/adata_Test.h5ad" -dr "$REAL_DE" \
    --pert-col target_gene --control-pert non-targeting --profile full \
    --skip-metrics "$SKIP_METRICS" --num-threads 16 -o "$OUT/cell_eval_target_metrics"
fi
