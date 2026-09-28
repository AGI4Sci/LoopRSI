#!/usr/bin/env bash
set -euo pipefail
ROOT=${VCC_H1_ROOT:-/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1}
IMPL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=$ROOT/runs/vcc25_candidate_r_official_20260914
EXTERNAL=$ROOT/runs/vcc25_candidate_q_quantitative_20260914/h1_official_replogle_gwps_quantitative_delta.npz
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/vcc-numba-cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/vcc-matplotlib-cache}
for seed in 20260907 20260908 20260909; do
  bash "$IMPL/scripts/run_official_h1_candidate_r.sh" "$seed" "$RUN_ROOT/seed_$seed" "$EXTERNAL"
done
PY312_SITE=$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/py312_infer_env/lib/python3.12/site-packages
PYTHONPATH="$PY312_SITE" python "$IMPL/official_h1_seed_ensemble.py" \
  --inputs "$RUN_ROOT"/seed_*/official_h1_prediction.h5ad \
  --output "$RUN_ROOT/ensemble/official_h1_prediction.h5ad"
ASSETS=$ROOT/assets/official_2025
CELL_EVAL_ENV=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval-env
CELL_EVAL_SRC=$ROOT/protocols/lingshu_cell_h1/tools/cell-eval/src
REAL_DE=$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/cell_eval_raw_vcc_test_full/real_de.csv
SKIP_METRICS=mse,mse_delta,mae_delta,discrimination_score_l2,discrimination_score_cosine,pearson_edistance,overlap_at_50,overlap_at_100,overlap_at_200,overlap_at_500,precision_at_N,precision_at_50,precision_at_100,precision_at_200,precision_at_500,de_direction_match,de_sig_genes_recall,de_nsig_counts,roc_auc,clustering_agreement
PYTHONPATH="$CELL_EVAL_SRC" "$CELL_EVAL_ENV/bin/python3.11" -m cell_eval run \
  -ap "$RUN_ROOT/ensemble/official_h1_prediction.h5ad" -ar "$ASSETS/test/adata_Test.h5ad" -dr "$REAL_DE" \
  --pert-col target_gene --control-pert non-targeting --profile full --skip-metrics "$SKIP_METRICS" --num-threads 16 \
  -o "$RUN_ROOT/ensemble/cell_eval_target_metrics"
