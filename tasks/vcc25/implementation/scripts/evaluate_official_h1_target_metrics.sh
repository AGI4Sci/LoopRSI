#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 SEED ROOT PRED_DE" >&2
  exit 2
fi

SEED=$1
ROOT=$2
PRED_DE=$3
OUT=$ROOT/runs/vcc_skill_candidate_f_3seeds_20260912/seed_$SEED
PRED=$OUT/official_h1_prediction.h5ad
EVAL_OUT=$OUT/cell_eval_target_metrics
SKIP_METRICS=mse,mse_delta,mae_delta,discrimination_score_l2,discrimination_score_cosine,pearson_edistance,overlap_at_50,overlap_at_100,overlap_at_200,overlap_at_500,precision_at_N,precision_at_50,precision_at_100,precision_at_200,precision_at_500,de_direction_match,de_sig_genes_recall,de_nsig_counts,roc_auc,clustering_agreement

test -f "$PRED"
test -f "$PRED_DE"
test ! -e "$EVAL_OUT"

export PYTHONPATH="$ROOT/protocols/lingshu_cell_h1/tools/cell-eval/src:${PYTHONPATH:-}"
"$ROOT/protocols/lingshu_cell_h1/tools/cell-eval-env/bin/python3.11" -m cell_eval run \
  -ap "$PRED" \
  -ar "$ROOT/assets/official_2025/test/adata_Test.h5ad" \
  -dp "$PRED_DE" \
  -dr "$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/cell_eval_raw_vcc_test_full/real_de.csv" \
  --pert-col target_gene \
  --control-pert non-targeting \
  --profile full \
  --skip-metrics "$SKIP_METRICS" \
  --num-threads 16 \
  -o "$EVAL_OUT"
