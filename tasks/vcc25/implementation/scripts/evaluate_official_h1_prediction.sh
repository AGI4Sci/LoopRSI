#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "usage: $0 SEED ROOT" >&2; exit 2; fi
SEED=$1
ROOT=$2
OUT=$ROOT/runs/vcc_skill_candidate_f_3seeds_20260912/seed_$SEED
PRED=$OUT/official_h1_prediction.h5ad
EVAL_OUT=$OUT/cell_eval
test -f "$PRED"
test ! -e "$EVAL_OUT"
export PYTHONPATH="$ROOT/protocols/lingshu_cell_h1/tools/cell-eval/src:${PYTHONPATH:-}"
"$ROOT/protocols/lingshu_cell_h1/tools/cell-eval-env/bin/python3.11" -m cell_eval run -ap "$PRED" -ar "$ROOT/assets/official_2025/test/adata_Test.h5ad" -dr "$ROOT/runs/lingshu_vcc_reproduction_b77f980_20260908/cell_eval_raw_vcc_test_full/real_de.csv" --pert-col target_gene --control-pert non-targeting --profile full --num-threads 16 -o "$EVAL_OUT"
