#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 /abs/path/to/pipeline.env" >&2
  exit 2
fi

CONFIG=$1
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -a
source "$CONFIG"
if [ -n "${HEURESIS_ENV_FILE:-}" ] && [ -f "$HEURESIS_ENV_FILE" ]; then
  source "$HEURESIS_ENV_FILE"
fi
set +a

: "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
: "${TASK_SPEC:?TASK_SPEC must point to a validated task_spec.yaml}"
: "${RESEARCHSTUDIO_DIR:?RESEARCHSTUDIO_DIR must be set}"
: "${RESEARCHSTUDIO_CONFIG:?RESEARCHSTUDIO_CONFIG must be set}"
: "${IDEA_CARD_DIR:?IDEA_CARD_DIR must be set}"
: "${IMPLEMENTATION_DIR:?IMPLEMENTATION_DIR must be set}"
: "${ABLATION_DIR:?ABLATION_DIR must be set}"

mkdir -p "$IDEA_CARD_DIR" "$IMPLEMENTATION_DIR" "$ABLATION_DIR/logs" "$ABLATION_DIR/state" "$ABLATION_DIR/reports" "$ABLATION_DIR/artifacts"

if [ "${RUN_RESEARCHSTUDIO:-1}" = "1" ]; then
  echo "[bridge] ResearchStudio IdeaSpark"
  bash "$RESEARCHSTUDIO_DIR/scripts/run_ideaspark_from_config.sh" "$RESEARCHSTUDIO_CONFIG" \
    > "$ABLATION_DIR/logs/researchstudio_ideaspark.log" 2>&1
fi

PREFIX=${OUTPUT_PREFIX:-idea}
if [ -f "$IDEA_CARD_DIR/${PREFIX}.detail.en.md" ] && [ "$IDEA_CARD_DIR/${PREFIX}.detail.en.md" != "$IDEA_CARD_DIR/idea.detail.en.md" ]; then
  cp "$IDEA_CARD_DIR/${PREFIX}.detail.en.md" "$IDEA_CARD_DIR/idea.detail.en.md"
fi
if [ -f "$IDEA_CARD_DIR/${PREFIX}.std.en.md" ] && [ "$IDEA_CARD_DIR/${PREFIX}.std.en.md" != "$IDEA_CARD_DIR/idea.std.en.md" ]; then
  cp "$IDEA_CARD_DIR/${PREFIX}.std.en.md" "$IDEA_CARD_DIR/idea.std.en.md"
fi
if [ -f "$IDEA_CARD_DIR/${PREFIX}.std.zh.md" ] && [ "$IDEA_CARD_DIR/${PREFIX}.std.zh.md" != "$IDEA_CARD_DIR/idea.std.zh.md" ]; then
  cp "$IDEA_CARD_DIR/${PREFIX}.std.zh.md" "$IDEA_CARD_DIR/idea.std.zh.md"
fi
if [ -f "$IDEA_CARD_DIR/ideaspark_run_auto/phase4/method_view.json" ] && [ "$IDEA_CARD_DIR/ideaspark_run_auto/phase4/method_view.json" != "$IDEA_CARD_DIR/method_view.json" ]; then
  cp "$IDEA_CARD_DIR/ideaspark_run_auto/phase4/method_view.json" "$IDEA_CARD_DIR/method_view.json"
fi
if [ -f "$IDEA_CARD_DIR/ideaspark_run_auto/phase4/phase4_implementability.json" ] && [ "$IDEA_CARD_DIR/ideaspark_run_auto/phase4/phase4_implementability.json" != "$IDEA_CARD_DIR/phase4_implementability.json" ]; then
  cp "$IDEA_CARD_DIR/ideaspark_run_auto/phase4/phase4_implementability.json" "$IDEA_CARD_DIR/phase4_implementability.json"
fi

if [ "${RUN_CODEX_PROTOTYPE:-1}" = "1" ]; then
  echo "[bridge] Codex prototype"
  bash "$SCRIPT_DIR/run_codex_prototype_from_config.sh" "$CONFIG"
fi

if [ ! -d "$ABLATION_DIR/implementation_initial_snapshot" ] && [ -n "$(find "$IMPLEMENTATION_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
  mkdir -p "$ABLATION_DIR/implementation_initial_snapshot"
  cp -a "$IMPLEMENTATION_DIR"/. "$ABLATION_DIR/implementation_initial_snapshot"/
fi

if [ "${RUN_HEURESIS_LOOP:-1}" = "1" ]; then
  echo "[bridge] Heuresis loop"
  GENERATED_CONFIG="$ABLATION_DIR/heuresis_loop.env"
  cat > "$GENERATED_CONFIG" <<EOF
ROOT_DIR=$ROOT_DIR
PROJECT_ROOT=$PROJECT_ROOT
TASK_SPEC=$TASK_SPEC
DATASET_DIR=${DATASET_DIR:-}
DATASET_REF=${DATASET_REF:-}
HEURESIS_DIR=${HEURESIS_DIR:-$ROOT_DIR/Heuresis_PJLAB-boyue}
IMPLEMENTATION_DIR=$IMPLEMENTATION_DIR
CONTROLLER_DIR=$SCRIPT_DIR
PYTHON=$PYTHON
CODEX_BIN=$CODEX_BIN
HEURESIS_ENV_FILE=$HEURESIS_ENV_FILE

CONTEXT_TEXT_FILES=$HUMAN_IDEA_DIR/*.md:$IDEA_CARD_DIR/*.md:$IMPLEMENTATION_DIR/README.md:$IMPLEMENTATION_DIR/IMPLEMENTATION_PLAN.md:$IMPLEMENTATION_DIR/ASSUMPTIONS_AND_DEVIATIONS.md:$IMPLEMENTATION_DIR/reports/*.md:$ADDITIONAL_SKILLS_DIR/**/*.md
CONTEXT_JSON_FILES=$IDEA_CARD_DIR/*.json:$IMPLEMENTATION_DIR/artifacts/*.json:$ABLATION_DIR/state/round_*/trial_results.json:$ABLATION_DIR/state/evidence_ledger.json
CONTEXT_CODE_GLOBS=$IMPLEMENTATION_DIR/**/*.py
CONTEXT_MAX_CHARS_PER_FILE=30000

LOOP_START_ROUND=${LOOP_START_ROUND:-0}
LOOP_MAX_ROUNDS=${LOOP_MAX_ROUNDS:-3}
CONTINUE_ON_TRIAL_FAILURE=${CONTINUE_ON_TRIAL_FAILURE:-1}
MAX_TRIALS_PER_ROUND=${MAX_TRIALS_PER_ROUND:-1}
MAX_CODE_CHANGES_PER_ROUND=8
ROUND_PREFIX=round
STATE_DIR=$ABLATION_DIR/state
REPORT_DIR=$ABLATION_DIR/reports
ARTIFACT_DIR=$ABLATION_DIR/artifacts
EVIDENCE_LEDGER=$ABLATION_DIR/state/evidence_ledger.json

HEURESIS_SUGGESTION_MODE=${HEURESIS_SUGGESTION_MODE:-real}
HEURESIS_SUGGESTION_TEMPERATURE=0.25
HEURESIS_SUGGESTION_TIMEOUT_SEC=180
HEURESIS_SUGGESTION_MAX_RETRIES=2
HEURESIS_SUGGESTION_JSON_RETRIES=2
SUGGESTION_SCHEMA=$SCRIPT_DIR/schemas/proposal_v2.schema.json

CODEX_MODEL=${CODEX_MODEL:-}
CODEX_SANDBOX=workspace-write
CODEX_APPROVAL_POLICY=never
CODEX_ALLOWED_DIR=$IMPLEMENTATION_DIR
CODEX_APPLY_MODE=${CODEX_APPLY_MODE:-real}
RUN_CODEX_APPLY=${RUN_CODEX_APPLY:-1}
RUN_STATIC_CHECK=1
STATIC_CHECK_COMMAND=

RUN_UPSTREAM=0
RUN_HEURESIS_SUGGEST=1
RUN_VALIDATE_SUGGESTION=1
RUN_TRIALS=1
RUN_SUMMARIZE_ROUND=1
RUN_UPDATE_LEDGER=1
DRY_RUN=0
RESUME_EXISTING=0

TRIAL_EXECUTION_MODE=${TRIAL_EXECUTION_MODE:-simulated}
TRIAL_WORKDIR=$IMPLEMENTATION_DIR
TRIAL_PYTHONPATH=$IMPLEMENTATION_DIR
TRIAL_WORKDIR_IN_PACKAGE=${TRIAL_WORKDIR_IN_PACKAGE:-}
TRIAL_PYTHONPATH_IN_PACKAGE=${TRIAL_PYTHONPATH_IN_PACKAGE:-}
COMMAND_IMPLEMENTATION_DIR_IN_PACKAGE=${COMMAND_IMPLEMENTATION_DIR_IN_PACKAGE:-.}
DATASET_DIR_IN_PACKAGE=${DATASET_DIR_IN_PACKAGE:-}
PROJECT_ROOT_IN_PACKAGE=${PROJECT_ROOT_IN_PACKAGE:-}
TRIAL_ARTIFACT_SUBDIR=controller_trials
TRIAL_COMMAND_TEMPLATE='python3 - <<'"'"'PY'"'"'
import json
from pathlib import Path
Path("{output}").write_text(json.dumps({{"status": "skipped_template", "trial": {trial_json}}}, indent=2))
print("{output}")
PY'

RJOB_NAMESPACE=${RJOB_NAMESPACE:-ailab-deepdivegzy}
RJOB_CHARGED_GROUP=${RJOB_CHARGED_GROUP:-deepdivegzy_gpu}
RJOB_IMAGE=${RJOB_IMAGE:-registry.h.pjlab.org.cn/ailab-deepdivegzy-deepdivegzy_gpu/slime:v0}
RJOB_GPU_LIMIT=${RJOB_GPU_LIMIT:-2}
RJOB_GPU_PER_TRIAL=${RJOB_GPU_PER_TRIAL:-1}
RJOB_CPU=${RJOB_CPU:-20}
RJOB_MEMORY=${RJOB_MEMORY:-20000}
RJOB_MOUNT=${RJOB_MOUNT-gpfs://gpfs2/beam-gpfs02:/mnt/shared-storage-gpfs2/beam-gpfs02}
RJOB_PRIVATE_MACHINE=${RJOB_PRIVATE_MACHINE:-group}
RJOB_HOST_NETWORK=${RJOB_HOST_NETWORK:-false}
RJOB_TAIL_LINES=500
RJOB_LOG_RETRIES=3
RJOB_LOG_RETRY_INTERVAL_SEC=10
RJOB_POLL_INTERVAL_SEC=${RJOB_POLL_INTERVAL_SEC:-60}
RJOB_TIMEOUT_MIN=${RJOB_TIMEOUT_MIN:-180}
RJOB_SYNC_SOURCE=${RJOB_SYNC_SOURCE:-$PROJECT_ROOT}
RJOB_SHARED_FOLDER=${RJOB_SHARED_FOLDER:-${RJOB_FOLDER:-}}
RJOB_SYNC_EXCLUDES=${RJOB_SYNC_EXCLUDES:-.git/:.venv/:__pycache__/:*.pyc}
RJOB_SYNC_PATHS=${RJOB_SYNC_PATHS:-.}
RJOB_CANCEL_ON_TIMEOUT=${RJOB_CANCEL_ON_TIMEOUT:-1}
RJOB_FOLDER=
EOF
  CONTROLLER_CONFIG="$GENERATED_CONFIG" bash "$SCRIPT_DIR/run_loop.sh" "${LOOP_MAX_ROUNDS:-3}" \
    > "$ABLATION_DIR/logs/heuresis_loop.log" 2>&1
fi

cat > "$ABLATION_DIR/e2e_summary.md" <<EOF
# ResearchStudio -> Codex -> Heuresis E2E Summary

- project: \`$PROJECT_ROOT\`
- idea card: \`$IDEA_CARD_DIR/idea.detail.en.md\`
- implementation: \`$IMPLEMENTATION_DIR\`
- initial snapshot: \`$ABLATION_DIR/implementation_initial_snapshot\`
- heuresis state: \`$ABLATION_DIR/state\`
- logs: \`$ABLATION_DIR/logs\`
- trial mode: \`${TRIAL_EXECUTION_MODE:-simulated}\`
EOF

echo "[bridge] done"
echo "$ABLATION_DIR/e2e_summary.md"
