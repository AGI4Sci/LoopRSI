#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPLICIT_CONTROLLER_CONFIG=${CONTROLLER_CONFIG:-}

OVERRIDE_VARS=(
  CONTROLLER_CONFIG
  TASK_SPEC
  REQUIRE_ADAPTER_TRIAL_ENTRYPOINT
  LOOP_START_ROUND
  LOOP_MAX_ROUNDS
  CONTINUE_ON_TRIAL_FAILURE
  DRY_RUN
  HEURESIS_SUGGESTION_MODE
  CODEX_APPLY_MODE
  CODEX_BIN
  RUN_CODEX_APPLY
  RUN_TRIALS
  RUN_SUMMARIZE_ROUND
  RUN_UPDATE_LEDGER
  MAX_TRIALS_PER_ROUND
  TRIAL_EXECUTION_MODE
  TRIAL_COMMAND_TEMPLATE
  TRIAL_ARTIFACT_SUBDIR
  TRIAL_WORKDIR
  TRIAL_PYTHONPATH
  TRIAL_WORKDIR_IN_PACKAGE
  TRIAL_PYTHONPATH_IN_PACKAGE
  COMMAND_IMPLEMENTATION_DIR_IN_PACKAGE
  DATASET_DIR_IN_PACKAGE
  PROJECT_ROOT_IN_PACKAGE
  RJOB_NAMESPACE
  RJOB_CHARGED_GROUP
  RJOB_IMAGE
  RJOB_GPU_LIMIT
  RJOB_GPU_PER_TRIAL
  RJOB_CPU
  RJOB_MEMORY
  RJOB_MOUNT
  RJOB_PRIVATE_MACHINE
  RJOB_HOST_NETWORK
  RJOB_TAIL_LINES
  RJOB_LOG_RETRIES
  RJOB_LOG_RETRY_INTERVAL_SEC
  RJOB_POLL_INTERVAL_SEC
  RJOB_TIMEOUT_MIN
  RJOB_FOLDER
  RJOB_SYNC_SOURCE
  RJOB_SHARED_FOLDER
  RJOB_SYNC_EXCLUDES
  RJOB_SYNC_PATHS
  RJOB_CANCEL_ON_TIMEOUT
)
for var in "${OVERRIDE_VARS[@]}"; do
  if [ -n "${!var+x}" ]; then
    export "USER_OVERRIDE_${var}=${!var}"
  fi
done

if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi
if [ -n "$EXPLICIT_CONTROLLER_CONFIG" ]; then
  export CONTROLLER_CONFIG="$EXPLICIT_CONTROLLER_CONFIG"
fi
CONFIG=${CONTROLLER_CONFIG:-"$SCRIPT_DIR/config.env"}
set -a
source "$CONFIG"
if [ -z "$EXPLICIT_CONTROLLER_CONFIG" ] && [ -f "$SCRIPT_DIR/.env" ]; then
  source "$SCRIPT_DIR/.env"
fi
set +a

for var in "${OVERRIDE_VARS[@]}"; do
  override="USER_OVERRIDE_${var}"
  if [ -n "${!override+x}" ]; then
    export "${var}=${!override}"
  fi
done

MAX_ROUNDS=${1:-${LOOP_MAX_ROUNDS:-3}}
if [ "$MAX_ROUNDS" -gt "${LOOP_MAX_ROUNDS:-3}" ]; then
  echo "ERROR: requested $MAX_ROUNDS rounds exceeds LOOP_MAX_ROUNDS=${LOOP_MAX_ROUNDS:-3}" >&2
  exit 2
fi
START_ROUND=${LOOP_START_ROUND:-0}
if [ "$START_ROUND" -lt 0 ] || [ "$START_ROUND" -ge "$MAX_ROUNDS" ]; then
  echo "ERROR: LOOP_START_ROUND=$START_ROUND must satisfy 0 <= start < max_rounds=$MAX_ROUNDS" >&2
  exit 2
fi

echo "[loop] start_round=$START_ROUND max_rounds=$MAX_ROUNDS"
LOOP_HAD_TRIAL_FAILURE=0
for ((round=START_ROUND; round<MAX_ROUNDS; round++)); do
  echo "[loop] ===== round $round ====="
  bash "$SCRIPT_DIR/run_round.sh" "$round"

  ROUND_NAME="${ROUND_PREFIX}_${round}"
  ROUND_DIR="$STATE_DIR/$ROUND_NAME"
  SUGGESTION_JSON="$ROUND_DIR/heuresis_suggestion.json"
  VALIDATION_JSON="$ROUND_DIR/validation.json"
  CONTEXT_JSON="$ROUND_DIR/context.json"
  TRIAL_RESULTS="$ROUND_DIR/trial_results.json"
  SUMMARY_JSON="$ROUND_DIR/round_summary.json"
  SUMMARY_MD="$REPORT_DIR/heuresis_codex_${ROUND_NAME}.md"

  if [ "${RUN_TRIALS:-1}" = "1" ]; then
    echo "[loop] run trials"
    set +e
    "$PYTHON" "$SCRIPT_DIR/scripts/run_trials.py" \
      --suggestion "$SUGGESTION_JSON" \
      --round-dir "$ROUND_DIR" \
      --artifact-dir "$ARTIFACT_DIR" \
      --out "$TRIAL_RESULTS"
    TRIAL_RC=$?
    set -e
    if [ "$TRIAL_RC" -ne 0 ]; then
      LOOP_HAD_TRIAL_FAILURE=1
      echo "[loop] trial step failed with rc=$TRIAL_RC"
      if [ "${CONTINUE_ON_TRIAL_FAILURE:-0}" != "1" ]; then
        exit "$TRIAL_RC"
      fi
    fi
  fi

  if [ "${RUN_SUMMARIZE_ROUND:-1}" = "1" ]; then
    echo "[loop] summarize round"
    "$PYTHON" "$SCRIPT_DIR/scripts/summarize_round.py" \
      --round "$round" \
      --round-dir "$ROUND_DIR" \
      --suggestion "$SUGGESTION_JSON" \
      --trial-results "$TRIAL_RESULTS" \
      --out-md "$SUMMARY_MD" \
      --out-json "$SUMMARY_JSON"
  fi

  if [ "${RUN_UPDATE_LEDGER:-1}" = "1" ]; then
    echo "[loop] update evidence ledger"
    "$PYTHON" "$SCRIPT_DIR/scripts/update_evidence_ledger.py" \
      --ledger "$EVIDENCE_LEDGER" \
      --round "$round" \
      --round-dir "$ROUND_DIR" \
      --context "$CONTEXT_JSON" \
      --suggestion "$SUGGESTION_JSON" \
      --validation "$VALIDATION_JSON" \
      --trial-results "$TRIAL_RESULTS" \
      --summary-json "$SUMMARY_JSON"
  fi
done

echo "[loop] done"
echo "$EVIDENCE_LEDGER"
if [ "$LOOP_HAD_TRIAL_FAILURE" -ne 0 ]; then
  echo "[loop] completed with one or more failed trial step(s)"
fi
