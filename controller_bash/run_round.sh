#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPLICIT_CONTROLLER_CONFIG=${CONTROLLER_CONFIG:-}

OVERRIDE_VARS=(
  DRY_RUN
  RUN_UPSTREAM
  RUN_HEURESIS_SUGGEST
  RUN_VALIDATE_SUGGESTION
  RUN_CODEX_APPLY
  RUN_STATIC_CHECK
  HEURESIS_SUGGESTION_MODE
  CODEX_APPLY_MODE
  CODEX_BIN
  CONTROLLER_CONFIG
  TASK_SPEC
  REQUIRE_ADAPTER_TRIAL_ENTRYPOINT
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
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: missing config: $CONFIG" >&2
  exit 2
fi

set -a
source "$CONFIG"
if [ -z "$EXPLICIT_CONTROLLER_CONFIG" ] && [ -f "$SCRIPT_DIR/.env" ]; then
  source "$SCRIPT_DIR/.env"
fi
if [ -n "${HEURESIS_ENV_FILE:-}" ] && [ -f "$HEURESIS_ENV_FILE" ]; then
  source "$HEURESIS_ENV_FILE"
fi
set +a

for var in "${OVERRIDE_VARS[@]}"; do
  override="USER_OVERRIDE_${var}"
  if [ -n "${!override+x}" ]; then
    export "${var}=${!override}"
  fi
done

ROUND=${1:-0}
ROUND_NAME="${ROUND_PREFIX}_${ROUND}"
ROUND_DIR="$STATE_DIR/$ROUND_NAME"
mkdir -p "$ROUND_DIR" "$REPORT_DIR" "$ARTIFACT_DIR" "$SCRIPT_DIR/logs"

CONTEXT_JSON="$ROUND_DIR/context.json"
SUGGESTION_JSON="$ROUND_DIR/heuresis_suggestion.json"
VALIDATION_JSON="$ROUND_DIR/validation.json"
CODEX_LOG="$ROUND_DIR/codex_apply.log"

echo "[controller] round=$ROUND"
echo "[controller] round_dir=$ROUND_DIR"

if [ -n "${TASK_SPEC:-}" ]; then
  echo "[controller] validate task contract"
  "$PYTHON" "$SCRIPT_DIR/scripts/task_contract.py" validate-task \
    --task-spec "$TASK_SPEC" \
    --out "$ROUND_DIR/task_contract_validation.json"
fi

if [ "${RUN_UPSTREAM:-0}" = "1" ] && [ -n "${UPSTREAM_RUN_COMMAND:-}" ]; then
  echo "[controller] upstream command"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN: $UPSTREAM_RUN_COMMAND"
  else
    bash -lc "$UPSTREAM_RUN_COMMAND" > "$ROUND_DIR/upstream.log" 2>&1
  fi
fi

echo "[controller] collect context"
"$PYTHON" "$SCRIPT_DIR/scripts/collect_context.py" --round "$ROUND" --out "$CONTEXT_JSON"

if [ "${RUN_SKILL_INJECTION:-0}" = "1" ]; then
  echo "[controller] apply bounded skill context"
  "$PYTHON" "$SCRIPT_DIR/scripts/apply_skill_context.py" --context "$CONTEXT_JSON"
fi

if [ "${RUN_HEURESIS_SUGGEST:-1}" = "1" ]; then
  echo "[controller] Heuresis suggestion"
  if [ "${HEURESIS_SUGGESTION_MODE:-real}" = "mock" ]; then
    "$PYTHON" - "$SUGGESTION_JSON" "$ROUND" "$CONTEXT_JSON" <<'PY'
import json
import sys
from pathlib import Path
dst, round_id, context_path = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
context = json.loads(context_path.read_text(encoding="utf-8"))
task_spec = context.get("task_spec") or {}
task_name = (task_spec.get("task") or {}).get("name", "legacy_task")
editable = (task_spec.get("search") or {}).get("editable_paths") or ["."]
obj = {
  "schema_version": "omni-ar-proposal/v2",
  "proposal_id": f"{task_name}-round-{round_id}-mock",
  "task_name": task_name,
  "round": round_id,
  "verdict": "mock generic suggestion for controller validation",
  "evidence_gaps": [],
  "experiment_proposals": [
    {
      "hypothesis": "The configured default trial verifies structured controller plumbing.",
      "change_scope": ["evaluation"],
      "parameters": {},
      "expected_effect": {},
      "acceptance_criteria": {},
      "resource_request": {}
    }
  ],
  "risks": []
}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(obj, indent=2), encoding="utf-8")
print(dst)
PY
  elif [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN: $PYTHON $SCRIPT_DIR/scripts/heuresis_suggest.py --context $CONTEXT_JSON --out $SUGGESTION_JSON"
  else
    "$PYTHON" "$SCRIPT_DIR/scripts/heuresis_suggest.py" \
      --context "$CONTEXT_JSON" \
      --out "$SUGGESTION_JSON"
  fi
fi

if [ "${RUN_VALIDATE_SUGGESTION:-1}" = "1" ] && [ -f "$SUGGESTION_JSON" ]; then
  echo "[controller] validate suggestion"
  "$PYTHON" "$SCRIPT_DIR/scripts/validate_suggestion.py" \
    --suggestion "$SUGGESTION_JSON" \
    --out "$VALIDATION_JSON"
fi

if [ "${RUN_CODEX_APPLY:-0}" = "1" ]; then
  if [ ! -f "$SUGGESTION_JSON" ]; then
    echo "ERROR: no suggestion JSON for Codex: $SUGGESTION_JSON" >&2
    exit 2
  fi
  echo "[controller] Codex apply suggestion"
  PROMPT_FILE="$SCRIPT_DIR/prompts/codex_apply_suggestion.md"
  USER_PROMPT="Suggestion JSON: $SUGGESTION_JSON
Task specification: ${TASK_SPEC:-not configured}
Implementation directory: $IMPLEMENTATION_DIR
Allowed edit directory: $CODEX_ALLOWED_DIR
Loop engineering skill: ${LOOP_ENGINEERING_SKILL:-$ROOT_DIR/skills-w/heuresis-codex-loop/SKILL.md}
Read the controller prompt at $PROMPT_FILE and, if the loop engineering skill exists, read it before editing smoke/rjob/data/checkpoint/path logic. Apply only validated, context-grounded, highest-priority feasible tasks. Keep task semantics inside the configured case implementation. Do not run GPU jobs."
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN: $CODEX_BIN exec --cd $ROOT_DIR --sandbox $CODEX_SANDBOX ..."
  elif [ "${CODEX_APPLY_MODE:-real}" = "mock" ]; then
    {
      echo "MOCK Codex apply"
      echo "suggestion=$SUGGESTION_JSON"
      echo "implementation_dir=$IMPLEMENTATION_DIR"
      echo "allowed_dir=$CODEX_ALLOWED_DIR"
      date -Is
    } | tee "$CODEX_LOG"
  else
    CODEX_ARGS=(exec --skip-git-repo-check --cd "$ROOT_DIR" --sandbox "$CODEX_SANDBOX")
    if [ -n "${CODEX_MODEL:-}" ]; then
      CODEX_ARGS+=(--model "$CODEX_MODEL")
    fi
    env -i \
      HOME="${HOME:-/root}" \
      PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}" \
      LANG="${LANG:-C.UTF-8}" \
      LC_ALL="${LC_ALL:-C.UTF-8}" \
      TERM="${TERM:-dumb}" \
      NO_COLOR="${NO_COLOR:-1}" \
      HTTP_PROXY="${HTTP_PROXY:-}" \
      HTTPS_PROXY="${HTTPS_PROXY:-}" \
      NO_PROXY="${NO_PROXY:-}" \
      http_proxy="${http_proxy:-}" \
      https_proxy="${https_proxy:-}" \
      no_proxy="${no_proxy:-}" \
      CODEX_CI=1 \
      CODEX_ALLOWED_DIR="$CODEX_ALLOWED_DIR" \
      "$CODEX_BIN" "${CODEX_ARGS[@]}" "$USER_PROMPT" | tee "$CODEX_LOG"
  fi
fi

if [ "${RUN_STATIC_CHECK:-1}" = "1" ]; then
  echo "[controller] static check"
  if [ -n "${STATIC_CHECK_COMMAND:-}" ]; then
    bash -lc "$STATIC_CHECK_COMMAND"
  else
    "$PYTHON" "$SCRIPT_DIR/scripts/static_check.py" --root "$IMPLEMENTATION_DIR"
  fi
fi

cat > "$ROUND_DIR/summary.md" <<EOF
# Controller Round $ROUND

- context: \`$CONTEXT_JSON\`
- suggestion: \`$SUGGESTION_JSON\`
- validation: \`$VALIDATION_JSON\`
- codex_log: \`$CODEX_LOG\`
- run_codex_apply: \`${RUN_CODEX_APPLY:-0}\`
EOF

echo "[controller] done"
echo "$ROUND_DIR/summary.md"
