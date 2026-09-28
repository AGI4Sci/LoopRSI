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

: "${ROOT_DIR:?ROOT_DIR must be set}"
: "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
: "${TASK_SPEC:?TASK_SPEC must point to a validated task_spec.yaml}"
: "${HUMAN_IDEA_DIR:?HUMAN_IDEA_DIR must be set}"
: "${DATASET_DIR:?DATASET_DIR must be set}"
: "${IDEA_CARD_DIR:?IDEA_CARD_DIR must be set}"
: "${IMPLEMENTATION_DIR:?IMPLEMENTATION_DIR must be set}"
: "${ADDITIONAL_SKILLS_DIR:?ADDITIONAL_SKILLS_DIR must be set}"
: "${ABLATION_DIR:?ABLATION_DIR must be set}"
: "${CODEX_BIN:?CODEX_BIN must be set}"

mkdir -p "$IMPLEMENTATION_DIR" "$ABLATION_DIR/logs"

if [ ! -s "${IDEA_CARD_MD:-$IDEA_CARD_DIR/idea.detail.en.md}" ]; then
  echo "ERROR: idea card not found: ${IDEA_CARD_MD:-$IDEA_CARD_DIR/idea.detail.en.md}" >&2
  exit 2
fi

PROMPT_FILE="$SCRIPT_DIR/prompts/codex_build_prototype.md"
LOG_FILE="$ABLATION_DIR/logs/codex_prototype.log"
USER_PROMPT="Project root: $PROJECT_ROOT
Task specification: $TASK_SPEC
Human idea directory: $HUMAN_IDEA_DIR
Dataset directory: $DATASET_DIR
ResearchStudio idea-card directory: $IDEA_CARD_DIR
Implementation directory: $IMPLEMENTATION_DIR
Additional skills directory: $ADDITIONAL_SKILLS_DIR
Loop engineering skill: ${LOOP_ENGINEERING_SKILL:-$ROOT_DIR/skills-w/heuresis-codex-loop/SKILL.md}
Prototype rjob permission: ${CODEX_PROTOTYPE_ALLOW_RJOB:-0}
Trial execution mode for later Heuresis loop: ${TRIAL_EXECUTION_MODE:-simulated}

Read the fixed builder prompt at $PROMPT_FILE. Also read the loop engineering skill if it exists, and read any human requirement file under $HUMAN_IDEA_DIR, especially requirement.md if present. Build the smallest runnable prototype in IMPLEMENTATION_DIR only.

Important execution boundary: unless Prototype rjob permission is exactly 1, do not run rjob commands and do not execute generated submit/smoke/full scripts. For bridge smoke, only write those scripts and run static-safe checks."

CODEX_ARGS=(exec --skip-git-repo-check --cd "$ROOT_DIR" --sandbox workspace-write)
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
  CODEX_PROTOTYPE_ALLOW_RJOB="${CODEX_PROTOTYPE_ALLOW_RJOB:-0}" \
  TASK_SPEC="$TASK_SPEC" \
  TRIAL_EXECUTION_MODE="${TRIAL_EXECUTION_MODE:-simulated}" \
  CODEX_ALLOWED_DIR="$IMPLEMENTATION_DIR" \
  "$CODEX_BIN" "${CODEX_ARGS[@]}" "$USER_PROMPT" | tee "$LOG_FILE"

echo "$LOG_FILE"
