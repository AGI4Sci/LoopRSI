#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?seed required}"
RUN_DIR="${2:?run_dir required}"
ROOT="${3:?root required}"
REQUEST_JSON="${4:?request_json required}"

REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"

RUNNER="$(python3 - "$REQUEST_JSON" <<'PY'
import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runner = request.get("autonomous_runner")
if not isinstance(runner, str) or not runner:
    raise SystemExit("autonomous_runner is required for autonomous_research_candidate")
if Path(runner).is_absolute():
    raise SystemExit("autonomous_runner must be repository-relative")
if ".." in Path(runner).parts:
    raise SystemExit("autonomous_runner must not contain '..'")
print(runner)
PY
)"

RUNNER_PATH="$REPO/$RUNNER"
if [[ ! -f "$RUNNER_PATH" ]]; then
  echo "autonomous runner not found: $RUNNER_PATH" >&2
  exit 2
fi

exec bash "$RUNNER_PATH" "$SEED" "$RUN_DIR" "$ROOT" "$REQUEST_JSON"
