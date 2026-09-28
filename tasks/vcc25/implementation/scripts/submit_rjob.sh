#!/usr/bin/env bash
set -euo pipefail

echo "Deprecated: task code no longer submits rjob jobs." >&2
echo "Use controller_bash/run_loop.sh with TRIAL_EXECUTION_MODE=rjob." >&2
exit 2
