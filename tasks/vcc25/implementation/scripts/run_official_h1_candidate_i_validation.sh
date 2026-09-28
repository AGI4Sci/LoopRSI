#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
OUTPUT=${1:?output JSON path is required}
RUNTIME_SITE=/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1/runs/lingshu_vcc_reproduction_b77f980_20260908/py312_infer_env/lib/python3.12/site-packages

cd "$ROOT"
test -d "$RUNTIME_SITE/h5py"
export PYTHONPATH="$ROOT/tasks/vcc25/implementation:$ROOT:$RUNTIME_SITE:${PYTHONPATH:-}"
mkdir -p "$(dirname "$OUTPUT")"
python -c 'import h5py, numpy, torch; assert torch.cuda.is_available(); print({"h5py": h5py.__version__, "numpy": numpy.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)})'
python tasks/vcc25/implementation/official_h1_candidate_i_pathway_basis.py --output "$OUTPUT"
