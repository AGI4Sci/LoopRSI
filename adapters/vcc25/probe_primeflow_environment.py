"""Read-only PRiMeFlow runtime and H1 train/validation asset probe."""

from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path
from typing import Any, Dict


DEFAULT_ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/"
    "naturebench/lingshu_cell_h1/assets/official_2025"
)


def build_report(train: Path, validation: Path) -> Dict[str, Any]:
    modules = ("torch", "anndata", "perturbench", "primeflow")
    try:
        import torch
        torch_version = str(torch.__version__)
    except ImportError:
        torch_version = None
    return {
        "schema_version": "vcc25.primeflow-environment-probe/v1",
        "assets_visible": {
            "train": train.is_file(),
            "validation": validation.is_file(),
        },
        "runtime_modules": {
            name: importlib.util.find_spec(name) is not None for name in modules
        },
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "expression_matrix_read": False,
        "final_test_expression_read": False,
    }


def main() -> None:
    report = build_report(
        DEFAULT_ROOT / "train" / "adata_Training.h5ad",
        DEFAULT_ROOT / "validation" / "adata_Validation.h5ad",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
