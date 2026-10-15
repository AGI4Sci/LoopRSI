"""Report the bounded PRESAGE rjob environment without reading expression data."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path


OFFICIAL_TRAIN = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/"
    "naturebench/lingshu_cell_h1/assets/official_2025/train/adata_Training.h5ad"
)


def main() -> None:
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch", "anndata", "pandas", "scipy", "sklearn", "scanpy",
            "pytorch_lightning", "torch_geometric",
        )
    }
    torch_info = {"cuda_available": False, "version": None, "gpu": None}
    if modules["torch"]:
        import torch

        torch_info["version"] = torch.__version__
        torch_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            torch_info["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps({
        "schema_version": "vcc25.presage-environment-probe/v1",
        "python": platform.python_version(),
        "charged_group": os.environ.get("LOOPRSI_CHARGED_GROUP"),
        "official_training_visible": OFFICIAL_TRAIN.is_file(),
        "official_training_expression_read": False,
        "final_test_expression_read": False,
        "modules": modules,
        "torch": torch_info,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
