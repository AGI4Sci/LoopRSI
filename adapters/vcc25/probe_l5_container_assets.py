"""Report whether rjob can see the immutable H1 train/validation assets."""
import json
from pathlib import Path


ROOT = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1")
PATHS = {
    "train": ROOT / "assets/official_2025/train/adata_Training.h5ad",
    "validation": ROOT / "assets/official_2025/validation/adata_Validation.h5ad",
    "embedding": ROOT / "assets/lingshu_hf_b77f980/gene_embeddings.npz",
}
print(json.dumps({"schema_version": "vcc25.l5-container-probe/v1",
                  "assets_visible": {key: path.is_file() for key, path in PATHS.items()},
                  "expression_matrix_read": False,
                  "final_test_expression_read": False}, sort_keys=True), flush=True)
