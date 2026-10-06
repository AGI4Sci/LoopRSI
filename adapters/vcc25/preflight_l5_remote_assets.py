"""Read-only preflight of remote L5 train/validation assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.resolve()
    paths = {
        "train_h5ad": root / "assets/official_2025/train/adata_Training.h5ad",
        "validation_h5ad": root / "assets/official_2025/validation/adata_Validation.h5ad",
        "train_targets": root / "assets/official_2025/train/pert_counts_Training.csv",
        "validation_targets": root / "assets/official_2025/validation/pert_counts_Validation.csv",
        "gene_names": root / "assets/official_2025/gene_names.csv",
        "gene_embeddings": root / "assets/lingshu_hf_b77f980/gene_embeddings.npz",
    }
    if any("test" in str(path).lower() for path in paths.values()):
        raise ValueError("test expression is outside this preflight")
    checked = {name: {"path": str(path), "readable": path.is_file(),
                      "bytes": path.stat().st_size if path.is_file() else None}
               for name, path in paths.items()}
    record = {"schema_version": "vcc25.l5-asset-preflight/v1",
              "status": "ready_on_login_node" if all(v["readable"] for v in checked.values()) else "blocked",
              "assets": checked, "expression_matrix_read": False,
              "final_test_expression_read": False, "gpu_container_access_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
