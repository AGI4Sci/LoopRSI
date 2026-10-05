"""Build bounded official GEARS graph shards from real H1 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gene2go", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-cells", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args()
    if args.chunk_cells < 1 or args.output.resolve() == args.dataset.resolve().parent:
        raise ValueError("invalid chunk size or output directory")
    if any(token in str(args.dataset).lower() for token in ("test_expression", "adata_test", "final_test")):
        raise ValueError("final-test expression is forbidden")

    import anndata as ad
    import numpy as np

    sys.path.insert(0, str(args.source_repository.resolve()))
    from gears import PertData

    source = ad.read_h5ad(args.dataset, backed="r")
    condition = source.obs["condition"].astype(str)
    split = source.obs["split"].astype(str)
    if set(split) - {"train", "val"}:
        raise ValueError("unexpected split labels")
    controls = np.flatnonzero((condition == "ctrl") & (split == "train"))
    if not len(controls):
        raise ValueError("training controls are required")
    selected = sorted(set(args.conditions or condition.unique()))
    if "ctrl" not in selected or set(selected) - set(condition):
        raise ValueError("conditions must include ctrl and exist in source")

    args.output.mkdir(parents=True, exist_ok=True)
    go_root = args.output / "go"
    go_root.mkdir(exist_ok=True)
    go_path = go_root / "gene2go_all.pkl"
    if not go_path.exists():
        go_path.symlink_to(args.gene2go.resolve())
    pert_data = PertData(str(go_root), default_pert_graph=False)
    pert_data.adata = source
    pert_data.set_pert_genes()
    pert_data.ctrl_adata = source[controls].to_memory()
    pert_data.gene_names = source.var["gene_name"]
    manifest = {
        "schema_version": "vcc25.gears-shards/v1", "source": str(args.dataset.resolve()),
        "genes": source.n_vars, "train_controls": len(controls), "seed": args.seed,
        "chunk_cells": args.chunk_cells, "shards": [], "final_test_expression_read": False,
    }
    for label in selected:
        positions = np.flatnonzero((condition == label) & ((split == "train") if label == "ctrl" else True))
        expected_split = "train" if label == "ctrl" else split.iloc[positions[0]]
        if bool((split.iloc[positions] != expected_split).any()):
            raise ValueError(f"condition crosses splits: {label}")
        for offset in range(0, len(positions), args.chunk_cells):
            chunk = positions[offset:offset + args.chunk_cells]
            digest = hashlib.sha256(label.encode()).hexdigest()[:12]
            name = f"{digest}-{offset:08d}.pkl"
            path = args.output / name
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing shard: {path}")
            np.random.seed((args.seed + int(digest, 16) + offset) % (2**32))
            subset = source[chunk].to_memory()
            graphs = pert_data.create_cell_graph_dataset(subset, label)
            if len(graphs) != len(chunk):
                raise ValueError("official GEARS graph count differs from source cells")
            with path.open("wb") as stream:
                pickle.dump(graphs, stream, protocol=pickle.HIGHEST_PROTOCOL)
            manifest["shards"].append({
                "condition": label, "split": expected_split, "cells": len(chunk),
                "file": name, "bytes": path.stat().st_size,
            })
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            print(json.dumps(manifest["shards"][-1], sort_keys=True), flush=True)
    source.file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
