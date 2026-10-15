"""Verify shard provenance against real H1 expression and train controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    args = parser.parse_args()
    import anndata as ad
    import numpy as np

    source = ad.read_h5ad(args.dataset, backed="r")
    condition = source.obs["condition"].astype(str)
    split = source.obs["split"].astype(str)
    controls = np.flatnonzero((condition == "ctrl") & (split == "train"))
    control_matrix = source[controls].to_memory().X
    control_hashes = {
        hashlib.sha256(row.toarray().astype("float32").tobytes()).digest()
        for row in control_matrix
    }
    manifest = json.loads((args.shards / "manifest.json").read_text())
    checked = 0
    for entry in manifest["shards"]:
        label = entry["condition"]
        positions = np.flatnonzero((condition == label) & ((split == "train") if label == "ctrl" else True))
        offset = int(entry["file"].removesuffix(".pkl").split("-")[-1])
        chunk = source[positions[offset:offset + entry["cells"]]].to_memory()
        with (args.shards / entry["file"]).open("rb") as stream:
            graphs = pickle.load(stream)
        if len(graphs) != entry["cells"]:
            raise ValueError("graph count mismatch")
        for graph, row in zip(graphs, chunk.X):
            y = row.toarray().astype("float32").reshape(-1)
            if graph.y.shape[-1] != source.n_vars or not np.array_equal(graph.y.numpy().reshape(-1), y):
                raise ValueError("graph target differs from real H1 expression")
            x = graph.x.numpy().astype("float32").reshape(-1)
            if hashlib.sha256(x.tobytes()).digest() not in control_hashes:
                raise ValueError("graph basal expression is not a training control")
            if graph.pert != label:
                raise ValueError("graph perturbation label mismatch")
            checked += 1
    result = {"checked_graphs": checked, "train_controls": len(controls),
              "shards": len(manifest["shards"]), "final_test_expression_read": False}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
