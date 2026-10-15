"""Estimate GEARS graph storage from real H1 metadata and a measured cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _obs_strings(obs, key, h5py):
    value = obs[key]
    if isinstance(value, h5py.Group):
        categories = value["categories"].asstr()[:]
        return [str(categories[code]) if code >= 0 else "" for code in value["codes"][:]]
    return [str(item) for item in value.asstr()[:]]


def plan(dataset: Path, sample_dataset: Path, sample_graphs: Path) -> dict:
    import h5py

    with h5py.File(dataset, "r") as h5:
        conditions = _obs_strings(h5["obs"], "condition", h5py)
        splits = _obs_strings(h5["obs"], "split", h5py)
        genes = len(h5["var"][h5["var"].attrs["_index"]])
    with h5py.File(sample_dataset, "r") as h5:
        sample_cells = len(h5["obs"][h5["obs"].attrs["_index"]])
    if len(conditions) != len(splits) or sample_cells < 1 or genes != 18080:
        raise ValueError("invalid H1 or sample dimensions")
    counts = {
        "train_control": sum(c == "ctrl" and s == "train" for c, s in zip(conditions, splits)),
        "validation_control_excluded": sum(c == "ctrl" and s == "val" for c, s in zip(conditions, splits)),
        "train_perturbed": sum(c != "ctrl" and s == "train" for c, s in zip(conditions, splits)),
        "validation_perturbed": sum(c != "ctrl" and s == "val" for c, s in zip(conditions, splits)),
    }
    if sum(counts.values()) != len(conditions):
        raise ValueError("unexpected H1 split labels")
    modeled_cells = len(conditions) - counts["validation_control_excluded"]
    sample_bytes = sample_graphs.stat().st_size
    return {
        "schema_version": "vcc25.h1-graph-plan/v1",
        "source_cells": len(conditions), "genes": genes,
        "counts": counts, "modeled_cells": modeled_cells,
        "sample_cells": sample_cells, "sample_graph_bytes": sample_bytes,
        "estimated_full_graph_bytes": (sample_bytes * modeled_cells) // sample_cells,
        "method": "linear extrapolation from measured official GEARS sample pickle",
        "caveat": "estimate excludes peak RAM and pickle/Python overhead",
        "final_test_expression_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-dataset", type=Path, required=True)
    parser.add_argument("--sample-graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.dataset, args.sample_dataset, args.sample_graphs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
