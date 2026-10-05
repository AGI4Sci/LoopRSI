"""Build a small, real H1 GEARS smoke dataset without validation controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-target", required=True)
    parser.add_argument("--val-target", required=True)
    parser.add_argument("--cells-per-condition", type=int, default=100)
    args = parser.parse_args()
    if args.cells_per_condition < 1:
        raise ValueError("cells-per-condition must be positive")
    if args.train_target == args.val_target or any(
        token in str(args.dataset).lower() for token in ("test_expression", "adata_test", "final_test")
    ):
        raise ValueError("targets must differ and final-test expression is forbidden")
    if args.output.resolve() == args.dataset.resolve().parent:
        raise ValueError("smoke output must not overwrite its source dataset")
    if (args.output / "perturb_processed.h5ad").exists():
        raise FileExistsError("smoke dataset already exists")
    source = ad.read_h5ad(args.dataset, backed="r")
    conditions = source.obs["condition"].astype(str)
    split = source.obs["split"].astype(str)
    wanted = (
        (conditions == "ctrl") & (split == "train")
    ) | (
        (conditions == args.train_target + "+ctrl") & (split == "train")
    ) | (
        (conditions == args.val_target + "+ctrl") & (split == "val")
    )
    indices = []
    counts = {"ctrl": 0, "train": 0, "val": 0}
    for i, keep in enumerate(wanted):
        if not keep:
            continue
        label = "ctrl" if conditions.iloc[i] == "ctrl" else split.iloc[i]
        if counts[label] < args.cells_per_condition:
            indices.append(i)
            counts[label] += 1
    if any(count != args.cells_per_condition for count in counts.values()):
        raise ValueError(f"insufficient real H1 cells: {counts}")
    selected = source[indices].to_memory()
    selected.obs["split"] = selected.obs["split"].astype(str)
    args.output.mkdir(parents=True, exist_ok=True)
    selected.write_h5ad(args.output / "perturb_processed.h5ad", compression="gzip")
    report = {
        "shape": list(selected.shape),
        "conditions": sorted(selected.obs["condition"].astype(str).unique()),
        "splits": sorted(selected.obs["split"].astype(str).unique()),
        "source": str(args.dataset),
        "cells_per_condition": args.cells_per_condition,
        "training_controls_only": True,
        "validation_controls_included": bool(((selected.obs["condition"].astype(str) == "ctrl") & (selected.obs["split"].astype(str) == "val")).any()),
        "final_test_expression_read": False,
    }
    (args.output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
