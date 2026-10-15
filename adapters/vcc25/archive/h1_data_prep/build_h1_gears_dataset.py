"""Materialize real H1 train/validation views as one GEARS input dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
from anndata.experimental import concat_on_disk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-json", required=True)
    args = parser.parse_args()
    split = json.loads(Path(args.split_json).read_text())
    if set(split) != {"train", "val"} or len(split["train"]) != 150 or len(split["val"]) != 50:
        raise ValueError("expected fixed 150/50 H1 split")
    if set(split["train"]) & set(split["val"]):
        raise ValueError("train and validation targets overlap")

    train = ad.read_h5ad(args.train, backed="r")
    validation = ad.read_h5ad(args.validation, backed="r")
    if train.shape[1] != validation.shape[1] or train.shape[1] != 18080:
        raise ValueError("H1 views must have 18,080 genes with matching dimensions")
    train_conditions = set(train.obs["condition"].astype(str))
    val_conditions = set(validation.obs["condition"].astype(str))
    if train_conditions != {"ctrl", *split["train"]}:
        raise ValueError("training view conditions differ from fixed train targets")
    if val_conditions != {"ctrl", *split["val"]}:
        raise ValueError("validation view conditions differ from fixed validation targets")
    if any("test" in c.lower() or "final" in c.lower() for c in train_conditions | val_conditions):
        raise ValueError("test-like condition found")
    if list(train.var["gene_name"]) != list(validation.var["gene_name"]):
        raise ValueError("train and validation gene order differs")

    gene_names = train.var[["gene_id", "gene_name"]].copy()
    gene_names.index = train.var_names

    args.output.parent.mkdir(parents=True, exist_ok=True)
    concat_on_disk(
        {"train": args.train, "val": args.validation},
        args.output,
        join="inner",
        label="source_split",
        index_unique="-",
        merge=None,
    )
    # Patch only obs metadata in backed mode; never materialize the 320k x 18k
    # expression matrix in memory after the streaming concat.
    merged = ad.read_h5ad(args.output, backed="r+")
    merged.var = gene_names
    conditions = merged.obs["condition"].astype(str)
    merged.obs["condition"] = conditions.map(
        lambda value: "ctrl" if value == "ctrl" else value + "+ctrl"
    )
    merged.obs["condition_name"] = merged.obs["condition"].astype(str)
    merged.obs["split"] = merged.obs["source_split"].astype(str).map(
        {"train": "train", "val": "val"}
    )
    if merged.obs["split"].isna().any():
        raise ValueError("source split labels were not preserved")
    merged.uns["source_scope"] = "real H1 train and validation views; final test excluded"
    merged.write()
    shape = list(merged.shape)
    train_cells = int((merged.obs["split"] == "train").sum())
    validation_cells = int((merged.obs["split"] == "val").sum())
    conditions_count = int(merged.obs["condition"].nunique())
    genes = int(merged.shape[1])
    merged.file.close()
    report = {
        "shape": shape,
        "train_cells": train_cells,
        "validation_cells": validation_cells,
        "conditions": conditions_count,
        "genes": genes,
        "final_test_expression_read": False,
    }
    (args.output.parent / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
