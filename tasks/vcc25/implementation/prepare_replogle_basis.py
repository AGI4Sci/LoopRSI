from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from official_h1_contract import read_condition_targets, read_gene_names


SOURCE_SHA256 = "412fd0df8c4ccea9f4db91cd88033c49200838b29d40945e48574be588b48789"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Replogle source hash mismatch")
    genes = read_gene_names(args.gene_names)
    validation = read_condition_targets(args.validation_targets)
    test = read_condition_targets(args.test_targets)
    forbidden = validation | test
    data = ad.read_h5ad(args.source, backed="r")
    source_genes = {str(gene): index for index, gene in enumerate(data.var_names)}
    shared = [(index, source_genes[gene]) for index, gene in enumerate(genes) if gene in source_genes]
    target_values = np.asarray(data.obs["perturbation"].astype(str))
    source_columns = np.asarray([source for _, source in shared], dtype=np.int64)
    targets = sorted(set(target_values) - {"control"} - forbidden)
    aggregate_targets = ["control", *targets]
    target_index = {target: index for index, target in enumerate(aggregate_targets)}
    sums = np.zeros((len(aggregate_targets), len(shared)), dtype=np.float64)
    counts_all = np.zeros(len(aggregate_targets), dtype=np.int64)
    for start in range(0, data.n_obs, args.chunk_rows):
        stop = min(start + args.chunk_rows, data.n_obs)
        labels = target_values[start:stop]
        values = np.asarray(data.X[start:stop, :])[:, source_columns]
        for target in set(labels):
            if target not in target_index:
                continue
            mask = labels == target
            index = target_index[target]
            sums[index] += values[mask].sum(axis=0, dtype=np.float64)
            counts_all[index] += int(mask.sum())
    means = sums / np.maximum(counts_all[:, None], 1)
    control = means[0].astype(np.float32)
    deltas = np.zeros((len(targets), len(genes)), dtype=np.float32)
    for row, target in enumerate(targets):
        values = means[row + 1].astype(np.float32) - control
        deltas[row, np.asarray([dest for dest, _ in shared])] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, targets=np.asarray(targets), genes=np.asarray(genes), delta=deltas, counts=counts_all[1:].astype(np.int32))
    manifest = {
        "dataset_id": "scperturb-replogle-k562-essential-7416068-v1",
        "source": "Zenodo record 7416068 / ReplogleWeissman2022_K562_essential.h5ad",
        "license": "CC BY 4.0",
        "source_sha256": SOURCE_SHA256,
        "artifact_sha256": sha256(args.output),
        "cells": int(data.n_obs), "source_genes": int(data.n_vars),
        "h1_genes_mapped": len(shared), "targets_retained": len(targets),
        "excluded_validation_targets": sorted(set(target_values) & validation),
        "excluded_test_targets": sorted(set(target_values) & test),
        "test_expression_used": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1/assets/official_2025")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gene-names", type=Path, default=root / "gene_names.csv")
    parser.add_argument("--validation-targets", type=Path, default=root / "validation/pert_counts_Validation.csv")
    parser.add_argument("--test-targets", type=Path, default=root / "test/pert_counts_Test.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
