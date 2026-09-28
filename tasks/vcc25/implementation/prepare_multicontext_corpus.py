from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

from official_h1_contract import read_condition_targets, read_gene_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dense(block) -> np.ndarray:
    return block.toarray() if hasattr(block, "toarray") else np.asarray(block)


def aggregate_source(
    spec: dict,
    genes: list[str],
    forbidden: set[str],
    official_targets: set[str],
    chunk_rows: int,
) -> dict:
    path = Path(spec["path"])
    actual_hash = sha256(path)
    if actual_hash != spec["sha256"]:
        raise ValueError(f"source hash mismatch for {path}: {actual_hash}")
    data = ad.read_h5ad(path, backed="r")
    pert_key = spec.get("perturbation_column", "perturbation")
    if pert_key not in data.obs:
        raise ValueError(f"missing perturbation column {pert_key!r} in {path}")
    labels = np.asarray(data.obs[pert_key].astype(str)).copy()
    controls = set(spec.get("control_labels", ["control", "non-targeting"]))
    control_mask = np.isin(labels, list(controls))
    if spec.get("control_column"):
        control_mask |= np.asarray(data.obs[spec["control_column"]], dtype=bool)
    if not control_mask.any():
        raise ValueError(f"no controls found in {path}")
    labels[control_mask] = "__control__"
    if spec.get("source_gene_column"):
        source_gene_names = np.asarray(data.var[spec["source_gene_column"]].astype(str))
    else:
        source_gene_names = np.asarray(data.var_names.astype(str))
    source_gene_index = {str(gene): index for index, gene in enumerate(source_gene_names)}
    shared = [(dest, source_gene_index[gene]) for dest, gene in enumerate(genes) if gene in source_gene_index]
    if not shared:
        raise ValueError(f"no H1 genes map into {path}")
    source_targets = set(labels) - {"__control__"}
    if spec.get("official_targets_only", False):
        source_targets &= official_targets
    targets = sorted(source_targets - forbidden)
    names = ["__control__", *targets]
    name_index = {name: index for index, name in enumerate(names)}
    sums = np.zeros((len(names), len(shared)), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)
    source_columns = np.asarray([source for _, source in shared], dtype=np.int64)
    matrix = data.layers[spec["matrix_layer"]] if spec.get("matrix_layer") else data.X
    for start in range(0, data.n_obs, chunk_rows):
        stop = min(start + chunk_rows, data.n_obs)
        block_labels = labels[start:stop]
        if spec.get("official_targets_only", False):
            selected = np.isin(block_labels, names)
            if not selected.any():
                continue
            row_indices = start + np.flatnonzero(selected)
            values = _dense(matrix[row_indices][:, source_columns]).astype(np.float32, copy=False)
            block_labels = block_labels[selected]
        else:
            values = _dense(matrix[start:stop, source_columns]).astype(np.float32, copy=False)
        for label in set(block_labels):
            normalized = label
            if normalized not in name_index:
                continue
            selected = block_labels == label
            index = name_index[normalized]
            sums[index] += values[selected].sum(axis=0, dtype=np.float64)
            counts[index] += int(selected.sum())
    if counts[0] == 0:
        raise ValueError(f"control count is zero in {path}")
    means = sums / np.maximum(counts[:, None], 1)
    delta = np.zeros((len(targets), len(genes)), dtype=np.float32)
    destinations = np.asarray([dest for dest, _ in shared], dtype=np.int64)
    delta[:, destinations] = (means[1:] - means[0]).astype(np.float32)
    return {
        "context": spec["context"], "targets": targets, "delta": delta,
        "counts": counts[1:].astype(np.int32), "source_sha256": actual_hash,
        "source_cells": int(data.n_obs), "source_genes": int(data.n_vars),
        "mapped_h1_genes": len(shared), "excluded_targets": sorted(set(labels) & forbidden),
        "official_target_overlap": len(set(labels) & official_targets),
        "official_targets_only": bool(spec.get("official_targets_only", False)),
    }


def run(args: argparse.Namespace) -> dict:
    specs = json.loads(args.sources.read_text())
    genes = read_gene_names(args.gene_names)
    train = read_condition_targets(args.train_targets)
    validation = read_condition_targets(args.validation_targets)
    test = read_condition_targets(args.test_targets)
    official_targets = train | validation | test
    if args.information_policy == "exclusion_safe":
        forbidden = validation | test
    elif args.information_policy == "vcc_official":
        forbidden = set()
    else:
        raise ValueError(f"unsupported information policy: {args.information_policy}")
    rows = [
        aggregate_source(spec, genes, forbidden, official_targets, args.chunk_rows)
        for spec in specs["sources"]
    ]
    contexts, targets, deltas, counts = [], [], [], []
    for row in rows:
        contexts.extend([row["context"]] * len(row["targets"]))
        targets.extend(row["targets"])
        deltas.append(row["delta"])
        counts.append(row["counts"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial.npz")
    np.savez_compressed(
        partial, contexts=np.asarray(contexts), targets=np.asarray(targets), genes=np.asarray(genes),
        delta=np.concatenate(deltas), counts=np.concatenate(counts),
        source_sha256s=np.asarray([row["source_sha256"] for row in rows]),
    )
    partial.replace(args.output)
    context_metadata = []
    for row in rows:
        metadata = {key: value for key, value in row.items() if key not in {"delta", "counts", "targets"}}
        metadata["targets_retained"] = len(row["targets"])
        context_metadata.append(metadata)
    excluded_validation = sorted(validation) if args.information_policy == "exclusion_safe" else []
    excluded_test = sorted(test) if args.information_policy == "exclusion_safe" else []
    manifest = {
        "dataset_id": specs["dataset_id"], "artifact_sha256": sha256(args.output),
        "contexts": context_metadata, "rows": len(targets), "h1_genes": len(genes),
        "information_policy": args.information_policy,
        "same_target_external_context_allowed": args.information_policy == "vcc_official",
        "excluded_validation_targets": excluded_validation, "excluded_test_targets": excluded_test,
        "official_target_identity_counts": {
            "train": len(train), "validation": len(validation), "test": len(test),
        },
        "h1_validation_expression_used": False,
        "h1_test_expression_used": False,
        "test_expression_used": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1/assets/official_2025")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gene-names", type=Path, default=root / "gene_names.csv")
    parser.add_argument("--train-targets", type=Path, default=root / "train/pert_counts_Training.csv")
    parser.add_argument("--validation-targets", type=Path, default=root / "validation/pert_counts_Validation.csv")
    parser.add_argument("--test-targets", type=Path, default=root / "test/pert_counts_Test.csv")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument(
        "--information-policy", choices=("exclusion_safe", "vcc_official"),
        default="exclusion_safe",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
