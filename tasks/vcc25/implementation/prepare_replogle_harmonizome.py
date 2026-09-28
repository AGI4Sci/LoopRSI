from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from official_h1_contract import read_condition_targets, read_gene_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perturbation_target(attribute: str) -> str:
    parts = attribute.split("_")
    return parts[1] if len(parts) >= 3 else ""


def run(args: argparse.Namespace) -> dict:
    actual_hash = sha256(args.matrix)
    if actual_hash != args.matrix_sha256:
        raise ValueError(f"matrix hash mismatch: {actual_hash}")
    genes = read_gene_names(args.gene_names)
    train = read_condition_targets(args.train_targets)
    validation = read_condition_targets(args.validation_targets)
    test = read_condition_targets(args.test_targets)
    official = train | validation | test

    with gzip.open(args.matrix, "rt", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        selected = defaultdict(list)
        for index, attribute in enumerate(header[1:]):
            target = perturbation_target(attribute)
            if target in official:
                selected[target].append(index)
        source_genes, source_rows = [], []
        selected_columns = sorted({index for indices in selected.values() for index in indices})
        compact_index = {source: dest for dest, source in enumerate(selected_columns)}
        for row in reader:
            source_genes.append(row[0])
            source_rows.append([float(row[index + 1]) for index in selected_columns])

    source = np.asarray(source_rows, dtype=np.float32)
    source_gene_index = {gene: index for index, gene in enumerate(source_genes)}
    target_rows, deltas, counts = [], [], []
    for target in sorted(selected):
        columns = [compact_index[index] for index in selected[target]]
        signature = source[:, columns].mean(axis=1)
        mapped = np.zeros(len(genes), dtype=np.float32)
        for index, gene in enumerate(genes):
            source_index = source_gene_index.get(gene)
            if source_index is not None:
                mapped[index] = signature[source_index]
        target_rows.append(target)
        deltas.append(mapped)
        counts.append(len(columns))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial.npz")
    np.savez_compressed(
        partial,
        contexts=np.asarray([args.context] * len(target_rows)),
        targets=np.asarray(target_rows),
        genes=np.asarray(genes),
        delta=np.asarray(deltas, dtype=np.float32),
        counts=np.asarray(counts, dtype=np.int32),
        source_sha256s=np.asarray([actual_hash]),
    )
    partial.replace(args.output)
    coverage = {
        "train": len(train & set(target_rows)),
        "validation": len(validation & set(target_rows)),
        "test": len(test & set(target_rows)),
    }
    manifest = {
        "dataset_id": "harmonizome-replogle-k562-genomewide-signatures-v1",
        "source": args.source_url,
        "source_sha256": actual_hash,
        "artifact_sha256": sha256(args.output),
        "context": args.context,
        "source_genes": len(source_genes),
        "h1_genes": len(genes),
        "mapped_h1_genes": len(set(genes) & set(source_genes)),
        "official_target_coverage": coverage,
        "targets_retained": len(target_rows),
        "information_policy": "vcc_official_external_only",
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
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context", default="K562_gwps_harmonizome")
    parser.add_argument("--source-url", default="https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.,+Cell,+2022+K562+Genome-wide+Perturb-seq+Gene+Perturbation+Signatures")
    parser.add_argument("--gene-names", type=Path, default=root / "gene_names.csv")
    parser.add_argument("--train-targets", type=Path, default=root / "train/pert_counts_Training.csv")
    parser.add_argument("--validation-targets", type=Path, default=root / "validation/pert_counts_Validation.csv")
    parser.add_argument("--test-targets", type=Path, default=root / "test/pert_counts_Test.csv")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
