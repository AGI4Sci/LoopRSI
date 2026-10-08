"""Run one bounded PRESAGE train batch and one train-derived validation batch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence


OFFICIAL_ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/"
    "naturebench/lingshu_cell_h1/assets/official_2025"
)
PACKAGE_ROOT = Path(__file__).resolve().parent
PRESAGE_REVISION = "2c7b231c60cf110c4aab0acf8ecd2c2b3268fffb"
CONDITIONS = ("ACAT2", "ACVR1B", "AKT2", "ARID1A")
INSTALL_REQUIREMENTS = (
    "anndata",
    "array-api-compat",
    "donfig",
    "google-crc32c",
    "h5py",
    "lightning-utilities",
    "natsort",
    "numcodecs",
    "packaging",
    "python-dateutil",
    "pytz",
    "scverse-misc",
    "scanpy",
    "session-info2",
    "six",
    "torchmetrics",
    "typing-extensions",
    "tzdata",
    "zarr",
    "pytorch-lightning",
    "torch-geometric",
)


def validate_source_path(root: Path, train: Path) -> None:
    expected = (root / "train" / "adata_Training.h5ad").resolve()
    if train.resolve() != expected:
        raise ValueError("PRESAGE smoke may read official training expression only")


def build_scope(
    *, train_cells: int, validation_cells: int, genes: int,
    conditions: Sequence[str],
) -> Dict[str, object]:
    return {
        "train_conditions": list(conditions),
        "validation_conditions": list(conditions),
        "train_cells": train_cells,
        "validation_cells": validation_cells,
        "genes": genes,
        "train_batches": 1,
        "validation_batches": 1,
        "validation_source": "deterministic_internal_split_of_official_training",
        "official_validation_expression_read": False,
        "final_test_expression_read": False,
    }


def _resolve_assets(root: Path) -> tuple[Path, Path]:
    search_roots = [root, root.parent, Path("/workdir/rjob")]
    wheelhouses = [candidate for base in search_roots for candidate in (
        base / "presage-wheelhouse", base / "wheelhouse") if candidate.is_dir()]
    archives = [archive for base in search_roots for archive in base.rglob(
        f"PRESAGE-{PRESAGE_REVISION}.tar.gz") if archive.is_file()]
    wheelhouse = next((candidate for candidate in wheelhouses if any(candidate.glob("*.whl"))), None)
    if wheelhouse is None or len(set(archives)) != 1:
        raise FileNotFoundError("packaged PRESAGE source archive or wheelhouse is missing")
    return wheelhouse, archives[0]


def _install_dependencies(wheelhouse: Path, target: Path) -> None:
    command = [
        sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
        str(wheelhouse), "--target", str(target), "--no-deps", *INSTALL_REQUIREMENTS,
    ]
    subprocess.run(command, check=True)


def _dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


def _take_cells(adata, conditions: Sequence[str], cells_per_condition: int):
    names = []
    targets = adata.obs["target_gene"].astype(str)
    for condition in conditions:
        matches = list(adata.obs_names[targets == condition][:cells_per_condition])
        if len(matches) != cells_per_condition:
            raise RuntimeError(f"insufficient training cells for {condition}: {len(matches)}")
        names.extend(matches)
    return adata[names].to_memory()


def _batch(adata, names, gene_to_index, reference, torch, device):
    expression = _dense(adata[names].X).astype("float32", copy=False)
    targets = adata[names].obs["target_gene"].astype(str)
    indicators = torch.zeros((len(names), adata.n_vars), dtype=torch.float32)
    for row, target in enumerate(targets):
        indicators[row, gene_to_index[target]] = 1.0
    return indicators.to(device), torch.as_tensor(expression - reference).to(device)


def main() -> None:
    if os.environ.get("LOOPRSI_REMOTE_GPU_JOB") != "1" or "_pool" not in os.environ.get(
        "LOOPRSI_CHARGED_GROUP", ""
    ):
        raise RuntimeError("PRESAGE smoke requires a charged public GPU pool job")

    train_path = OFFICIAL_ROOT / "train" / "adata_Training.h5ad"
    validate_source_path(OFFICIAL_ROOT, train_path)
    wheelhouse, source_archive = _resolve_assets(PACKAGE_ROOT)

    with tempfile.TemporaryDirectory(prefix="presage-h1-smoke-") as tmp:
        root = Path(tmp)
        deps, sources = root / "deps", root / "sources"
        deps.mkdir(); sources.mkdir()
        _install_dependencies(wheelhouse, deps)
        with tarfile.open(source_archive) as archive:
            archive.extractall(sources, filter="data")
        source_roots = list(sources.glob(f"PRESAGE-{PRESAGE_REVISION}/src"))
        if len(source_roots) != 1:
            raise RuntimeError("expected exactly one pinned PRESAGE source tree")
        sys.path[:0] = [str(deps), str(source_roots[0])]

        import anndata as ad
        import numpy as np
        import torch
        import presage

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the pool job")
        torch.manual_seed(42); np.random.seed(42)
        device = torch.device("cuda")

        backed = ad.read_h5ad(train_path, backed="r")
        selected = _take_cells(backed, CONDITIONS, 16)
        backed.file.close()
        train_names, validation_names = [], []
        target_values = selected.obs["target_gene"].astype(str)
        for condition in CONDITIONS:
            names = list(selected.obs_names[target_values == condition])
            train_names.extend(names[:12]); validation_names.extend(names[12:16])

        genes = selected.var_names.astype(str).to_numpy()
        if len(genes) != 18080 or len(set(genes)) != len(genes):
            raise RuntimeError("expected 18,080 unique H1 genes")
        gene_to_index = {gene: index for index, gene in enumerate(genes)}
        missing = sorted(set(CONDITIONS) - set(gene_to_index))
        if missing:
            raise RuntimeError(f"PRESAGE targets absent from H1 gene space: {missing}")

        train_expression = _dense(selected[train_names].X).astype("float32", copy=False)
        reference = train_expression.mean(axis=0, keepdims=True)
        train_inds, train_expr = _batch(
            selected, train_names, gene_to_index, reference, torch, device
        )
        val_inds, val_expr = _batch(
            selected, validation_names, gene_to_index, reference, torch, device
        )

        train_adata = SimpleNamespace(var=SimpleNamespace(index=selected.var_names))
        train_dataset = SimpleNamespace(
            adata=train_adata,
            X=train_expression - reference,
            indmtx=train_inds.cpu().numpy(),
        )
        datamodule = SimpleNamespace(
            batch_size=len(train_names), train_dataset=train_dataset, X_train_pca=None
        )
        embeddings = np.random.default_rng(42).normal(
            0.0, 0.02, size=(len(genes), 4, 2)
        ).astype("float32")
        presage.PrepareInputs._prep_inputs = lambda _self: embeddings
        # Upstream passes a mask to every pool although its mean pool accepts
        # one argument. Keep its mean-pooling calculation and bridge that API bug.
        presage.Pool.forward = lambda self, tensor, _locs, _mask: self.pool(tensor)
        config = {
            "added_singles_loss_scale": 0.0,
            "n_nmf_embedding": 4,
            "learnable_gene_embedding": False,
            "item_hidden_size": 16,
            "pca_dim": None,
            "softmax_temperature": 1.0,
            "pathway_item_hidden_size": 16,
            "pathway_item_nlayers": 1,
            "batch_norm": False,
            "pathway_weight_type": "mean",
            "item_nlayers": 0,
        }
        model = presage.PRESAGE(config, datamodule, len(genes), len(genes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train(); optimizer.zero_grad(set_to_none=True)
        train_prediction, train_embedding, _ = model(train_inds, None)
        train_loss = model.compute_loss(train_prediction, train_expr, train_embedding)
        train_loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction, validation_embedding, _ = model(val_inds, None)
            validation_loss = model.compute_loss(
                validation_prediction, val_expr, validation_embedding
            )

        if train_prediction.shape != train_expr.shape or validation_prediction.shape != val_expr.shape:
            raise RuntimeError("PRESAGE prediction shape violates the H1 smoke contract")
        if not torch.isfinite(train_prediction).all() or not torch.isfinite(validation_prediction).all():
            raise RuntimeError("PRESAGE produced non-finite predictions")
        result = {
            "schema_version": "vcc25.presage-h1-smoke/v1",
            "source_revision": PRESAGE_REVISION,
            "scope": build_scope(
                train_cells=len(train_names), validation_cells=len(validation_names),
                genes=len(genes), conditions=CONDITIONS,
            ),
            "execution": {
                "train_loss": float(train_loss.detach().cpu()),
                "validation_loss": float(validation_loss.detach().cpu()),
                "prediction_shape": list(validation_prediction.shape),
                "predictions_finite": True,
                "gpu": torch.cuda.get_device_name(0),
            },
            "preprocessing": "train-only global-expression reference; deterministic synthetic gene embeddings",
            "official_score_claim": False,
        }
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
