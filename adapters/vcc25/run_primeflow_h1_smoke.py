"""Run a bounded PRiMeFlow H1 train/validation smoke without test data."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import List

try:
    from adapters.vcc25.probe_primeflow_offline_install import (
        PACKAGE_ROOT, activate_paths, build_install_command,
        build_resolved_install_command, resolve_assets,
    )
except ModuleNotFoundError:
    from probe_primeflow_offline_install import (
        PACKAGE_ROOT, activate_paths, build_install_command,
        build_resolved_install_command, resolve_assets,
    )


OFFICIAL_ROOT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/"
    "naturebench/lingshu_cell_h1/assets/official_2025"
)
TRAIN_CONDITIONS = ("ACAT2", "ACVR1B", "AKT2", "ARID1A")


def validate_source_paths(root: Path, train: Path, validation: Path) -> None:
    expected = {
        (root / "train" / "adata_Training.h5ad").resolve(),
        (root / "validation" / "adata_Validation.h5ad").resolve(),
    }
    actual = {train.resolve(), validation.resolve()}
    forbidden = ("final_test", "adata_test", "/test/")
    if actual != expected or any(token in str(path).lower() for path in actual for token in forbidden):
        raise ValueError("final-test and non-official inputs are forbidden")


def build_train_command(data: Path, split: Path, genes: Path, output: Path) -> List[str]:
    return [
        "python3", "-m", "primeflow.modelcore.train",
        "experiment=primeflow/vcc/flow_matching_gaussian_source_annbacked_unet",
        f"data.data.filename={data}",
        f"data.splitter.split_path={split}",
        f"data.data_iter_factory.feature_filter_path={genes}",
        "model.dynamics_model.gene_embedding_parquet_filepath=null",
        # The upstream linear-warmup scheduler imports an old pl_bolts API that
        # is incompatible with the pinned Lightning runtime.  Use the upstream
        # PerturBench scheduler backed directly by torch for this bounded smoke.
        "model/lr_scheduler=cosine_annealing_warm_restarts",
        "~model.lr_scheduler.conf.warmup_epochs",
        "~model.lr_scheduler.conf.warmup_start_lr",
        "~model.lr_scheduler.conf.eta_min",
        "trainer.devices=1", "trainer.num_nodes=1", "trainer.strategy=auto",
        "trainer.min_epochs=1", "trainer.max_epochs=1",
        "+trainer.limit_train_batches=1", "+trainer.limit_val_batches=1",
        "trainer.precision=32-true", "test=False", "+finetune=True",
        "data.loader.batch_size=8", "data.loader.num_workers=0",
        "data.loader.persistent_workers=False", "data.loader.ddp_mode=False",
        f"hydra.run.dir={output}",
    ]


def _take_conditions(adata, conditions, cells_per_condition):
    indices = []
    values = adata.obs["target_gene"].astype(str)
    for condition in conditions:
        matches = list(adata.obs_names[values == condition][:cells_per_condition])
        if len(matches) != cells_per_condition:
            raise RuntimeError(f"insufficient cells for {condition}: {len(matches)}")
        indices.extend(matches)
    return adata[indices].to_memory()


def _add_primeflow_metadata(adata, prefix: str) -> None:
    adata.obs_names = [f"{prefix}:{name}" for name in adata.obs_names]
    adata.obs["condition"] = adata.obs["target_gene"].astype(str)
    adata.obs["cell_type"] = "K562"
    adata.obs["treatment"] = "CRISPRi"
    adata.obs["dataset"] = "VCC25-H1"
    adata.obs["scrna_method"] = "10x"
    adata.obs["vector"] = "CRISPRi"


def main() -> None:
    train_path = OFFICIAL_ROOT / "train" / "adata_Training.h5ad"
    validation_path = OFFICIAL_ROOT / "validation" / "adata_Validation.h5ad"
    validate_source_paths(OFFICIAL_ROOT, train_path, validation_path)
    wheelhouse, archives = resolve_assets(PACKAGE_ROOT)

    with tempfile.TemporaryDirectory(prefix="primeflow-h1-smoke-") as tmp:
        root = Path(tmp)
        deps, sources, output = root / "deps", root / "sources", root / "output"
        deps.mkdir(); sources.mkdir(); output.mkdir()
        subprocess.run(build_install_command(wheelhouse, deps), check=True)
        subprocess.run(build_resolved_install_command(wheelhouse, deps), check=True)
        source_paths = []
        for name, archive in archives.items():
            with tarfile.open(archive) as handle:
                handle.extractall(sources, filter="data")
            candidates = sorted(sources.glob(f"{name}-*/src"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one extracted {name} source tree")
            source_paths.append(candidates[0])
        activate_paths(deps, source_paths, __import__("sys").path)

        import anndata as ad
        import pandas as pd

        train_backed = ad.read_h5ad(train_path, backed="r")
        selected = _take_conditions(train_backed, TRAIN_CONDITIONS, 16)
        train_backed.file.close()
        train_indices, validation_indices = [], []
        values = selected.obs["target_gene"].astype(str)
        for condition in TRAIN_CONDITIONS:
            names = list(selected.obs_names[values == condition])
            train_indices.extend(names[:12])
            validation_indices.extend(names[12:16])
        train = selected[train_indices].copy()
        validation = selected[validation_indices].copy()
        _add_primeflow_metadata(train, "train")
        _add_primeflow_metadata(validation, "validation")
        combined = ad.concat([train, validation], join="inner", merge="same")
        data_path = root / "bounded_train_validation.h5ad"
        combined.uns["filename"] = str(data_path)
        combined.write_h5ad(data_path)
        split = pd.Series(
            ["train"] * train.n_obs + ["val"] * validation.n_obs,
            index=combined.obs_names,
        )
        split_path = root / "split.csv"
        split.to_csv(split_path, header=False)
        genes_path = root / "genes.csv"
        pd.Series(combined.var_names).to_csv(genes_path, index=False, header=False)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([*(str(p) for p in source_paths), str(deps)])
        env["HYDRA_FULL_ERROR"] = "1"
        completed = subprocess.run(
            build_train_command(data_path, split_path, genes_path, output),
            check=False, env=env,
        )
        print(json.dumps({
            "schema_version": "vcc25.primeflow-h1-smoke/v1",
            "returncode": completed.returncode,
            "train_cells": train.n_obs,
            "validation_cells": validation.n_obs,
            "genes": combined.n_vars,
            "train_conditions": list(TRAIN_CONDITIONS),
            "validation_conditions": list(TRAIN_CONDITIONS),
            "validation_source": "deterministic_internal_split_of_official_training",
            "official_validation_expression_read": False,
            "final_test_expression_read": False,
        }, sort_keys=True), flush=True)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
