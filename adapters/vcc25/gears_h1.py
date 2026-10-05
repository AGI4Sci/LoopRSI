"""Train pinned GEARS on a prepared H1 train/validation dataset only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import sys
from pathlib import Path


REQUIRED_MODULES = ("numpy", "torch", "torch_geometric", "scanpy", "scipy", "sklearn", "dcor", "h5py")
FORBIDDEN_NAME_PARTS = ("test_expression", "adata_test", "competition_test")


def _has_validation_controls(conditions, splits) -> bool:
    return any(condition == "ctrl" and label != "train" for condition, label in zip(conditions, splits))


def _has_one_cuda_device(torch) -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() == 1


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight(args: argparse.Namespace) -> dict:
    paths = {
        "source_repository": Path(args.source_repository).resolve(),
        "processed_dataset": Path(args.processed_dataset).resolve(),
        "split_json": Path(args.split_json).resolve(),
        "gene2go": Path(args.gene2go).resolve(),
        "gene_names": Path(args.gene_names).resolve(),
    }
    output = Path(args.output).resolve()
    if any(part in str(path).lower() for path in paths.values() for part in FORBIDDEN_NAME_PARTS):
        raise ValueError("final-test expression path is forbidden in GEARS reproduction")
    source = paths["source_repository"]
    dataset = paths["processed_dataset"]
    required = {
        "gears_source": source / "gears" / "gears.py",
        "dataset": dataset / "perturb_processed.h5ad",
        "cell_graphs": dataset / "data_pyg" / "cell_graphs.pkl",
        "split_json": paths["split_json"],
        "gene2go": paths["gene2go"],
        "gene_names": paths["gene_names"],
    }
    missing = sorted(name for name, path in required.items() if not path.is_file())
    modules = sorted(name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None)
    data_errors = []
    if required["dataset"].is_file() and importlib.util.find_spec("h5py") is not None:
        import h5py
        from adapters.vcc25.scgenept_h1 import _obs_strings

        with h5py.File(required["dataset"], "r") as h5:
            obs = h5["obs"]
            if "condition" not in obs or "split" not in obs:
                data_errors.append("dataset needs cell-level condition and split metadata")
            else:
                conditions = _obs_strings(obs, "condition", h5py)
                splits = _obs_strings(obs, "split", h5py)
                if _has_validation_controls(conditions, splits):
                    data_errors.append("validation controls would enter the GEARS training graph cache")
    split = None
    if not missing and not modules:
        split = json.loads(paths["split_json"].read_text(encoding="utf-8"))
    elif paths["split_json"].is_file():
        split = json.loads(paths["split_json"].read_text(encoding="utf-8"))
    if split is not None:
        if set(split) != {"train", "val"} or any(
            not isinstance(split[key], list) or not split[key]
            or not all(isinstance(item, str) and item for item in split[key])
            for key in ("train", "val")
        ):
            raise ValueError("GEARS split must contain nonempty train and val condition lists only")
        if set(split["train"]) & set(split["val"]):
            raise ValueError("GEARS train and validation conditions overlap")
        if len(split["train"]) != len(set(split["train"])) or len(split["val"]) != len(set(split["val"])):
            raise ValueError("GEARS split contains duplicate conditions")
    if output == dataset or dataset in output.parents or output == source or source in output.parents:
        raise ValueError("GEARS output must be outside source and prepared dataset")
    return {
        "schema_version": "vcc25.gears-preflight/v1",
        "status": "ready" if not missing and not modules and not data_errors else "blocked",
        "missing_assets": missing,
        "missing_modules": modules,
        "data_errors": data_errors,
        "inputs": {name: str(path) for name, path in paths.items()},
        "data_scope": "prepared train and validation only; final-test expression excluded",
        "test_expression_read": False,
        "gpu_used": False,
    }


def _prediction_perturbation(condition: str) -> list[str]:
    if condition == "ctrl":
        return []
    genes = [value for value in condition.split("+") if value != "ctrl"]
    if not genes or len(genes) > 2 or any(not gene for gene in genes):
        raise ValueError(f"unsupported GEARS perturbation condition: {condition!r}")
    return genes


def _gears_condition(condition: str) -> str:
    """Map an H1 single-gene label to the pinned GEARS condition format."""
    genes = _prediction_perturbation(condition)
    return "ctrl" if not genes else "+".join([*genes, "ctrl"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--processed-dataset", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--gene2go", required=True)
    parser.add_argument("--gene-names", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-dir")
    parser.add_argument("--compute-smoke-de", action="store_true")
    parser.add_argument("--official-go-csv")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    result = preflight(args)
    _write(output / "preflight.json", result)
    if result["status"] != "ready" or args.preflight_only:
        print(json.dumps(result, sort_keys=True))
        return 0 if args.preflight_only else 2

    # GEARS reads its GO cache from data_path and uses its own pickle split format.
    # Both files are created from explicitly supplied, reviewed inputs.
    import numpy as np
    import scanpy as sc
    import torch

    source = Path(args.source_repository).resolve()
    dataset = Path(args.processed_dataset).resolve()
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    expected_genes = [line.strip() for line in Path(args.gene_names).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(expected_genes) != 18080 or len(set(expected_genes)) != len(expected_genes):
        raise ValueError("expected H1 gene list must contain 18,080 unique names")
    adata = sc.read_h5ad(dataset / "perturb_processed.h5ad")
    if not {"condition", "cell_type", "condition_name"}.issubset(adata.obs.columns):
        raise ValueError("prepared GEARS dataset lacks required obs fields")
    if "gene_name" not in adata.var.columns or list(adata.var["gene_name"]) != expected_genes:
        raise ValueError("prepared GEARS gene order differs from expected H1 gene list")
    observed_conditions = set(adata.obs["condition"])
    gears_split = {key: [_gears_condition(value) for value in split[key]] for key in ("train", "val")}
    allowed_conditions = set(gears_split["train"] + gears_split["val"] + ["ctrl"])
    if not set(gears_split["train"] + gears_split["val"]).issubset(observed_conditions):
        raise ValueError("GEARS split refers to absent conditions")
    if observed_conditions - allowed_conditions:
        raise ValueError("prepared GEARS dataset contains conditions outside train and validation")
    if "ctrl" not in observed_conditions:
        raise ValueError("prepared GEARS dataset has no control condition")
    if (os.environ.get("LOOPRSI_REMOTE_GPU_JOB") != "1"
            or "_pool" not in os.environ.get("LOOPRSI_CHARGED_GROUP", "")):
        raise ValueError("GEARS training requires a remote charged _pool GPU job")
    if not _has_one_cuda_device(torch):
        raise ValueError("GEARS training requires one assigned CUDA device")

    output.mkdir(parents=True, exist_ok=True)
    data_root = output / "gears_data"
    data_root.mkdir(exist_ok=True)
    shutil.copyfile(args.gene2go, data_root / "gene2go_all.pkl")
    if args.official_go_csv:
        import pandas as pd

        go_dir = data_root / "go_essential_all"
        go_dir.mkdir(exist_ok=True)
        shutil.copyfile(args.official_go_csv, go_dir / "go_essential_all.csv")
        go_edges = pd.read_csv(args.official_go_csv, usecols=["source", "target"])
        go_genes = sorted(set(go_edges["source"]) | set(go_edges["target"]))
        required_perts = set(split["train"] + split["val"])
        if not required_perts.issubset(go_genes):
            raise ValueError("train or validation perturbation is absent from official GO graph")
        with (data_root / "essential_all_data_pert_genes.pkl").open("wb") as stream:
            pickle.dump(go_genes, stream)
    sys.path.insert(0, str(source))
    from gears import GEARS, PertData

    if args.compute_smoke_de:
        if adata.n_obs > 500 or not args.shard_dir:
            raise ValueError("smoke DE preparation requires at most 500 cells and shards")
        from gears.data_utils import get_DE_genes, get_dropout_non_zero_genes

        prepared = output / "prepared_dataset"
        (prepared / "data_pyg").mkdir(parents=True, exist_ok=True)
        adata = get_dropout_non_zero_genes(get_DE_genes(adata, skip_calc_de=False))
        adata.write_h5ad(prepared / "perturb_processed.h5ad")
        shutil.copyfile(dataset / "data_pyg" / "cell_graphs.pkl", prepared / "data_pyg" / "cell_graphs.pkl")
        dataset = prepared

    with (output / "split.pkl").open("wb") as stream:
        pickle.dump({**gears_split, "test": []}, stream)
    pert_data = PertData(str(data_root), default_pert_graph=bool(args.official_go_csv))
    pert_data.load(data_path=str(dataset))
    (data_root / pert_data.dataset_name).mkdir(parents=True, exist_ok=True)
    pert_data.prepare_split(split="custom", split_dict_path=str(output / "split.pkl"))
    pert_data.split = "no_test"
    pert_data.get_dataloader(batch_size=32)
    if args.shard_dir:
        from adapters.vcc25.gears_shard_loader import ShardBatchLoader
        pert_data.dataloader = {
            "train_loader": ShardBatchLoader(args.shard_dir, "train", 32),
            "val_loader": ShardBatchLoader(args.shard_dir, "val", 32),
        }
    model = GEARS(pert_data, device="cuda:0")
    model.model_initialize(hidden_size=64)
    model.train(epochs=1)
    model.save_model(str(output / "model"))
    predictions = model.predict([
        _prediction_perturbation(condition)
        for condition in split["val"] if condition != "ctrl"
    ])
    np.savez_compressed(output / "validation_predictions.npz", **predictions)
    _write(output / "run.json", {
        "schema_version": "vcc25.gears-run/v1", "status": "partial",
        "command": ["gears_h1", *(argv if argv is not None else sys.argv[1:])],
        "source_repository": str(source),
        "train_conditions": len(split["train"]), "validation_conditions": len(split["val"]),
        "gene_count": len(expected_genes), "epochs": 1,
        "prediction_keys": sorted(predictions), "test_expression_read": False,
        "gpu_used": True, "official_score_claim": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
