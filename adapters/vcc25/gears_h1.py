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


REQUIRED_MODULES = ("numpy", "torch", "torch_geometric", "scanpy", "scipy", "sklearn", "dcor")
FORBIDDEN_NAME_PARTS = ("test_expression", "adata_test", "competition_test")


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
        "status": "ready" if not missing and not modules else "blocked",
        "missing_assets": missing,
        "missing_modules": modules,
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
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise ValueError("GEARS training requires one assigned CUDA device")

    output.mkdir(parents=True, exist_ok=True)
    data_root = output / "gears_data"
    data_root.mkdir(exist_ok=True)
    shutil.copyfile(args.gene2go, data_root / "gene2go_all.pkl")
    sys.path.insert(0, str(source))
    from gears import GEARS, PertData

    with (output / "split.pkl").open("wb") as stream:
        pickle.dump({**gears_split, "test": []}, stream)
    pert_data = PertData(str(data_root), default_pert_graph=False)
    pert_data.load(data_path=str(dataset))
    pert_data.prepare_split(split="custom", split_dict_path=str(output / "split.pkl"))
    pert_data.split = "no_test"
    pert_data.get_dataloader(batch_size=32)
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
        "command": sys.argv, "source_repository": str(source),
        "train_conditions": len(split["train"]), "validation_conditions": len(split["val"]),
        "gene_count": len(expected_genes), "epochs": 1,
        "prediction_keys": sorted(predictions), "test_expression_read": False,
        "gpu_used": True, "official_score_claim": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
