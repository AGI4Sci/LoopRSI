"""Run pinned scGenePT GO-All on prepared H1 train/validation data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


REQUIRED_MODULES = ("torch", "torch_geometric", "scanpy", "scgpt", "gears", "dcor", "h5py")
FORBIDDEN_PARTS = ("test_expression", "adata_test", "competition_test", "final_test")


def _obs_strings(obs, key, h5py):
    value = obs[key]
    if isinstance(value, h5py.Group):
        categories = value["categories"].asstr()[:]
        return [str(categories[code]) if code >= 0 else "" for code in value["codes"][:]]
    return [str(item) for item in value.asstr()[:]]


def preflight(args: argparse.Namespace) -> dict:
    inputs = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "source_repository", "processed_dataset", "split_json", "gene_names",
            "checkpoint", "vocab", "go_all_embedding",
        )
    }
    output = Path(args.output).resolve()
    if any(part in str(path).lower() for path in inputs.values() for part in FORBIDDEN_PARTS):
        raise ValueError("final-test expression path is forbidden")
    if any(
        output == path or path in output.parents or output in path.parents
        for path in inputs.values()
    ):
        raise ValueError("output must be outside all input assets")
    required = {
        "train_entry": inputs["source_repository"] / "train.py",
        "model_source": inputs["source_repository"] / "models" / "scGenePT.py",
        "dataset": inputs["processed_dataset"] / "perturb_processed.h5ad",
        "cell_graphs": inputs["processed_dataset"] / "data_pyg" / "cell_graphs.pkl",
        **{name: inputs[name] for name in (
            "split_json", "gene_names", "checkpoint", "vocab", "go_all_embedding",
        )},
    }
    missing = sorted(name for name, path in required.items() if not path.is_file())
    modules = sorted(name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None)
    split_path = inputs["split_json"]
    split = None
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if set(split) != {"train", "val"} or any(
            not isinstance(split[key], list) or not all(isinstance(gene, str) and gene for gene in split[key])
            for key in ("train", "val")
        ):
            raise ValueError("split must contain train and val gene-name lists only")
        if len(split["train"]) != 150 or len(split["val"]) != 50:
            raise ValueError("H1 split must contain 150 train and 50 validation targets")
        if len(set(split["train"] + split["val"])) != 200:
            raise ValueError("H1 train and validation targets overlap or repeat")
    genes_path = inputs["gene_names"]
    if genes_path.is_file():
        genes = [line.strip() for line in genes_path.read_text(encoding="utf-8").splitlines()]
        if len(genes) != 18080 or len(set(genes)) != 18080 or not all(genes):
            raise ValueError("H1 gene list must contain 18,080 unique names")
    missing_conditions = []
    split_metadata_errors = []
    if required["dataset"].is_file() and split is not None and "h5py" not in modules:
        import h5py

        with h5py.File(required["dataset"], "r") as h5:
            obs = h5["obs"]
            if "condition" not in obs or "split" not in obs:
                split_metadata_errors.append("obs must contain condition and split")
                observed = set(_obs_strings(obs, "condition", h5py)) if "condition" in obs else set()
            else:
                conditions = _obs_strings(obs, "condition", h5py)
                splits = _obs_strings(obs, "split", h5py)
                observed = set(conditions)
                assignments = dict(zip(
                    (gene + "+ctrl" for gene in split["train"] + split["val"]),
                    (["train"] * 150 + ["val"] * 50),
                ))
                if any(
                    label != assignments.get(condition)
                    for condition, label in zip(conditions, splits) if condition != "ctrl"
                ):
                    split_metadata_errors.append("cell-level split disagrees with fixed H1 targets")
                if any(condition == "ctrl" and label != "train" for condition, label in zip(conditions, splits)):
                    split_metadata_errors.append("validation controls would enter the scGenePT training graph cache")
        expected = {"ctrl"} | {
            gene + "+ctrl" for gene in split["train"] + split["val"]
        }
        missing_conditions = sorted(expected - observed)
        if observed - expected:
            split_metadata_errors.append("dataset contains conditions outside fixed H1 split")
    return {
        "schema_version": "vcc25.scgenept-preflight/v1",
        "status": "ready" if not missing and not modules and not missing_conditions and not split_metadata_errors else "blocked",
        "missing_assets": missing,
        "missing_modules": modules,
        "missing_conditions": missing_conditions,
        "split_metadata_errors": split_metadata_errors,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "test_expression_read": False,
        "gpu_used": False,
    }


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("LOOPRSI_REMOTE_GPU_JOB") != "1" or "_pool" not in os.environ.get(
        "LOOPRSI_CHARGED_GROUP", ""
    ):
        raise ValueError("scGenePT training requires a charged _pool GPU job")
    import numpy as np
    import torch

    if not torch.cuda.is_available() or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("scGenePT training requires one assigned CUDA device")
    sys.path.insert(0, str(Path(args.source_repository).resolve()))
    import train as upstream
    from gears import PertData
    from utils.data_loading import create_embs_w, match_genes_to_scgpt_vocab
    from utils.scgpt_config import SPECIAL_TOKENS
    from utils.evaluation import eval_perturb
    from gears.inference import compute_metrics

    dataset = Path(args.processed_dataset).resolve()
    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    expected_genes = Path(args.gene_names).read_text(encoding="utf-8").splitlines()
    pert_data = PertData(str(Path(args.output).resolve()))
    pert_data.load(data_path=str(dataset))
    if list(pert_data.adata.var["gene_name"]) != expected_genes:
        raise ValueError("prepared H1 gene order differs from the fixed gene list")
    condition = lambda gene: gene + "+ctrl"
    observed = set(pert_data.adata.obs["condition"].astype(str))
    required = {"ctrl"} | {condition(gene) for gene in split["train"] + split["val"]}
    if observed != required:
        raise ValueError("prepared H1 dataset conditions differ from fixed train/validation split")
    assignments = {
        **{condition(gene): "train" for gene in split["train"]},
        **{condition(gene): "val" for gene in split["val"]},
        "ctrl": "train",
    }
    if "split" not in pert_data.adata.obs or any(
        label != assignments.get(perturbation)
        for perturbation, label in zip(
            pert_data.adata.obs["condition"].astype(str),
            pert_data.adata.obs["split"].astype(str),
        )
    ):
        raise ValueError("prepared H1 cells do not follow the fixed train/validation split")
    pert_data.split = "no_test"
    pert_data.set2conditions = {
        "train": ["ctrl", *(condition(gene) for gene in split["train"])],
        "val": [condition(gene) for gene in split["val"]],
    }
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.batch_size)
    upstream.set_seed(args.seed)
    vocab, gene_ids, dataset_genes, gene2idx = match_genes_to_scgpt_vocab(
        args.vocab, pert_data, upstream.scg.logger, SPECIAL_TOKENS
    )
    go_matrix, matched_go_genes = create_embs_w(
        dataset_genes, vocab, args.go_all_embedding, 1536
    )
    model_type = "scgenept_go_all_gpt_concat"
    model = upstream.scGenePT(
        ntoken=len(vocab), d_model=upstream.EMBSIZE, nhead=upstream.NHEAD,
        d_hid=upstream.D_HID, nlayers=upstream.NLAYERS,
        nlayers_cls=upstream.N_LAYERS_CLS, n_cls=upstream.N_CLS,
        vocab=vocab, n_perturbagens=2, dropout=0.2,
        pad_token=upstream.PAD_TOKEN, pad_value=upstream.PAD_VALUE,
        pert_pad_id=upstream.PERT_PAD_ID, use_fast_transformer=True,
        embs_to_include=upstream.get_embs_to_include(model_type),
        genept_embs=[], genept_emb_type=None, genept_emb_size=None,
        go_embs_to_include={"all": go_matrix}, go_emb_type="all", go_emb_size=1536,
    )
    device = "cuda:0"
    model = upstream.load_pretrained_model(
        model, ["encoder", "value_encoder", "transformer_encoder"],
        False, Path(args.checkpoint), device,
    ).to(device)
    output = Path(args.output).resolve()
    (output / "metrics" / "val").mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler()
    best = upstream.train_model(
        model, pert_data, args.epochs, upstream.masked_mse_loss, optimizer,
        scheduler, scaler, device, gene_ids, upstream.scg.logger,
        upstream.INCLUDE_ZERO_GENE, True, "H1", model_type, args.seed,
        1536, 100, args.epochs, gene2idx, False, output,
    )
    if best is None:
        raise RuntimeError("scGenePT did not produce a best validation model")
    torch.save(best.state_dict(), output / "model.pt")
    validation = eval_perturb(
        pert_data.dataloader["val_loader"], best, device,
        upstream.INCLUDE_ZERO_GENE, gene_ids,
    )
    metrics, _ = compute_metrics(validation)
    np.savez_compressed(output / "validation_predictions.npz", **validation)
    result = {
        "schema_version": "vcc25.scgenept-run/v1",
        "status": "partial",
        "train_targets": 150,
        "validation_targets": 50,
        "genes": 18080,
        "go_matched_genes": len(matched_go_genes),
        "epochs": args.epochs,
        "validation_metrics": {key: float(value) for key, value in metrics.items()},
        "test_expression_read": False,
        "official_score_claim": False,
    }
    (output / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "source-repository", "processed-dataset", "split-json", "gene-names",
        "checkpoint", "vocab", "go-all-embedding", "output",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    result = preflight(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.preflight_only or result["status"] != "ready":
        print(json.dumps(result, sort_keys=True))
        return 0 if args.preflight_only else 2
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
