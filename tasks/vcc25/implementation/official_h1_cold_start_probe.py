from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from crpm.cold_start import TargetFeatureCRPM
from crpm.target_features import (
    go_bp_hash_features,
    hashed_gene_symbol_features,
    projected_gene_embedding_features,
)
from official_h1_contract import read_condition_targets, read_gene_names


DEFAULT_ROOT = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1")
DEFAULT_ASSETS = DEFAULT_ROOT / "assets/official_2025"
CONTROL = "non-targeting"
PROTOCOL_ID = "vcc-h1-cold-start-target-feature-probe-v1"


@dataclass
class Sample:
    x: np.ndarray
    target_names_per_row: list[str]
    batch_names_per_row: list[str]
    guide_names_per_row: list[str]
    obs_names: list[str]
    gene_names: list[str]


def _decode(values: Any) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def _h5ad_index(group: Any) -> list[str]:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return _decode(group[str(key)][...])


def _h5ad_column(group: Any, key: str) -> list[str]:
    node = group[key]
    if hasattr(node, "keys") and "categories" in node and "codes" in node:
        categories = _decode(node["categories"][...])
        return [categories[int(code)] for code in node["codes"][...] if int(code) >= 0]
    return _decode(node[...])



def read_symbol_to_gene_id(h5ad_path: Path, official_genes: list[str]) -> dict[str, str]:
    import h5py

    with h5py.File(h5ad_path, "r") as handle:
        var_names = _h5ad_index(handle["var"])
        if var_names != official_genes:
            raise ValueError(f"{h5ad_path} var_names do not match official gene order")
        gene_ids = _h5ad_column(handle["var"], "gene_id")
    return dict(zip(var_names, gene_ids))


def _read_csr_rows(handle: Any, rows: np.ndarray) -> np.ndarray:
    x = handle["X"]
    shape = tuple(int(v) for v in x.attrs["shape"])
    indptr = x["indptr"][...]
    dense = np.zeros((len(rows), shape[1]), dtype=np.float32)
    for out_row, source_row in enumerate(rows.tolist()):
        start = int(indptr[source_row])
        end = int(indptr[source_row + 1])
        if end <= start:
            continue
        cols = x["indices"][start:end]
        vals = x["data"][start:end]
        dense[out_row, cols] = vals
    return dense


def choose_targets(all_targets: set[str], n: int, seed: int) -> list[str]:
    targets = sorted(t for t in all_targets if t != CONTROL)
    if n <= 0 or n > len(targets):
        n = len(targets)
    rng = random.Random(seed)
    return sorted(rng.sample(targets, n))


def load_sample(h5ad_path: Path, official_genes: list[str], targets: list[str], max_rows_per_target: int, seed: int) -> Sample:
    import h5py

    wanted = [CONTROL] + list(targets)
    with h5py.File(h5ad_path, "r") as handle:
        var_names = _h5ad_index(handle["var"])
        if var_names != official_genes:
            raise ValueError(f"{h5ad_path} var_names do not match official gene order")
        target_col = _h5ad_column(handle["obs"], "target_gene")
        guide_col = _h5ad_column(handle["obs"], "guide_id")
        batch_col = _h5ad_column(handle["obs"], "batch")
        obs_names_all = _h5ad_index(handle["obs"])
        rng = np.random.default_rng(seed)
        rows = []
        target_array = np.asarray(target_col, dtype=object)
        for target in wanted:
            hits = np.flatnonzero(target_array == target)
            if len(hits) == 0:
                raise ValueError(f"{target} has no rows in {h5ad_path}")
            if max_rows_per_target > 0 and len(hits) > max_rows_per_target:
                hits = rng.choice(hits, size=max_rows_per_target, replace=False)
            rows.append(np.asarray(hits, dtype=np.int64))
        row_index = np.concatenate(rows)
        row_index.sort()
        x = _read_csr_rows(handle, row_index)
    return Sample(
        x=x,
        target_names_per_row=[target_col[i] for i in row_index],
        batch_names_per_row=[batch_col[i] for i in row_index],
        guide_names_per_row=[guide_col[i] for i in row_index],
        obs_names=[obs_names_all[i] for i in row_index],
        gene_names=official_genes,
    )


def select_sample_metadata(h5ad_path: Path, official_genes: list[str], targets: list[str], max_rows_per_target: int, seed: int) -> tuple[np.ndarray, Sample]:
    """Select rows without materializing X; used by the streaming validator."""
    import h5py

    wanted = [CONTROL] + list(targets)
    with h5py.File(h5ad_path, "r") as handle:
        var_names = _h5ad_index(handle["var"])
        if var_names != official_genes:
            raise ValueError(f"{h5ad_path} var_names do not match official gene order")
        target_col = _h5ad_column(handle["obs"], "target_gene")
        guide_col = _h5ad_column(handle["obs"], "guide_id")
        batch_col = _h5ad_column(handle["obs"], "batch")
        obs_names_all = _h5ad_index(handle["obs"])
        rng = np.random.default_rng(seed)
        target_array = np.asarray(target_col, dtype=object)
        rows = []
        for target in wanted:
            hits = np.flatnonzero(target_array == target)
            if len(hits) == 0:
                raise ValueError(f"{target} has no rows in {h5ad_path}")
            if max_rows_per_target > 0 and len(hits) > max_rows_per_target:
                hits = rng.choice(hits, size=max_rows_per_target, replace=False)
            rows.append(np.asarray(hits, dtype=np.int64))
        row_index = np.concatenate(rows)
        row_index.sort()
    return row_index, Sample(
        x=np.empty((len(row_index), len(official_genes)), dtype=np.float32),
        target_names_per_row=[target_col[i] for i in row_index],
        batch_names_per_row=[batch_col[i] for i in row_index],
        guide_names_per_row=[guide_col[i] for i in row_index],
        obs_names=[obs_names_all[i] for i in row_index],
        gene_names=official_genes,
    )


def iter_csr_rows(h5ad_path: Path, rows: np.ndarray, chunk_size: int = 64):
    """Yield dense chunks while keeping the H5AD expression matrix out of RAM."""
    import h5py

    with h5py.File(h5ad_path, "r") as handle:
        x = handle["X"]
        shape = tuple(int(v) for v in x.attrs["shape"])
        indptr = x["indptr"][...]
        for start in range(0, len(rows), chunk_size):
            chunk_rows = rows[start:start + chunk_size]
            dense = np.zeros((len(chunk_rows), shape[1]), dtype=np.float32)
            for out_row, source_row in enumerate(chunk_rows.tolist()):
                left, right = int(indptr[source_row]), int(indptr[source_row + 1])
                if right > left:
                    dense[out_row, x["indices"][left:right]] = x["data"][left:right]
            yield start, dense



def encode_rows(sample: Sample, target_order: list[str], batch_order: list[str], feature_map: dict[str, np.ndarray], allow_unknown_batch: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    target_index = {t: i for i, t in enumerate(target_order)}
    batch_index = {b: i for i, b in enumerate(batch_order)}
    global_batch = batch_index["__global__"]
    target_ids = []
    batch_ids = []
    features = []
    unknown_batches = []
    for target, batch in zip(sample.target_names_per_row, sample.batch_names_per_row):
        target_ids.append(target_index.get(target, -1))
        if batch in batch_index:
            batch_ids.append(batch_index[batch])
        elif allow_unknown_batch:
            batch_ids.append(global_batch)
            unknown_batches.append(batch)
        else:
            raise ValueError(f"unknown batch {batch}")
        features.append(feature_map[target])
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(target_ids, dtype=np.int64),
        np.asarray(batch_ids, dtype=np.int64),
        sorted(set(unknown_batches)),
    )


def control_baselines(x: np.ndarray, targets: list[str], batches: list[str], batch_order: list[str]) -> np.ndarray:
    baselines = np.zeros((len(batch_order), x.shape[1]), dtype=np.float32)
    control_mask = np.asarray([t == CONTROL for t in targets], dtype=bool)
    if not control_mask.any():
        raise ValueError("training sample has no controls")
    global_control = x[control_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    for i, batch in enumerate(batch_order):
        if batch == "__global__":
            baselines[i] = global_control
            continue
        mask = control_mask & (np.asarray(batches, dtype=object) == batch)
        baselines[i] = x[mask].mean(axis=0, dtype=np.float64).astype(np.float32) if mask.any() else global_control
    return baselines


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def mean_pairwise_l2(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    vals = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            vals.append(float(np.linalg.norm(x[i] - x[j])))
    return float(np.mean(vals)) if vals else 0.0



def delta_profile_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray, targets: list[str]) -> dict[str, float]:
    correlations = []
    true_norms = []
    pred_norms = []
    target_array = np.asarray(targets, dtype=object)
    for target in sorted(set(targets) - {CONTROL}):
        mask = target_array == target
        if not mask.any():
            continue
        truth = (y_true[mask] - baseline_pred[mask]).mean(axis=0, dtype=np.float64)
        pred = (y_pred[mask] - baseline_pred[mask]).mean(axis=0, dtype=np.float64)
        true_norms.append(float(np.linalg.norm(truth)))
        pred_norms.append(float(np.linalg.norm(pred)))
        centered_truth = truth - truth.mean()
        centered_pred = pred - pred.mean()
        denom = float(np.linalg.norm(centered_truth) * np.linalg.norm(centered_pred))
        if denom > 0:
            correlations.append(float(np.dot(centered_truth, centered_pred) / denom))
    mean_true_norm = float(np.mean(true_norms)) if true_norms else float("nan")
    mean_pred_norm = float(np.mean(pred_norms)) if pred_norms else float("nan")
    return {
        "delta_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "true_delta_l2_mean": mean_true_norm,
        "predicted_delta_l2_mean": mean_pred_norm,
        "predicted_to_true_delta_l2_ratio": float(mean_pred_norm / mean_true_norm) if mean_true_norm > 0 else float("nan"),
    }


def mean_profile_correlation(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> float:
    values = []
    target_array = np.asarray(targets, dtype=object)
    for target in sorted(set(targets) - {CONTROL}):
        mask = target_array == target
        if not mask.any():
            continue
        truth = y_true[mask].mean(axis=0, dtype=np.float64)
        pred = y_pred[mask].mean(axis=0, dtype=np.float64)
        truth = truth - truth.mean()
        pred = pred - pred.mean()
        denom = float(np.linalg.norm(truth) * np.linalg.norm(pred))
        if denom > 0:
            values.append(float(np.dot(truth, pred) / denom))
    return float(np.mean(values)) if values else float("nan")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    genes = read_gene_names(args.gene_names)
    train_targets_all = read_condition_targets(args.train_targets)
    val_targets_all = read_condition_targets(args.validation_targets)
    test_targets_all = read_condition_targets(args.test_targets)
    if train_targets_all & val_targets_all or train_targets_all & test_targets_all or val_targets_all & test_targets_all:
        raise ValueError("official target splits are not disjoint")
    train_targets = choose_targets(train_targets_all, args.n_train_targets, args.seed)
    val_targets = choose_targets(val_targets_all, args.n_val_targets, args.seed + 1)
    train = load_sample(args.train_h5ad, genes, train_targets, args.max_rows_per_target, args.seed + 10)
    val = load_sample(args.validation_h5ad, genes, val_targets, args.max_rows_per_target, args.seed + 20)

    all_feature_targets = sorted({CONTROL, *train_targets, *val_targets, *test_targets_all})
    hash_features = hashed_gene_symbol_features(all_feature_targets, genes, args.feature_dim)
    go_features, go_metadata = go_bp_hash_features(all_feature_targets, args.go_node_table, args.go_edge_table, args.feature_dim)
    symbol_to_gene_id = read_symbol_to_gene_id(args.train_h5ad, genes)
    embedding_features, embedding_metadata = projected_gene_embedding_features(
        all_feature_targets, symbol_to_gene_id, str(args.gene_embedding_npz), args.feature_dim, args.embedding_projection_seed
    )
    if args.representation == "hash":
        feature_matrix = hash_features
        representation_metadata = {
            "type": "deterministic_gene_symbol_hash_features",
            "feature_dim": args.feature_dim,
            "test_time_available": True,
            "uses_expression": False,
            "coverage_train_selected": len(train_targets),
            "coverage_validation_selected": len(val_targets),
            "coverage_test_panel": len(test_targets_all),
            "note": "Legal fixed representation from target_gene identity/symbol plus official gene index metadata.",
        }
    elif args.representation == "go":
        feature_matrix = go_features
        representation_metadata = {
            **go_metadata,
            "coverage_train_selected": int(np.count_nonzero(np.linalg.norm(go_features[[all_feature_targets.index(t) for t in train_targets]], axis=1) > 0)),
            "coverage_validation_selected": int(np.count_nonzero(np.linalg.norm(go_features[[all_feature_targets.index(t) for t in val_targets]], axis=1) > 0)),
            "coverage_test_panel": int(np.count_nonzero(np.linalg.norm(go_features[[all_feature_targets.index(t) for t in test_targets_all]], axis=1) > 0)),
            "coverage_insufficient_for_official_h1": go_metadata["coverage"] < 1.0,
        }
    elif args.representation == "embedding":
        feature_matrix = embedding_features
        representation_metadata = {
            **embedding_metadata,
            "coverage_train_selected": int(np.count_nonzero(np.linalg.norm(embedding_features[[all_feature_targets.index(t) for t in train_targets]], axis=1) > 0)),
            "coverage_validation_selected": int(np.count_nonzero(np.linalg.norm(embedding_features[[all_feature_targets.index(t) for t in val_targets]], axis=1) > 0)),
            "coverage_test_panel": int(np.count_nonzero(np.linalg.norm(embedding_features[[all_feature_targets.index(t) for t in test_targets_all]], axis=1) > 0)),
            "coverage_insufficient_for_official_h1": any(
                target in embedding_metadata["unmapped_sample"] for target in [*train_targets, *val_targets, *test_targets_all]
            ),
            "note": "Fixed pretrained gene embedding branch from local authorized Lingshu release asset, mapped through official H1 var gene_id; non-targeting control is not a perturbation target.",
        }
    elif args.representation == "hash_embedding":
        feature_matrix = np.concatenate([hash_features, embedding_features], axis=1)
        representation_metadata = {
            "type": "hash_plus_projected_fixed_gene_embedding_features",
            "feature_dim": int(feature_matrix.shape[1]),
            "hash_dim": args.feature_dim,
            "embedding_dim": args.feature_dim,
            "test_time_available": True,
            "uses_expression": False,
            "embedding_metadata": embedding_metadata,
            "coverage_insufficient_for_official_h1": any(
                target in embedding_metadata["unmapped_sample"] for target in [*train_targets, *val_targets, *test_targets_all]
            ),
            "note": "Hash branch covers all targets; embedding branch adds fixed pretrained biological/gene-prior information where mapped; non-targeting control is not a perturbation target.",
        }
    else:
        feature_matrix = np.concatenate([hash_features, go_features], axis=1)
        representation_metadata = {
            "type": "hash_plus_go_bp_materialized_graph_features",
            "feature_dim": int(feature_matrix.shape[1]),
            "hash_dim": args.feature_dim,
            "go_dim": args.feature_dim,
            "test_time_available": True,
            "uses_expression": False,
            "go_metadata": go_metadata,
            "coverage_insufficient_for_official_h1": go_metadata["coverage"] < 1.0,
            "note": "Hash branch covers all targets; GO branch is included only where the existing fixed graph maps a target.",
        }
    feature_map = {target: feature_matrix[i] for i, target in enumerate(all_feature_targets)}
    train_target_order = [CONTROL] + train_targets
    batch_order = ["__global__"] + sorted(set(train.batch_names_per_row))
    train_features, train_target_ids, train_batch_ids, _ = encode_rows(train, train_target_order, batch_order, feature_map, False)
    val_features, val_target_ids, val_batch_ids, unknown_val_batches = encode_rows(val, train_target_order, batch_order, feature_map, True)
    baselines = control_baselines(train.x, train.target_names_per_row, train.batch_names_per_row, batch_order)
    baseline_pred = baselines[val_batch_ids]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = TargetFeatureCRPM(torch.from_numpy(baselines), int(feature_matrix.shape[1]), args.rank, len(train_target_order), len(batch_order), args.learned_id)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ds = TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train_features), torch.from_numpy(train_target_ids), torch.from_numpy(train_batch_ids))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    losses = []
    steps = 0
    while steps < args.max_steps:
        for y, feat, tid, bid in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(feat, bid, tid, allow_id=True)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            steps += 1
            if steps >= args.max_steps:
                break
    with torch.no_grad():
        val_pred = model(torch.from_numpy(val_features), torch.from_numpy(val_batch_ids), None, allow_id=False).numpy().astype(np.float32)
        val_conditions = model.condition(torch.from_numpy(np.asarray([feature_map[t] for t in val_targets], dtype=np.float32)), None, allow_id=False).numpy()
        train_conditions_fixed = model.condition(torch.from_numpy(np.asarray([feature_map[t] for t in train_targets], dtype=np.float32)), None, allow_id=False).numpy()
    target_means = []
    for target in val_targets:
        mask = np.asarray([t == target for t in val.target_names_per_row], dtype=bool)
        target_means.append(val_pred[mask].mean(axis=0))
    target_means = np.asarray(target_means, dtype=np.float32)
    val_mse = mse(val.x, val_pred)
    base_mse = mse(val.x, baseline_pred)
    delta_diagnostics = delta_profile_diagnostics(val.x, val_pred, baseline_pred, val.target_names_per_row)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "pass" if np.isfinite(val_pred).all() and mean_pairwise_l2(val_conditions) > 0 and mean_pairwise_l2(target_means) > 0 else "failed",
        "data_contract": {
            "train_perturb_targets_total": len(train_targets_all),
            "validation_perturb_targets_total": len(val_targets_all),
            "test_perturb_targets_total": len(test_targets_all),
            "split_disjointness": True,
            "gene_count": len(genes),
            "gene_order_preserved": True,
            "test_expression_read": False,
        },
        "representation": representation_metadata,
        "model": {
            "objective": "predict perturbation response / residual via CRPM-style baseline plus low-rank gene programs",
            "target_conditioning": "fixed target features encoded to rank weights; learned id correction used only for train targets during training" if args.learned_id else "fixed target features encoded to rank weights only",
            "learned_id_correction_allowed_for_validation": False,
            "rank": args.rank,
            "max_steps": args.max_steps,
            "optimizer_steps": steps,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        },
        "validation_probe": {
            "targets": val_targets,
            "rows": int(len(val.x)),
            "prediction_shape": list(val_pred.shape),
            "prediction_finite": bool(np.isfinite(val_pred).all()),
            "baseline_mse": base_mse,
            "fixed_feature_mse": val_mse,
            "delta_mse_vs_baseline": float(base_mse - val_mse),
            "baseline_mean_profile_correlation": mean_profile_correlation(val.x, baseline_pred, val.target_names_per_row),
            "fixed_feature_mean_profile_correlation": mean_profile_correlation(val.x, val_pred, val.target_names_per_row),
            "delta_mean_profile_correlation_vs_baseline": float(mean_profile_correlation(val.x, val_pred, val.target_names_per_row) - mean_profile_correlation(val.x, baseline_pred, val.target_names_per_row)),
            "conditioning_pairwise_l2_mean": mean_pairwise_l2(val_conditions),
            "prediction_target_mean_pairwise_l2_mean": mean_pairwise_l2(target_means),
            **delta_diagnostics,
            "unknown_validation_batches_mapped_to_global_control": unknown_val_batches,
        },
        "train_probe": {
            "targets": train_targets,
            "rows": int(len(train.x)),
            "train_fixed_condition_pairwise_l2_mean": mean_pairwise_l2(train_conditions_fixed),
        },
        "official_h1_prediction": "not_generated_validation_probe_only",
        "official_score_claim": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official H1 cold-start target-feature feasibility probe")
    parser.add_argument("--train-h5ad", type=Path, default=DEFAULT_ASSETS / "train/adata_Training.h5ad")
    parser.add_argument("--validation-h5ad", type=Path, default=DEFAULT_ASSETS / "validation/adata_Validation.h5ad")
    parser.add_argument("--test-h5ad", type=Path, default=DEFAULT_ASSETS / "test/adata_Test.h5ad")
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_ASSETS / "train/pert_counts_Training.csv")
    parser.add_argument("--validation-targets", type=Path, default=DEFAULT_ASSETS / "validation/pert_counts_Validation.csv")
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_ASSETS / "test/pert_counts_Test.csv")
    parser.add_argument("--gene-names", type=Path, default=DEFAULT_ASSETS / "gene_names.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-train-targets", type=int, default=16)
    parser.add_argument("--n-val-targets", type=int, default=8)
    parser.add_argument("--max-rows-per-target", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--representation", choices=["hash", "go", "hash_go", "embedding", "hash_embedding"], default="hash")
    parser.add_argument("--go-node-table", type=Path, default=Path("/data/zhangzhicheng/omni-ar_heuresis/datasets/vcc25/annotations/go-bp-20260818/node_table.csv"))
    parser.add_argument("--go-edge-table", type=Path, default=Path("/data/zhangzhicheng/omni-ar_heuresis/datasets/vcc25/annotations/go-bp-20260818/edge_table.csv"))
    parser.add_argument("--gene-embedding-npz", type=Path, default=DEFAULT_ROOT / "assets/lingshu_hf_b77f980/gene_embeddings.npz")
    parser.add_argument("--embedding-projection-seed", type=int, default=20260908)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--learned-id", action="store_true")
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
