from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from crpm.target_features import go_bp_ic_features
from official_h1_autonomous_research_loop import delta_diagnostics
from official_h1_candidate_h_research import train_predict
from official_h1_cold_start_probe import CONTROL, DEFAULT_ASSETS, DEFAULT_ROOT, mse, read_symbol_to_gene_id
from official_h1_contract import read_condition_targets, read_gene_names
from official_h1_neighborhood_prior_probe import load_embedding_matrix

PROTOCOL_ID = "vcc-h1-candidate-i-pathway-basis-v1"


def target_matrix(run, targets: list[str]) -> np.ndarray:
    return np.asarray([run["target_delta"][target] for target in targets], dtype=np.float32)


def fit_pathway_basis(run, feature_map, rank, ridge, de_strength, shuffle, seed):
    cache = run.setdefault("pathway_basis_cache", {})
    cache_key = (rank, ridge, de_strength, shuffle)
    if cache_key in cache:
        return cache[cache_key]
    targets = run["train_targets"]
    delta = target_matrix(run, targets)
    basis_key = (rank, de_strength)
    basis_cache = run.setdefault("pathway_svd_cache", {})
    if basis_key not in basis_cache:
        reliability = np.mean(np.abs(delta), axis=0)
        reliability /= max(float(reliability.max()), 1e-8)
        gene_scale = np.sqrt(1.0 + de_strength * reliability).astype(np.float32)
        weighted = delta * gene_scale[None, :]
        _, _, vt = np.linalg.svd(weighted, full_matrices=False)
        basis_cache[basis_key] = vt[:rank] / gene_scale[None, :]
    basis = basis_cache[basis_key]
    coefficients = delta @ basis.T
    features = np.asarray([feature_map[target] for target in targets], dtype=np.float32)
    if shuffle:
        features = features[np.random.default_rng(seed + 1701).permutation(len(features))]
    design = np.concatenate([np.ones((len(features), 1), dtype=np.float32), features], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float32) * ridge
    penalty[0, 0] = 0.0
    mapping = np.linalg.solve(design.T @ design + penalty, design.T @ coefficients)
    cache[cache_key] = (basis, mapping)
    return cache[cache_key]


def predict(run, feature_map, config, shuffle=False):
    prior_key = (
        config["basis_rank"], config["ridge"], config["de_strength"],
        config["confidence_power"], shuffle,
    )
    prior_cache = run.setdefault("pathway_prior_cache", {})
    basis, mapping = fit_pathway_basis(
        run, feature_map, config["basis_rank"], config["ridge"],
        config["de_strength"], shuffle, run["seed"],
    )
    if prior_key not in prior_cache:
        train_features = np.asarray([feature_map[target] for target in run["train_targets"]], dtype=np.float32)
        global_delta = target_matrix(run, run["train_targets"]).mean(axis=0)
        targets = run["val"].target_names_per_row
        target_cache = {}
        for target in sorted(set(targets)):
            if target == CONTROL:
                continue
            feature = feature_map[target]
            coefficient = np.concatenate([[1.0], feature]).astype(np.float32) @ mapping
            pathway_delta = coefficient @ basis
            confidence = float(np.clip(np.max(train_features @ feature), 0.0, 1.0))
            confidence = confidence ** config["confidence_power"]
            target_cache[target] = confidence * pathway_delta + (1.0 - confidence) * global_delta
        prior_cache[prior_key] = target_cache
    target_cache = prior_cache[prior_key]
    zero = np.zeros(run["base_prediction"].shape[1], dtype=np.float32)
    priors = np.asarray([target_cache.get(target, zero) for target in run["val"].target_names_per_row])
    base = np.maximum(run["base_prediction"], 1e-6)
    base_logit = base + np.log(-np.expm1(-base))
    prediction = np.logaddexp(base_logit + config["scale"] * priors, 0.0).astype(np.float32)
    return np.maximum(
        0.0, run["baseline"] + config["calibration"] * (prediction - run["baseline"])
    ).astype(np.float32)


def score(run, prediction, folds, salt):
    targets = np.asarray(run["val"].target_names_per_row, dtype=object)
    rows = []
    for fold in range(folds):
        selected = {
            target for target in set(targets) - {CONTROL}
            if int.from_bytes(hashlib.sha256(f"{salt}:{target}".encode()).digest()[:4], "little") % folds == fold
        }
        mask = np.asarray([target in selected for target in targets])
        diagnostics = delta_diagnostics(
            run["val_log"][mask], prediction[mask], run["baseline"][mask], targets[mask].tolist()
        )
        rows.append({
            "fold": fold,
            "delta_correlation": diagnostics["delta_correlation"],
            "de_direction": diagnostics["de_direction_agreement_top5pct"],
            "mse": mse(run["val_log"][mask], prediction[mask]),
        })
    return rows


def evaluate(config, runs, feature_maps, folds, salt, shuffle=False):
    rows = []
    for run, feature_map in zip(runs, feature_maps):
        prediction = predict(run, feature_map, config, shuffle=shuffle)
        rows.extend({**item, "seed": run["seed"]} for item in score(run, prediction, folds, salt))
    return {
        "config": config,
        "shuffle_go": shuffle,
        "delta_correlation_mean": float(np.mean([row["delta_correlation"] for row in rows])),
        "de_direction_mean": float(np.mean([row["de_direction"] for row in rows])),
        "mse_mean": float(np.mean([row["mse"] for row in rows])),
        "fold_runs": rows,
    }


def run(args):
    genes = read_gene_names(args.gene_names)
    train_targets = read_condition_targets(args.train_targets)
    validation_targets = read_condition_targets(args.validation_targets)
    test_targets = read_condition_targets(args.test_targets)
    if train_targets & validation_targets or train_targets & test_targets or validation_targets & test_targets:
        raise ValueError("official target splits are not disjoint")
    seeds = [int(value) for value in args.seeds.split(",")]
    symbol_to_gene_id = read_symbol_to_gene_id(args.train_h5ad, genes)
    embedding_matrix, mapped = load_embedding_matrix(genes, symbol_to_gene_id, str(args.gene_embedding_npz))
    embedding_coverage = float(mapped.mean())
    runs = [
        train_predict(
            args, seed, "pca", True, genes, train_targets, validation_targets, test_targets,
            embedding_matrix, embedding_coverage,
        )
        for seed in seeds
    ]
    all_targets = sorted({CONTROL, *train_targets, *validation_targets, *test_targets})
    feature_maps = []
    metadata = []
    for run_state in runs:
        values, details = go_bp_ic_features(
            all_targets, run_state["train_targets"], str(args.go_node_table),
            str(args.go_edge_table), args.go_feature_dim,
        )
        feature_maps.append({target: values[index] for index, target in enumerate(all_targets)})
        metadata.append(details)
    candidates = [
        {
            "basis_rank": rank, "ridge": ridge, "de_strength": de_strength,
            "confidence_power": confidence, "scale": scale, "calibration": calibration,
        }
        for rank in [8, 16, 32]
        for ridge in [0.01, 0.1, 1.0, 10.0]
        for de_strength in [0.0, 1.0]
        for confidence in [0.0, 1.0, 2.0]
        for scale in [0.5, 1.0, 1.5]
        for calibration in [0.75, 1.0]
    ]
    results = [evaluate(item, runs, feature_maps, args.folds, args.fold_salt) for item in candidates]
    results.sort(key=lambda row: (row["delta_correlation_mean"], row["de_direction_mean"], -row["mse_mean"]), reverse=True)
    best = results[0]
    shuffled = evaluate(best["config"], runs, feature_maps, args.folds, args.fold_salt, shuffle=True)
    promoted = (
        best["delta_correlation_mean"] > args.promotion_reference
        and best["delta_correlation_mean"] > shuffled["delta_correlation_mean"]
        and min(item["coverage"] for item in metadata) >= args.minimum_go_coverage
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "status": "promoted" if promoted else "rejected",
        "search_policy": "single_bounded_candidate_i_search_on_new_hash_target_folds",
        "seeds": seeds,
        "folds": args.folds,
        "fold_salt": args.fold_salt,
        "test_expression_used": False,
        "official_test_prediction_generated": False,
        "go_metadata": metadata,
        "best": best,
        "shuffled_control": shuffled,
        "promotion_reference": args.promotion_reference,
        "promotion_margin": best["delta_correlation_mean"] - args.promotion_reference,
        "gain_vs_shuffled": best["delta_correlation_mean"] - shuffled["delta_correlation_mean"],
        "search_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, default=DEFAULT_ASSETS / "train/adata_Training.h5ad")
    parser.add_argument("--validation-h5ad", type=Path, default=DEFAULT_ASSETS / "validation/adata_Validation.h5ad")
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_ASSETS / "train/pert_counts_Training.csv")
    parser.add_argument("--validation-targets", type=Path, default=DEFAULT_ASSETS / "validation/pert_counts_Validation.csv")
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_ASSETS / "test/pert_counts_Test.csv")
    parser.add_argument("--gene-names", type=Path, default=DEFAULT_ASSETS / "gene_names.csv")
    parser.add_argument("--gene-embedding-npz", type=Path, default=DEFAULT_ROOT / "assets/lingshu_hf_b77f980/gene_embeddings.npz")
    parser.add_argument("--go-node-table", type=Path, default=Path(__file__).resolve().parents[3] / "datasets/vcc25/annotations/official-h1-go-bp-v1/node_table.csv")
    parser.add_argument("--go-edge-table", type=Path, default=Path(__file__).resolve().parents[3] / "datasets/vcc25/annotations/official-h1-go-bp-v1/edge_table.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-train-targets", type=int, default=150)
    parser.add_argument("--n-val-targets", type=int, default=50)
    parser.add_argument("--max-rows-per-target", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--go-feature-dim", type=int, default=256)
    parser.add_argument("--embedding-projection-seed", type=int, default=20260908)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--direction-weight", type=float, default=0.05)
    parser.add_argument("--seeds", default="20260907,20260908,20260909")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold-salt", default="candidate-i-new-folds-20260913")
    parser.add_argument("--promotion-reference", type=float, default=0.608694)
    parser.add_argument("--minimum-go-coverage", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
