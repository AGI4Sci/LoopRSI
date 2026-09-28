from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from crpm.cold_start import TargetFeatureCRPM
from crpm.target_features import (
    go_bp_hash_features,
    hashed_gene_symbol_features,
    pca_gene_embedding_features,
    projected_gene_embedding_features,
)
from official_h1_autonomous_research_loop import AmplitudeAwareHead, delta_diagnostics, train_delta_weights
from official_h1_cold_start_probe import (
    CONTROL,
    choose_targets,
    control_baselines,
    encode_rows,
    load_sample,
    mse,
    read_symbol_to_gene_id,
)
from official_h1_contract import read_condition_targets, read_gene_names
from official_h1_neighborhood_prior_probe import (
    DEFAULT_ASSETS,
    DEFAULT_ROOT,
    load_embedding_matrix,
    train_direction_stats,
)
from official_h1_nonnegative_probe import normalize_log1p

PROTOCOL_ID = "vcc-h1-pathway-conditioned-delta-basis-v1"


def build_features(args, genes, targets, train_targets):
    symbol_to_gene_id = read_symbol_to_gene_id(args.train_h5ad, genes)
    hashed = hashed_gene_symbol_features(targets, genes, args.feature_dim)
    if args.representation in {"pca", "go_pca"}:
        embedded, metadata = pca_gene_embedding_features(
            targets, train_targets, symbol_to_gene_id, str(args.gene_embedding_npz), args.feature_dim
        )
    else:
        embedded, metadata = projected_gene_embedding_features(
            targets, symbol_to_gene_id, str(args.gene_embedding_npz), args.feature_dim, args.embedding_projection_seed
        )
    branches = [hashed, embedded]
    if args.representation == "go_pca":
        go_features, go_metadata = go_bp_hash_features(
            targets, str(args.go_node_table), str(args.go_edge_table), args.go_feature_dim
        )
        branches.append(go_features)
        metadata = {"embedding": metadata, "go": go_metadata, "type": "hash_plus_train_pca_plus_official_go_bp"}
    values = np.concatenate(branches, axis=1)
    return {target: values[index] for index, target in enumerate(targets)}, symbol_to_gene_id, metadata


def train_predict(
    args, seed, representation, use_direction_loss, genes, train_all, val_all, test_all,
    shared_embedding_matrix=None, shared_embedding_coverage=None,
):
    args.representation = representation
    train_targets = choose_targets(train_all, args.n_train_targets, seed)
    val_targets = choose_targets(val_all, args.n_val_targets, seed + 1)
    train = load_sample(args.train_h5ad, genes, train_targets, args.max_rows_per_target, seed + 10)
    val = load_sample(args.validation_h5ad, genes, val_targets, args.max_rows_per_target, seed + 20)
    all_targets = sorted({CONTROL, *train_all, *val_all, *test_all})
    fmap, symbol_to_gene_id, feature_metadata = build_features(args, genes, all_targets, train_targets)
    go_values, go_metadata = go_bp_hash_features(
        all_targets, str(args.go_node_table), str(args.go_edge_table), args.go_feature_dim
    )
    go_features = {target: go_values[index] for index, target in enumerate(all_targets)}
    batch_order = ["__global__"] + sorted(set(train.batch_names_per_row))
    train_log, target_sum = normalize_log1p(train.x)
    val_log, _ = normalize_log1p(val.x, target_sum)
    trf, trtid, trbid, _ = encode_rows(train, [CONTROL] + train_targets, batch_order, fmap, False)
    vf, _, vbid, _ = encode_rows(val, [CONTROL] + train_targets, batch_order, fmap, True)
    baselines = control_baselines(train_log, train.target_names_per_row, train.batch_names_per_row, batch_order)
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = TargetFeatureCRPM(torch.from_numpy(baselines), trf.shape[1], args.rank, len(train_targets) + 1, len(batch_order), False)
    model = AmplitudeAwareHead(base, args.rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.from_numpy(train_log), torch.from_numpy(trf), torch.from_numpy(trtid), torch.from_numpy(trbid))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    gene_weights = torch.from_numpy(train_delta_weights(train_log, train.target_names_per_row, train.batch_names_per_row, batch_order))
    steps = 0
    while steps < args.max_steps:
        for truth, features, target_ids, batch_ids in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, batch_ids, target_ids, allow_id=False)
            loss = torch.mean((prediction - truth) ** 2 * gene_weights)
            if use_direction_loss:
                anchor = torch.from_numpy(baselines)[batch_ids]
                pred_delta = prediction - anchor
                true_delta = truth - anchor
                cosine = torch.nn.functional.cosine_similarity(pred_delta, true_delta, dim=1, eps=1e-6)
                loss = loss + args.direction_weight * torch.mean(1.0 - cosine)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= args.max_steps:
                break
    with torch.no_grad():
        base_prediction = model(torch.from_numpy(vf), torch.from_numpy(vbid), None, allow_id=False).numpy().astype(np.float32)
    if shared_embedding_matrix is None:
        embedding_matrix, mapped = load_embedding_matrix(genes, symbol_to_gene_id, str(args.gene_embedding_npz))
        embedding_coverage = float(mapped.mean())
    else:
        embedding_matrix = shared_embedding_matrix
        embedding_coverage = float(shared_embedding_coverage)
    _, _, target_delta = train_direction_stats(train_log, train, genes, batch_order)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    self_values = [float(delta[gene_index[target]]) for target, delta in target_delta.items() if target in gene_index]
    reliability = train_delta_weights(train_log, train.target_names_per_row, train.batch_names_per_row, batch_order) - 1.0
    reliability = reliability / max(float(reliability.max()), 1e-6)
    return {
        "seed": seed,
        "representation": representation,
        "direction_loss": use_direction_loss,
        "feature_metadata": feature_metadata,
        "go_features": go_features,
        "go_metadata": go_metadata,
        "genes": genes,
        "train_targets": train_targets,
        "val": val,
        "val_log": val_log,
        "baseline": baselines[vbid],
        "base_prediction": base_prediction,
        "embedding_matrix": embedding_matrix,
        "embedding_coverage": embedding_coverage,
        "target_delta": target_delta,
        "self_effect": float(np.median(self_values)) if self_values else 0.0,
        "reliability": reliability.astype(np.float32),
    }


def pathway_neighbors(run, target, k, pathway_alpha, shuffle_go):
    genes = run["genes"]
    gene_index = {gene: index for index, gene in enumerate(genes)}
    if target not in gene_index:
        return []
    target_embedding = run["embedding_matrix"][gene_index[target]]
    train_targets = [item for item in run["train_targets"] if item in gene_index]
    embedding_scores = np.asarray([
        float(target_embedding @ run["embedding_matrix"][gene_index[item]])
        for item in train_targets
    ], dtype=np.float32)
    go_target = run["go_features"].get(target)
    go_sources = train_targets
    if shuffle_go:
        rng = np.random.default_rng(run["seed"] + 991)
        go_sources = list(np.asarray(train_targets)[rng.permutation(len(train_targets))])
    go_scores = np.asarray([
        float(go_target @ run["go_features"][source]) if go_target is not None else 0.0
        for source in go_sources
    ], dtype=np.float32)
    scores = (1.0 - pathway_alpha) * embedding_scores + pathway_alpha * go_scores
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[index]), train_targets[index]) for index in order]


def predict_arm(run, k, temperature, scale, gate_power, calibration, pathway_alpha=0.0, shuffle_go=False):
    genes = run["genes"]
    gene_index = {gene: index for index, gene in enumerate(genes)}
    targets = run["val"].target_names_per_row
    prior = np.zeros_like(run["base_prediction"])
    cache = {}
    for row, target in enumerate(targets):
        if target == CONTROL:
            continue
        if target not in cache:
            neighbors = pathway_neighbors(run, target, k, pathway_alpha, shuffle_go)
            if neighbors:
                similarities = np.asarray([similarity for similarity, _ in neighbors], dtype=np.float32)
                weights = np.exp((similarities - similarities.max()) / temperature)
                weights /= weights.sum() + 1e-8
                value = sum(float(weight) * run["target_delta"][neighbor] for weight, (_, neighbor) in zip(weights, neighbors))
            else:
                value = np.zeros(len(genes), dtype=np.float32)
            if target in gene_index:
                value = value.copy()
                value[gene_index[target]] += run["self_effect"]
            cache[target] = value
        prior[row] = cache[target]
    if gate_power > 0:
        prior *= np.power(run["reliability"] + 1e-6, gate_power)[None, :]
    base = np.maximum(run["base_prediction"], 1e-6)
    base_logit = base + np.log(-np.expm1(-base))
    prediction = np.logaddexp(base_logit + scale * prior, 0.0).astype(np.float32)
    prediction = np.maximum(0.0, run["baseline"] + calibration * (prediction - run["baseline"])).astype(np.float32)
    return prediction


def score_folds(run, prediction, folds):
    targets = np.asarray(run["val"].target_names_per_row, dtype=object)
    unique = sorted(set(targets) - {CONTROL})
    rows = []
    for fold in range(folds):
        selected = {target for index, target in enumerate(unique) if index % folds == fold}
        mask = np.asarray([target in selected for target in targets])
        diagnostics = delta_diagnostics(run["val_log"][mask], prediction[mask], run["baseline"][mask], targets[mask].tolist())
        rows.append({"fold": fold, "delta_correlation": diagnostics["delta_correlation"], "de_direction": diagnostics["de_direction_agreement_top5pct"], "mse": mse(run["val_log"][mask], prediction[mask])})
    return rows


def select(stage, candidates, runs, fixed, folds):
    results = []
    for candidate in candidates:
        config = {**fixed, **candidate}
        fold_rows = []
        for run in runs:
            prediction = predict_arm(
                run, config["k"], config["temperature"], config["scale"],
                config["gate_power"], config["calibration"],
                config.get("pathway_alpha", 0.0), config.get("shuffle_go", False),
            )
            fold_rows.extend({**row, "seed": run["seed"]} for row in score_folds(run, prediction, folds))
        summary = {
            "stage": stage,
            "config": config,
            "delta_correlation_mean": float(np.mean([row["delta_correlation"] for row in fold_rows])),
            "de_direction_mean": float(np.mean([row["de_direction"] for row in fold_rows])),
            "mse_mean": float(np.mean([row["mse"] for row in fold_rows])),
            "fold_runs": fold_rows,
        }
        results.append(summary)
    if stage == "amplitude_calibration":
        results.sort(key=lambda row: (row["mse_mean"], -row["delta_correlation_mean"], -row["de_direction_mean"]))
    else:
        results.sort(key=lambda row: (row["delta_correlation_mean"], row["de_direction_mean"], -row["mse_mean"]), reverse=True)
    return results[0], results


def run(args):
    genes = read_gene_names(args.gene_names)
    train_all = read_condition_targets(args.train_targets)
    val_all = read_condition_targets(args.validation_targets)
    test_all = read_condition_targets(args.test_targets)
    if train_all & val_all or train_all & test_all or val_all & test_all:
        raise ValueError("official target splits are not disjoint")
    seeds = [int(value) for value in args.seeds.split(",")]
    symbol_to_gene_id = read_symbol_to_gene_id(args.train_h5ad, genes)
    shared_embedding_matrix, mapped = load_embedding_matrix(
        genes, symbol_to_gene_id, str(args.gene_embedding_npz)
    )
    shared_embedding_coverage = float(mapped.mean())
    trained = {}
    for representation in ["random_projection", "pca", "go_pca"]:
        for direction_loss in [False, True]:
            trained[(representation, direction_loss)] = [
                train_predict(
                    args, seed, representation, direction_loss, genes, train_all, val_all, test_all,
                    shared_embedding_matrix, shared_embedding_coverage,
                )
                for seed in seeds
            ]
    default = {"k": 8, "temperature": 1.0, "scale": 0.75, "gate_power": 0.0, "calibration": 1.0, "pathway_alpha": 0.0, "shuffle_go": False}
    representation_rows = []
    for key, runs in trained.items():
        best, _ = select("representation_loss", [{"representation": key[0], "direction_loss": key[1]}], runs, default, args.folds)
        representation_rows.append(best)
    representation_rows.sort(key=lambda row: (row["delta_correlation_mean"], row["de_direction_mean"], -row["mse_mean"]), reverse=True)
    best_rep = representation_rows[0]
    runs = trained[(best_rep["config"]["representation"], best_rep["config"]["direction_loss"])]
    pathway_candidates = [
        {"k": k, "temperature": temperature, "scale": scale, "pathway_alpha": alpha, "shuffle_go": False}
        for k in [64, 96, 128, 150]
        for temperature in [0.25, 0.5, 1.0]
        for scale in [1.25, 1.5, 1.75]
        for alpha in [0.0, 0.75, 1.0]
    ]
    best_pathway, pathway_rows = select(
        "joint_pathway_conditioning", pathway_candidates, runs, best_rep["config"], args.folds
    )
    selected_alpha = best_pathway["config"]["pathway_alpha"]
    _, shuffled_rows = select(
        "pathway_shuffled_control",
        [{"pathway_alpha": selected_alpha, "shuffle_go": True}],
        runs, best_pathway["config"], args.folds,
    )
    shuffled_control = shuffled_rows[0]
    embedding_only = max(
        (row for row in pathway_rows if row["config"]["pathway_alpha"] == 0.0),
        key=lambda row: (row["delta_correlation_mean"], row["de_direction_mean"], -row["mse_mean"]),
    )
    best_gate, gate_rows = select("de_reliability_gate", [{"gate_power": power} for power in [0.0, 0.5, 1.0]], runs, best_pathway["config"], args.folds)
    best_calibration, calibration_rows = select("amplitude_calibration", [{"calibration": value} for value in [0.75, 1.0, 1.25]], runs, best_gate["config"], args.folds)
    go_rows = [row for row in representation_rows if row["config"]["representation"] == "go_pca"]
    non_go_rows = [row for row in representation_rows if row["config"]["representation"] != "go_pca"]
    go_gain = max(row["delta_correlation_mean"] for row in go_rows) - max(row["delta_correlation_mean"] for row in non_go_rows)
    go_coverage = min(run["feature_metadata"]["go"]["coverage"] for key, rows in trained.items() if key[0] == "go_pca" for run in rows)
    promoted = (
        best_calibration["delta_correlation_mean"] > args.promotion_reference
        and go_coverage >= args.minimum_go_coverage
        and selected_alpha > 0.0
        and best_pathway["delta_correlation_mean"] > embedding_only["delta_correlation_mean"]
        and best_pathway["delta_correlation_mean"] > shuffled_control["delta_correlation_mean"]
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "search_policy": "single_predeclared_boundary_extension_after_k64_scale1.25_boundary_optimum",
        "status": "promoted" if promoted else "rejected",
        "seeds": seeds,
        "folds": args.folds,
        "test_expression_used": False,
        "official_test_prediction_generated": False,
        "representation_loss": representation_rows,
        "pathway_conditioning": pathway_rows,
        "pathway_shuffled_control": shuffled_control,
        "de_reliability_gate": gate_rows,
        "amplitude_calibration": calibration_rows,
        "promoted_config": best_calibration["config"],
        "promotion_metrics": {key: best_calibration[key] for key in ["delta_correlation_mean", "de_direction_mean", "mse_mean"]},
        "promotion_reference": args.promotion_reference,
        "promotion_margin": best_calibration["delta_correlation_mean"] - args.promotion_reference,
        "go_representation_gate": {"coverage": go_coverage, "minimum_coverage": args.minimum_go_coverage, "delta_correlation_gain_vs_best_non_go": go_gain, "passed": go_coverage >= args.minimum_go_coverage and go_gain > 0.0},
        "pathway_conditioning_gate": {
            "selected_alpha": selected_alpha,
            "gain_vs_embedding_only": best_pathway["delta_correlation_mean"] - embedding_only["delta_correlation_mean"],
            "gain_vs_shuffled_control": best_pathway["delta_correlation_mean"] - shuffled_control["delta_correlation_mean"],
            "passed": selected_alpha > 0.0 and best_pathway["delta_correlation_mean"] > embedding_only["delta_correlation_mean"] and best_pathway["delta_correlation_mean"] > shuffled_control["delta_correlation_mean"],
        },
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
    parser.add_argument("--go-feature-dim", type=int, default=128)
    parser.add_argument("--minimum-go-coverage", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-train-targets", type=int, default=150)
    parser.add_argument("--n-val-targets", type=int, default=50)
    parser.add_argument("--max-rows-per-target", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--embedding-projection-seed", type=int, default=20260908)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--direction-weight", type=float, default=0.05)
    parser.add_argument("--seeds", default="20260907,20260908,20260909")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--promotion-reference", type=float, default=0.608694)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
