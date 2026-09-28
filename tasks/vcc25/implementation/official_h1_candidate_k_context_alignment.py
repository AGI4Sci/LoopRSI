from __future__ import annotations

import json

import numpy as np

import official_h1_candidate_i_pathway_basis as ci
import official_h1_candidate_j_external_basis as cj
from official_h1_contract import read_condition_targets, read_gene_names

PROTOCOL_ID = "vcc-h1-candidate-k-context-alignment-v1"


def context_aligned_model(run, artifact, config, shuffle_features=False, shuffle_context=False):
    cache = run.setdefault("context_alignment_cache", {})
    key = (config["external_rank"], config["external_ridge"], config["context_rank"],
           config["context_ridge"], shuffle_features, shuffle_context)
    if key in cache:
        return cache[key]
    ext_basis, ext_mapping, ext_features = cj.external_model(
        run, artifact, config["external_rank"], config["external_ridge"], shuffle_features
    )
    ext_targets = artifact["targets"].astype(str).tolist()
    ext_delta = artifact["delta"].astype(np.float32)
    ext_index = {target: index for index, target in enumerate(ext_targets)}
    anchors = sorted(set(run["train_targets"]) & set(ext_targets))
    if len(anchors) < config["minimum_anchors"]:
        raise ValueError(f"insufficient train-only context anchors: {len(anchors)}")
    h1_delta = ci.target_matrix(run, run["train_targets"])
    _, _, h1_vt = np.linalg.svd(h1_delta, full_matrices=False)
    h1_basis = h1_vt[:config["context_rank"]]
    ext_coeff = ext_delta[[ext_index[target] for target in anchors]] @ ext_basis.T
    h1_coeff = np.asarray([run["target_delta"][target] for target in anchors]) @ h1_basis.T
    if shuffle_context:
        h1_coeff = h1_coeff[np.random.default_rng(run["seed"] + 3907).permutation(len(h1_coeff))]
    design = np.c_[np.ones(len(ext_coeff), dtype=np.float32), ext_coeff]
    penalty = config["context_ridge"] * np.eye(design.shape[1], dtype=np.float32)
    penalty[0, 0] = 0.0
    context_mapping = np.linalg.solve(design.T @ design + penalty, design.T @ h1_coeff)
    cache[key] = (ext_basis, ext_mapping, ext_features, context_mapping, h1_basis, anchors)
    return cache[key]


def predict_aligned(run, artifact, config, shuffle_features=False, shuffle_context=False):
    ext_basis, ext_mapping, ext_features, context_mapping, h1_basis, anchors = context_aligned_model(
        run, artifact, config, shuffle_features, shuffle_context
    )
    gene_index = {gene: index for index, gene in enumerate(run["genes"])}
    cache = {}
    rows = []
    for target in run["val"].target_names_per_row:
        if target not in cache:
            if target not in gene_index:
                cache[target] = np.zeros(len(run["genes"]), dtype=np.float32)
            else:
                feature = run["embedding_matrix"][gene_index[target]]
                confidence = max(0.0, float(np.max(ext_features @ feature))) ** config["confidence_power"]
                ext_coeff = np.r_[1.0, feature] @ ext_mapping
                h1_coeff = np.r_[1.0, ext_coeff] @ context_mapping
                cache[target] = confidence * (h1_coeff @ h1_basis)
        rows.append(cache[target])
    return np.asarray(rows, dtype=np.float32), len(anchors)


def evaluate(config, runs, feature_maps, artifact, folds, salt, control="none"):
    rows, anchor_counts = [], []
    for state, feature_map in zip(runs, feature_maps):
        internal = ci.predict(state, feature_map, config["internal"])
        aligned, anchors = predict_aligned(
            state, artifact, config, shuffle_features=control == "features",
            shuffle_context=control == "context",
        )
        prediction = np.maximum(0.0, internal + config["external_scale"] * aligned)
        fold_rows = ci.score(state, prediction, folds, salt)
        rank_loss = cj.pairwise_rank_loss(state["val_log"], prediction, state["baseline"])
        rows.extend({**row, "seed": state["seed"], "pairwise_rank_loss": rank_loss} for row in fold_rows)
        anchor_counts.append(anchors)
    return {
        "config": config, "control": control, "train_only_anchor_counts": anchor_counts,
        "delta_correlation_mean": float(np.mean([row["delta_correlation"] for row in rows])),
        "de_direction_mean": float(np.mean([row["de_direction"] for row in rows])),
        "pairwise_rank_loss_mean": float(np.mean([row["pairwise_rank_loss"] for row in rows])),
        "fold_runs": rows,
    }


def run(args):
    genes = read_gene_names(args.gene_names)
    train = read_condition_targets(args.train_targets)
    validation = read_condition_targets(args.validation_targets)
    test = read_condition_targets(args.test_targets)
    seeds = [int(value) for value in args.seeds.split(",")]
    symbol_to_gene_id = ci.read_symbol_to_gene_id(args.train_h5ad, genes)
    embedding, mapped = ci.load_embedding_matrix(genes, symbol_to_gene_id, str(args.gene_embedding_npz))
    runs = [ci.train_predict(args, seed, "pca", True, genes, train, validation, test,
                             embedding, float(mapped.mean())) for seed in seeds]
    all_targets = sorted({ci.CONTROL, *train, *validation, *test})
    feature_maps = []
    for state in runs:
        values, _ = ci.go_bp_ic_features(all_targets, state["train_targets"], str(args.go_node_table),
                                         str(args.go_edge_table), args.go_feature_dim)
        feature_maps.append({target: values[index] for index, target in enumerate(all_targets)})
    artifact = np.load(args.external_artifact)
    internal = {"basis_rank": 32, "ridge": 1.0, "de_strength": 0.0,
                "confidence_power": 2.0, "scale": 1.5, "calibration": 0.75}
    configs = [
        {"internal": internal, "external_rank": er, "external_ridge": 10.0,
         "context_rank": cr, "context_ridge": ridge, "external_scale": scale,
         "confidence_power": power, "minimum_anchors": 20}
        for er in [16, 32] for cr in [8, 16] for ridge in [0.1, 1.0, 10.0]
        for scale in [0.1, 0.25, 0.5] for power in [1.0, 2.0]
    ]
    results = [evaluate(config, runs, feature_maps, artifact, args.folds, args.fold_salt) for config in configs]
    results.sort(key=lambda row: (row["delta_correlation_mean"], -row["pairwise_rank_loss_mean"]), reverse=True)
    best = results[0]
    feature_control = evaluate(best["config"], runs, feature_maps, artifact, args.folds, args.fold_salt, "features")
    context_control = evaluate(best["config"], runs, feature_maps, artifact, args.folds, args.fold_salt, "context")
    promoted = (best["delta_correlation_mean"] > args.promotion_reference
                and best["delta_correlation_mean"] > feature_control["delta_correlation_mean"]
                and best["delta_correlation_mean"] > context_control["delta_correlation_mean"])
    report = {"protocol_id": PROTOCOL_ID, "status": "promoted" if promoted else "rejected",
              "test_expression_used": False, "official_test_prediction_generated": False,
              "external_artifact_sha256": args.external_sha256, "best": best,
              "shuffled_feature_control": feature_control, "shuffled_context_control": context_control,
              "promotion_reference": args.promotion_reference, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(cj.parse_args()), indent=2, sort_keys=True))
