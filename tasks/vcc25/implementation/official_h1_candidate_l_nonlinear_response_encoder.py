from __future__ import annotations

import json

import numpy as np

import official_h1_candidate_i_pathway_basis as ci
import official_h1_candidate_j_external_basis as cj
from official_h1_contract import read_condition_targets, read_gene_names

PROTOCOL_ID = "vcc-h1-candidate-l-nonlinear-response-encoder-v1"


def random_fourier(features, weights, phase):
    return np.sqrt(2.0 / weights.shape[1]) * np.cos(features @ weights + phase)


def fit_model(run, artifact, config, control="none"):
    delta = artifact["delta"].astype(np.float32)
    targets = artifact["targets"].astype(str).tolist()
    gene_index = {gene: index for index, gene in enumerate(run["genes"])}
    usable = [index for index, target in enumerate(targets) if target in gene_index]
    features = run["embedding_matrix"][[gene_index[targets[index]] for index in usable]]
    _, _, ext_vt = np.linalg.svd(delta[usable], full_matrices=False)
    ext_basis = ext_vt[:config["external_rank"]]
    ext_coeff = delta[usable] @ ext_basis.T
    rng = np.random.default_rng(config["encoder_seed"])
    weights = rng.normal(0.0, config["kernel_scale"], (features.shape[1], config["encoder_width"])).astype(np.float32)
    phase = rng.uniform(0.0, 2.0 * np.pi, config["encoder_width"]).astype(np.float32)
    encoded = random_fourier(features, weights, phase)
    if control == "features":
        encoded = encoded[rng.permutation(len(encoded))]
    design = np.c_[np.ones(len(encoded), dtype=np.float32), encoded]
    penalty = config["encoder_ridge"] * np.eye(design.shape[1], dtype=np.float32); penalty[0, 0] = 0.0
    response_mapping = np.linalg.solve(design.T @ design + penalty, design.T @ ext_coeff)

    ext_index = {target: index for index, target in enumerate(targets)}
    anchors = sorted(set(run["train_targets"]) & set(targets))
    h1_delta = ci.target_matrix(run, run["train_targets"])
    _, _, h1_vt = np.linalg.svd(h1_delta, full_matrices=False)
    h1_basis = h1_vt[:config["context_rank"]]
    anchor_ext = delta[[ext_index[target] for target in anchors]] @ ext_basis.T
    anchor_h1 = np.asarray([run["target_delta"][target] for target in anchors]) @ h1_basis.T
    if control == "context":
        anchor_h1 = anchor_h1[np.random.default_rng(run["seed"] + 4909).permutation(len(anchor_h1))]
    anchor_design = np.c_[np.ones(len(anchor_ext), dtype=np.float32), anchor_ext]
    context_penalty = config["context_ridge"] * np.eye(anchor_design.shape[1], dtype=np.float32); context_penalty[0, 0] = 0.0
    context_mapping = np.linalg.solve(anchor_design.T @ anchor_design + context_penalty, anchor_design.T @ anchor_h1)
    return weights, phase, response_mapping, context_mapping, h1_basis, features, len(anchors)


def predict(run, artifact, config, control="none"):
    weights, phase, response_mapping, context_mapping, h1_basis, external_features, anchors = fit_model(run, artifact, config, control)
    gene_index = {gene: index for index, gene in enumerate(run["genes"])}
    cache, rows = {}, []
    for target in run["val"].target_names_per_row:
        if target not in cache:
            if target not in gene_index:
                cache[target] = np.zeros(len(run["genes"]), dtype=np.float32)
            else:
                feature = run["embedding_matrix"][gene_index[target]]
                confidence = max(0.0, float(np.max(external_features @ feature)))
                encoded = random_fourier(feature[None, :], weights, phase)[0]
                external_coeff = np.r_[1.0, encoded] @ response_mapping
                h1_coeff = np.r_[1.0, external_coeff] @ context_mapping
                cache[target] = confidence * (h1_coeff @ h1_basis)
        rows.append(cache[target])
    return np.asarray(rows, dtype=np.float32), anchors


def evaluate(config, runs, feature_maps, artifact, folds, salt, control="none"):
    rows, anchors = [], []
    for state, feature_map in zip(runs, feature_maps):
        internal = ci.predict(state, feature_map, config["internal"])
        external, count = predict(state, artifact, config, control)
        prediction = np.maximum(0.0, internal + config["external_scale"] * external)
        rank_loss = cj.pairwise_rank_loss(state["val_log"], prediction, state["baseline"])
        rows.extend({**row, "seed": state["seed"], "pairwise_rank_loss": rank_loss}
                    for row in ci.score(state, prediction, folds, salt))
        anchors.append(count)
    return {"control": control, "config": config, "train_only_anchor_counts": anchors,
            "delta_correlation_mean": float(np.mean([row["delta_correlation"] for row in rows])),
            "de_direction_mean": float(np.mean([row["de_direction"] for row in rows])),
            "pairwise_rank_loss_mean": float(np.mean([row["pairwise_rank_loss"] for row in rows])),
            "fold_runs": rows}


def run(args):
    genes = read_gene_names(args.gene_names)
    train = read_condition_targets(args.train_targets); validation = read_condition_targets(args.validation_targets)
    test = read_condition_targets(args.test_targets); seeds = [int(value) for value in args.seeds.split(",")]
    symbol_to_gene_id = ci.read_symbol_to_gene_id(args.train_h5ad, genes)
    embedding, mapped = ci.load_embedding_matrix(genes, symbol_to_gene_id, str(args.gene_embedding_npz))
    runs = [ci.train_predict(args, seed, "pca", True, genes, train, validation, test,
                             embedding, float(mapped.mean())) for seed in seeds]
    all_targets = sorted({ci.CONTROL, *train, *validation, *test}); feature_maps = []
    for state in runs:
        values, _ = ci.go_bp_ic_features(all_targets, state["train_targets"], str(args.go_node_table),
                                         str(args.go_edge_table), args.go_feature_dim)
        feature_maps.append({target: values[index] for index, target in enumerate(all_targets)})
    artifact = np.load(args.external_artifact)
    config = {"internal": {"basis_rank": 32, "ridge": 1.0, "de_strength": 0.0,
                           "confidence_power": 2.0, "scale": 1.5, "calibration": 0.75},
              "external_rank": 32, "context_rank": 8, "context_ridge": 10.0,
              "external_scale": 0.5, "encoder_width": 256, "encoder_ridge": 10.0,
              "kernel_scale": 2.0, "encoder_seed": 20260913}
    primary = evaluate(config, runs, feature_maps, artifact, args.folds, args.fold_salt)
    feature_control = evaluate(config, runs, feature_maps, artifact, args.folds, args.fold_salt, "features")
    context_control = evaluate(config, runs, feature_maps, artifact, args.folds, args.fold_salt, "context")
    promoted = (primary["delta_correlation_mean"] > args.promotion_reference
                and primary["delta_correlation_mean"] > feature_control["delta_correlation_mean"]
                and primary["delta_correlation_mean"] > context_control["delta_correlation_mean"])
    report = {"protocol_id": PROTOCOL_ID, "status": "promoted" if promoted else "rejected",
              "test_expression_used": False, "official_test_prediction_generated": False,
              "external_artifact_sha256": args.external_sha256, "primary": primary,
              "shuffled_feature_control": feature_control, "shuffled_context_control": context_control,
              "promotion_reference": args.promotion_reference}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(cj.parse_args()), indent=2, sort_keys=True))
