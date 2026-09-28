from __future__ import annotations

import hashlib
import json

import numpy as np

import official_h1_candidate_i_pathway_basis as ci
import official_h1_candidate_j_external_basis as cj
from official_h1_contract import read_condition_targets, read_gene_names

PROTOCOL_ID = "vcc-h1-candidate-n-multicontext-nested-gate-v1"


def _ridge(x, y, strength):
    design = np.c_[np.ones(len(x), dtype=np.float32), x]
    penalty = strength * np.eye(design.shape[1], dtype=np.float32)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _randomized_vt(matrix, rank, seed, power_iterations=2, oversample=8):
    width = min(rank + oversample, min(matrix.shape))
    omega = np.random.default_rng(seed).standard_normal((matrix.shape[1], width), dtype=np.float32)
    projected = matrix @ omega
    for _ in range(power_iterations):
        projected = matrix @ (matrix.T @ projected)
    q, _ = np.linalg.qr(projected, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
    return vt[:rank]


def fit_context_model(run, artifact, config, control="none"):
    cache = run.setdefault("candidate_n_fit_cache", {})
    cache_key = (config["response_rank"], config["decoder_rank"], config["encoder_ridge"],
                 config["adapter_ridge"], config["minimum_anchors"], control)
    if cache_key in cache:
        return cache[cache_key]
    delta = artifact["delta"].astype(np.float32)
    targets = artifact["targets"].astype(str)
    contexts = artifact["contexts"].astype(str)
    gene_index = {gene: index for index, gene in enumerate(run["genes"])}
    usable = np.asarray([target in gene_index for target in targets])
    delta, targets, contexts = delta[usable], targets[usable], contexts[usable]
    features = run["embedding_matrix"][[gene_index[target] for target in targets]]
    rng = np.random.default_rng(run["seed"] + 7907)
    if control == "targets":
        features = features[rng.permutation(len(features))]
    context_names = sorted(set(contexts))
    normalized = delta.copy()
    row_weights = np.ones(len(delta), dtype=np.float32)
    for context in context_names:
        selected = contexts == context
        norms = np.linalg.norm(normalized[selected], axis=1)
        scale = max(float(np.median(norms[norms > 0])), 1e-6)
        normalized[selected] /= scale
        row_weights[selected] = np.sqrt(len(delta) / (len(context_names) * int(selected.sum())))
    decomposition = run.setdefault("candidate_n_decomposition_cache", {})
    if "external" not in decomposition:
        reliability = np.sqrt(np.maximum(1e-6, np.mean(np.abs(normalized), axis=0)))
        weighted = normalized * reliability[None, :] * row_weights[:, None]
        vt = _randomized_vt(weighted, 32, run["seed"] + 8101)
        decomposition["external"] = (vt, reliability)
    vt, reliability = decomposition["external"]
    basis = vt[:config["response_rank"]] / reliability[None, :]
    coefficients = normalized @ basis.T
    mappings = {}
    for context in context_names:
        selected = contexts == context
        mappings[context] = _ridge(features[selected], coefficients[selected], config["encoder_ridge"])
    anchors = sorted(set(run["train_targets"]) & set(targets))
    if len(anchors) < config["minimum_anchors"]:
        raise ValueError(f"insufficient shared train-only anchors: {len(anchors)}")
    h1_delta = ci.target_matrix(run, anchors)
    if "h1" not in decomposition:
        _, _, decomposition["h1"] = np.linalg.svd(h1_delta, full_matrices=False)
    h1_vt = decomposition["h1"]
    h1_basis = h1_vt[:config["decoder_rank"]]
    anchor_features = run["embedding_matrix"][[gene_index[target] for target in anchors]]
    context_predictions = []
    for context in context_names:
        context_predictions.append(np.c_[np.ones(len(anchors)), anchor_features] @ mappings[context])
    shared_coeff = np.mean(context_predictions, axis=0)
    h1_coeff = h1_delta @ h1_basis.T
    if control == "contexts":
        h1_coeff = h1_coeff[rng.permutation(len(h1_coeff))]
    adapter = _ridge(shared_coeff, h1_coeff, config["adapter_ridge"])
    train_features = run["embedding_matrix"][[gene_index[target] for target in run["train_targets"]]]
    cache[cache_key] = (basis, mappings, adapter, h1_basis, train_features, anchors)
    return cache[cache_key]


def predict(run, artifact, config, control="none"):
    _, mappings, adapter, h1_basis, train_features, anchors = fit_context_model(run, artifact, config, control)
    gene_index = {gene: index for index, gene in enumerate(run["genes"])}
    rows = []
    for target in run["val"].target_names_per_row:
        if target not in gene_index:
            rows.append(np.zeros(len(run["genes"]), dtype=np.float32)); continue
        feature = run["embedding_matrix"][gene_index[target]]
        context_coeff = [np.r_[1.0, feature] @ mapping for mapping in mappings.values()]
        shared = np.mean(context_coeff, axis=0)
        decoded = np.r_[1.0, shared] @ adapter @ h1_basis
        confidence = max(0.0, float(np.max(train_features @ feature))) ** config["confidence_power"]
        rows.append(confidence * decoded)
    return np.asarray(rows, dtype=np.float32), len(anchors)


def fold_id(target, folds, salt):
    return int.from_bytes(hashlib.sha256(f"{salt}:{target}".encode()).digest()[:4], "little") % folds


def evaluate_config(config, runs, feature_maps, artifact, folds, salt, control="none"):
    by_fold = []
    for state, feature_map in zip(runs, feature_maps):
        internal = ci.predict(state, feature_map, config["internal"])
        external, anchors = predict(state, artifact, config, control)
        prediction = np.maximum(0.0, internal + config["external_scale"] * external)
        scored = ci.score(state, prediction, folds, salt)
        rank_loss = cj.pairwise_rank_loss(state["val_log"], prediction, state["baseline"])
        by_fold.extend({**row, "seed": state["seed"], "rank_loss": rank_loss, "anchors": anchors} for row in scored)
    return by_fold


def mean_score(rows):
    return float(np.mean([row["delta_correlation"] for row in rows]))


def nested_select(configs, runs, feature_maps, artifact, folds, salt, control="none"):
    evaluated = [evaluate_config(config, runs, feature_maps, artifact, folds, salt, control) for config in configs]
    outer = []
    for fold in range(folds):
        candidates = []
        for config, rows in zip(configs, evaluated):
            inner = [row for row in rows if row["fold"] != fold]
            candidates.append((mean_score(inner) - config["rank_weight"] * np.mean([r["rank_loss"] for r in inner]), config, rows))
        _, selected, rows = max(candidates, key=lambda item: item[0])
        heldout = [row for row in rows if row["fold"] == fold]
        outer.append({"fold": fold, "selected_config": selected, "heldout_runs": heldout,
                      "delta_correlation": mean_score(heldout)})
    return {"control": control, "outer_folds": outer,
            "delta_correlation_mean": float(np.mean([row["delta_correlation"] for row in outer]))}


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
    if len(set(artifact["contexts"].astype(str))) < 2:
        raise ValueError("Candidate N requires at least two external contexts")
    internal = {"basis_rank": 32, "ridge": 1.0, "de_strength": 0.0,
                "confidence_power": 2.0, "scale": 1.5, "calibration": 0.75}
    configs = [{"internal": internal, "response_rank": rr, "decoder_rank": dr,
                "encoder_ridge": er, "adapter_ridge": ar, "external_scale": scale,
                "confidence_power": power, "minimum_anchors": 20, "rank_weight": 0.02}
               for rr in (16, 32) for dr in (8, 16) for er in (1.0, 10.0)
               for ar in (1.0, 10.0) for scale in (0.1, 0.25) for power in (1.0, 2.0)]
    best = nested_select(configs, runs, feature_maps, artifact, args.folds, args.fold_salt)
    target_control = nested_select(configs, runs, feature_maps, artifact, args.folds, args.fold_salt, "targets")
    context_control = nested_select(configs, runs, feature_maps, artifact, args.folds, args.fold_salt, "contexts")
    promoted = (best["delta_correlation_mean"] > args.promotion_reference
                and best["delta_correlation_mean"] > target_control["delta_correlation_mean"]
                and best["delta_correlation_mean"] > context_control["delta_correlation_mean"])
    report = {"protocol_id": PROTOCOL_ID, "status": "promoted" if promoted else "rejected",
              "test_expression_used": False, "official_test_prediction_generated": False,
              "external_artifact_sha256": args.external_sha256, "promotion_reference": args.promotion_reference,
              "nested_multifold": best, "shuffled_target_control": target_control,
              "shuffled_context_control": context_control}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(cj.parse_args()), indent=2, sort_keys=True))
