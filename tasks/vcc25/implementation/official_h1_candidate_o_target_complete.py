from __future__ import annotations

import json

import numpy as np

import official_h1_candidate_i_pathway_basis as ci
import official_h1_candidate_j_external_basis as cj
import official_h1_candidate_n_multicontext as cn
from official_h1_contract import read_condition_targets, read_gene_names

PROTOCOL_ID = "vcc-h1-candidate-o-target-complete-nested-gate-v1"


def _artifact_index(artifact):
    targets = artifact["targets"].astype(str)
    contexts = artifact["contexts"].astype(str)
    delta = artifact["delta"].astype(np.float32)
    return {
        context: {target: delta[index] for index, target in enumerate(targets) if contexts[index] == context}
        for context in sorted(set(contexts))
    }


def _fit_transport(run, by_context, ridge, control):
    cache = run.setdefault("candidate_o_transport_cache", {})
    key = (ridge, control)
    if key in cache:
        return cache[key]
    rng = np.random.default_rng(run["seed"] + 1409)
    models = {}
    for context, rows in by_context.items():
        anchors = sorted(set(run["train_targets"]) & set(rows))
        if len(anchors) < 20:
            continue
        external = np.asarray([rows[target] for target in anchors], dtype=np.float32)
        h1 = ci.target_matrix(run, anchors)
        if control == "context":
            h1 = h1[rng.permutation(len(h1))]
        numerator = np.sum(external * h1, axis=0)
        denominator = np.sum(external * external, axis=0) + ridge
        diagonal = np.clip(numerator / denominator, -4.0, 4.0).astype(np.float32)
        external_norm = np.linalg.norm(external, axis=1)
        h1_norm = np.linalg.norm(h1, axis=1)
        amplitude = float(np.median(h1_norm / np.maximum(external_norm, 1e-6)))
        models[context] = {"diagonal": diagonal, "amplitude": amplitude, "anchors": anchors}
    if not models:
        raise ValueError("no external context has at least 20 H1 train anchors")
    cache[key] = models
    return models


def _target_priors(run, by_context, config, control="none"):
    models = _fit_transport(run, by_context, config["transport_ridge"], control)
    targets = sorted(set(run["val"].target_names_per_row) - {ci.CONTROL})
    rng = np.random.default_rng(run["seed"] + 2417)
    source_targets = targets.copy()
    if control == "targets":
        source_targets = list(np.asarray(source_targets)[rng.permutation(len(source_targets))])
    priors = {}
    coverage = 0
    for target, source_target in zip(targets, source_targets):
        predictions = []
        for context, model in models.items():
            external = by_context[context].get(source_target)
            if external is None:
                continue
            transported = external * model["diagonal"]
            reliability = np.abs(external)
            threshold = np.quantile(reliability[reliability > 0], config["deg_quantile"]) if np.any(reliability > 0) else 0.0
            deg_weight = np.where(reliability >= threshold, config["deg_boost"], 1.0)
            predictions.append(transported * deg_weight)
        if predictions:
            coverage += 1
            priors[target] = np.mean(predictions, axis=0).astype(np.float32)
    return priors, coverage / max(len(targets), 1), {name: len(model["anchors"]) for name, model in models.items()}


def predict(run, feature_map, by_context, config, control="none"):
    internal = ci.predict(run, feature_map, config["internal"])
    priors, coverage, anchors = _target_priors(run, by_context, config, control)
    rows = []
    zero = np.zeros(len(run["genes"]), dtype=np.float32)
    for index, target in enumerate(run["val"].target_names_per_row):
        prior = priors.get(target, zero)
        external_prediction = np.maximum(0.0, run["baseline"][index] + config["external_scale"] * prior)
        rows.append(config["external_weight"] * external_prediction + (1.0 - config["external_weight"]) * internal[index])
    return np.asarray(rows, dtype=np.float32), coverage, anchors


def evaluate_config(config, runs, feature_maps, by_context, folds, salt, control="none"):
    rows, coverages, anchor_counts = [], [], []
    for state, feature_map in zip(runs, feature_maps):
        prediction, coverage, anchors = predict(state, feature_map, by_context, config, control)
        scored = ci.score(state, prediction, folds, salt)
        rank_loss = cj.pairwise_rank_loss(state["val_log"], prediction, state["baseline"])
        rows.extend({**row, "seed": state["seed"], "rank_loss": rank_loss} for row in scored)
        coverages.append(coverage)
        anchor_counts.append(anchors)
    return rows, coverages, anchor_counts


def nested_select(configs, runs, feature_maps, by_context, folds, salt, control="none"):
    evaluated = [evaluate_config(config, runs, feature_maps, by_context, folds, salt, control) for config in configs]
    outer = []
    for fold in range(folds):
        candidates = []
        for config, (rows, coverage, anchors) in zip(configs, evaluated):
            inner = [row for row in rows if row["fold"] != fold]
            objective = np.mean([row["delta_correlation"] for row in inner]) - config["rank_weight"] * np.mean([row["rank_loss"] for row in inner])
            candidates.append((objective, config, rows, coverage, anchors))
        _, selected, rows, coverage, anchors = max(candidates, key=lambda item: item[0])
        heldout = [row for row in rows if row["fold"] == fold]
        outer.append({"fold": fold, "selected_config": selected, "heldout_runs": heldout,
                      "delta_correlation": float(np.mean([row["delta_correlation"] for row in heldout])),
                      "target_coverage": float(np.mean(coverage)), "anchor_counts": anchors})
    return {"control": control, "outer_folds": outer,
            "delta_correlation_mean": float(np.mean([row["delta_correlation"] for row in outer])),
            "target_coverage_mean": float(np.mean([row["target_coverage"] for row in outer]))}


def run(args):
    genes = read_gene_names(args.gene_names)
    train = read_condition_targets(args.train_targets)
    validation = read_condition_targets(args.validation_targets)
    test = read_condition_targets(args.test_targets)
    seeds = [int(value) for value in args.seeds.split(",")]
    symbol_to_gene_id = ci.read_symbol_to_gene_id(args.train_h5ad, genes)
    embedding, mapped = ci.load_embedding_matrix(genes, symbol_to_gene_id, str(args.gene_embedding_npz))
    runs = [ci.train_predict(args, seed, "pca", True, genes, train, validation, test, embedding, float(mapped.mean())) for seed in seeds]
    all_targets = sorted({ci.CONTROL, *train, *validation, *test})
    feature_maps = []
    for state in runs:
        values, _ = ci.go_bp_ic_features(all_targets, state["train_targets"], str(args.go_node_table), str(args.go_edge_table), args.go_feature_dim)
        feature_maps.append({target: values[index] for index, target in enumerate(all_targets)})
    artifact = np.load(args.external_artifact)
    by_context = _artifact_index(artifact)
    internal = {"basis_rank": 32, "ridge": 1.0, "de_strength": 0.0, "confidence_power": 2.0,
                "scale": 1.5, "calibration": 0.75}
    configs = [{"internal": internal, "transport_ridge": ridge, "external_scale": scale,
                "external_weight": weight, "deg_quantile": quantile, "deg_boost": boost,
                "rank_weight": 0.02}
               for ridge in (0.01, 0.1, 1.0) for scale in (0.5, 1.0, 1.5)
               for weight in (0.25, 0.5, 0.75, 1.0) for quantile in (0.9, 0.95)
               for boost in (1.0, 1.5)]
    best = nested_select(configs, runs, feature_maps, by_context, args.folds, args.fold_salt)
    target_control = nested_select(configs, runs, feature_maps, by_context, args.folds, args.fold_salt, "targets")
    context_control = nested_select(configs, runs, feature_maps, by_context, args.folds, args.fold_salt, "context")
    promoted = (best["delta_correlation_mean"] > args.promotion_reference
                and best["delta_correlation_mean"] > target_control["delta_correlation_mean"]
                and best["delta_correlation_mean"] > context_control["delta_correlation_mean"])
    report = {"protocol_id": PROTOCOL_ID, "status": "promoted" if promoted else "rejected",
              "information_policy": "vcc_official", "same_target_external_context_allowed": True,
              "h1_validation_expression_used_for_training": False, "h1_test_expression_used": False,
              "official_test_prediction_generated": False, "external_artifact_sha256": args.external_sha256,
              "promotion_reference": args.promotion_reference, "nested_multifold": best,
              "shuffled_target_control": target_control, "shuffled_context_control": context_control}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(cj.parse_args()), indent=2, sort_keys=True))
