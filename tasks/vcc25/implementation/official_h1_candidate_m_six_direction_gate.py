from __future__ import annotations

import json
import numpy as np

import official_h1_candidate_i_pathway_basis as ci
import official_h1_candidate_j_external_basis as cj
import official_h1_candidate_k_context_alignment as ck
import official_h1_candidate_l_nonlinear_response_encoder as cl
from official_h1_contract import read_condition_targets, read_gene_names

PROTOCOL_ID = "vcc-h1-candidate-m-six-direction-gate-v1"


def hierarchical_prediction(state, feature_map, artifact, config, control="none"):
    internal = ci.predict(state, feature_map, config["internal"])
    aligned, anchors = ck.predict_aligned(
        state, artifact, config["context"],
        shuffle_features=control == "features", shuffle_context=control == "context")
    nonlinear, _ = cl.predict(state, artifact, config["nonlinear"], control)
    numerator = np.sum(aligned * nonlinear, axis=1)
    denominator = np.linalg.norm(aligned, axis=1) * np.linalg.norm(nonlinear, axis=1)
    agreement = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    confidence = np.clip((agreement - config["agreement_floor"]) /
                         (1.0 - config["agreement_floor"]), 0.0, 1.0)
    external = config["linear_mix"] * aligned + (1.0 - config["linear_mix"]) * nonlinear
    prediction = np.maximum(0.0, internal + config["external_scale"] * confidence[:, None] * external)
    return prediction, anchors, confidence


def evaluate(config, runs, feature_maps, artifact, folds, salt, control="none"):
    rows, ranks, anchor_counts, confidences = [], [], [], []
    for state, feature_map in zip(runs, feature_maps):
        prediction, anchors, confidence = hierarchical_prediction(state, feature_map, artifact, config, control)
        fold_rows = ci.score(state, prediction, folds, salt)
        rows.extend({**row, "seed": state["seed"]} for row in fold_rows)
        ranks.append(cj.pairwise_rank_loss(state["val_log"], prediction, state["baseline"]))
        anchor_counts.append(anchors); confidences.extend(confidence.tolist())
    delta = float(np.mean([row["delta_correlation"] for row in rows]))
    rank = float(np.mean(ranks))
    return {
        "control": control, "config": config, "fold_runs": rows,
        "delta_correlation_mean": delta,
        "de_direction_mean": float(np.mean([row["de_direction"] for row in rows])),
        "pairwise_rank_loss_mean": rank,
        "selection_score": delta - config["rank_weight"] * rank,
        "train_only_anchor_counts": anchor_counts,
        "confidence_mean": float(np.mean(confidences)),
        "fallback_fraction": float(np.mean(np.asarray(confidences) == 0.0)),
    }


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
    internal = {"basis_rank": 32, "ridge": 1.0, "de_strength": 0.0,
                "confidence_power": 2.0, "scale": 1.5, "calibration": 0.75}
    context = {"internal": internal, "external_rank": 16, "external_ridge": 10.0,
               "context_rank": 8, "context_ridge": 10.0, "external_scale": 0.5,
               "confidence_power": 2.0, "minimum_anchors": 20}
    nonlinear = {"internal": internal, "external_rank": 32, "context_rank": 8,
                 "context_ridge": 10.0, "external_scale": 0.5, "encoder_width": 256,
                 "encoder_ridge": 10.0, "kernel_scale": 2.0, "encoder_seed": 20260913}
    configs = [{"internal": internal, "context": context, "nonlinear": nonlinear,
                "linear_mix": mix, "external_scale": scale, "agreement_floor": floor, "rank_weight": 0.02}
               for mix in (0.25, 0.5, 0.75) for scale in (0.1, 0.25, 0.5) for floor in (0.0, 0.25, 0.5)]
    results = [evaluate(config, runs, feature_maps, artifact, args.folds, args.fold_salt) for config in configs]
    results.sort(key=lambda row: row["selection_score"], reverse=True); best = results[0]
    feature_control = evaluate(best["config"], runs, feature_maps, artifact, args.folds, args.fold_salt, "features")
    context_control = evaluate(best["config"], runs, feature_maps, artifact, args.folds, args.fold_salt, "context")
    promoted = (best["delta_correlation_mean"] > args.promotion_reference
                and best["delta_correlation_mean"] > feature_control["delta_correlation_mean"]
                and best["delta_correlation_mean"] > context_control["delta_correlation_mean"])
    report = {
        "protocol_id": PROTOCOL_ID, "status": "promoted" if promoted else "rejected",
        "test_expression_used": False, "official_test_prediction_generated": False,
        "external_artifact_sha256": args.external_sha256, "promotion_reference": args.promotion_reference,
        "directions": {
            "larger_shared_perturbation_corpus": "blocked_zenodo_504_no_unverified_data_used",
            "train_only_cell_context_transport": "enabled",
            "external_pretraining_h1_calibration": "enabled_fixed_rff_encoder",
            "pathway_gene_coverage_decoder": "enabled_internal_go_basis_full_panel",
            "de_ranking_objective": "enabled_validation_tiebreak_only",
            "hierarchical_confidence_fallback": "enabled_label_free_model_agreement"},
        "best": best, "shuffled_feature_control": feature_control,
        "shuffled_context_control": context_control, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(cj.parse_args()), indent=2, sort_keys=True))
