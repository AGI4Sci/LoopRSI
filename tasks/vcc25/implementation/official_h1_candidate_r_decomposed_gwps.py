from __future__ import annotations

import json

import numpy as np

import official_h1_candidate_i_pathway_basis as ci
import official_h1_candidate_j_external_basis as cj
import official_h1_candidate_o_target_complete as candidate_o


PROTOCOL_ID = "vcc-h1-candidate-r-decomposed-gwps-transport-v1"


def _fit_decomposed_transport(run, by_context, ridge, control):
    cache = run.setdefault("candidate_r_transport_cache", {})
    key = (ridge, control)
    if key in cache:
        return cache[key]
    rng = np.random.default_rng(run["seed"] + 3203)
    models = {}
    for context, rows in by_context.items():
        anchors = sorted(set(run["train_targets"]) & set(rows))
        if len(anchors) < 20:
            continue
        external = np.asarray([rows[target] for target in anchors], dtype=np.float32)
        h1 = ci.target_matrix(run, anchors)
        if control == "context":
            h1 = h1[rng.permutation(len(h1))]
        external_template = external.mean(axis=0)
        h1_template = h1.mean(axis=0)
        external_residual = external - external_template
        h1_residual = h1 - h1_template
        numerator = np.sum(external_residual * h1_residual, axis=0)
        denominator = np.sum(external_residual * external_residual, axis=0) + ridge
        diagonal = np.clip(numerator / denominator, -4.0, 4.0).astype(np.float32)
        models[context] = {
            "diagonal": diagonal,
            "external_template": external_template.astype(np.float32),
            "h1_template": h1_template.astype(np.float32),
            "anchors": anchors,
        }
    if not models:
        raise ValueError("no external context has at least 20 H1 train anchors")
    cache[key] = models
    return models


def _decomposed_target_priors(run, by_context, config, control="none"):
    models = _fit_decomposed_transport(run, by_context, config["transport_ridge"], control)
    targets = sorted(set(run["val"].target_names_per_row) - {ci.CONTROL})
    rng = np.random.default_rng(run["seed"] + 4211)
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
            residual = external - model["external_template"]
            transported = model["h1_template"] + residual * model["diagonal"]
            reliability = np.abs(residual)
            threshold = np.quantile(reliability[reliability > 0], config["deg_quantile"]) if np.any(reliability > 0) else 0.0
            deg_weight = np.where(reliability >= threshold, config["deg_boost"], 1.0)
            predictions.append(transported * deg_weight)
        if predictions:
            coverage += 1
            priors[target] = np.mean(predictions, axis=0).astype(np.float32)
    return priors, coverage / max(len(targets), 1), {name: len(model["anchors"]) for name, model in models.items()}


if __name__ == "__main__":
    candidate_o.PROTOCOL_ID = PROTOCOL_ID
    candidate_o._target_priors = _decomposed_target_priors
    print(json.dumps(candidate_o.run(cj.parse_args()), indent=2, sort_keys=True))
