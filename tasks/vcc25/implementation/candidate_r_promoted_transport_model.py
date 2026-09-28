from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

import candidate_g_promoted_delta_model as candidate_g
import official_h1_candidate_o_target_complete as candidate_o

CANDIDATE_ID = "candidate_r_promoted_decomposed_gwps"
PROTOCOL_ID = "vcc-h1-candidate-r-official-v1"
TRANSPORT_RIDGE = 0.1
EXTERNAL_SCALE = 1.0
EXTERNAL_WEIGHT = 0.5


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def fit_transport(state, artifact_path):
    by_context = candidate_o._artifact_index(np.load(artifact_path))
    models = []
    for context, rows in by_context.items():
        anchors = sorted(set(state["train_targets"]) & set(rows) & set(state["target_delta"]))
        if len(anchors) < 20:
            continue
        external = np.asarray([rows[t] for t in anchors], dtype=np.float32)
        h1 = np.asarray([state["target_delta"][t] for t in anchors], dtype=np.float32)
        external_template, h1_template = external.mean(0), h1.mean(0)
        external_residual, h1_residual = external - external_template, h1 - h1_template
        diagonal = np.sum(external_residual * h1_residual, axis=0)
        diagonal /= np.sum(external_residual ** 2, axis=0) + TRANSPORT_RIDGE
        models.append({"context": context, "rows": rows, "anchors": anchors,
                       "external_template": external_template, "h1_template": h1_template,
                       "diagonal": np.clip(diagonal, -4, 4).astype(np.float32)})
    if not models:
        raise ValueError("quantitative GWPS artifact has fewer than 20 train anchors")
    return models


def external_priors(targets, models, n_genes):
    priors, covered = [], 0
    for target in targets:
        values = []
        for model in models:
            external = model["rows"].get(target)
            if external is not None:
                values.append(model["h1_template"] +
                              (external - model["external_template"]) * model["diagonal"])
        covered += bool(values)
        priors.append(np.mean(values, axis=0) if values else np.zeros(n_genes, dtype=np.float32))
    return np.asarray(priors, dtype=np.float32), covered


def chunk_predictions(state, obs, models, chunk_size):
    batch_index = {b: i for i, b in enumerate(state["batch_order"])}
    global_batch, model = batch_index["__global__"], state["model"]
    model.eval()
    with torch.no_grad():
        for start in range(0, obs["shape"][0], chunk_size):
            end = min(start + chunk_size, obs["shape"][0])
            targets, batches = obs["target_gene"][start:end], obs["batch"][start:end]
            features = np.asarray([state["fmap"][t] for t in targets], dtype=np.float32)
            bids = np.asarray([batch_index.get(b, global_batch) for b in batches], dtype=np.int64)
            base = model(torch.from_numpy(features), torch.from_numpy(bids), None, allow_id=False).numpy()
            neighbor = candidate_g.build_self_neighbor_prior_rows(list(targets), state["genes"], state)
            internal = candidate_g.apply_candidate_f_prior(base.astype(np.float32), neighbor)
            prior, _ = external_priors(targets, models, len(state["genes"]))
            external = np.maximum(0, state["baselines"][bids] + EXTERNAL_SCALE * prior)
            yield start, end, ((1 - EXTERNAL_WEIGHT) * internal + EXTERNAL_WEIGHT * external).astype(np.float32)


def run(args):
    started = time.time()
    state = candidate_g.train_model(args, args.seed)
    obs = candidate_g.read_obs_schema(args.reference_h5ad, state["genes"])
    models = fit_transport(state, args.external_artifact)
    test_targets = sorted(set(obs["target_gene"]) - {candidate_g.CONTROL})
    _, covered = external_priors(test_targets, models, len(state["genes"]))
    metadata = {"candidate_id": CANDIDATE_ID, "protocol_id": PROTOCOL_ID, "seed": args.seed,
                "validation_score": 0.6125796296773004, "promotion_reference": 0.608694,
                "external_artifact_sha256": args.external_sha256,
                "transport_ridge": TRANSPORT_RIDGE, "external_scale": EXTERNAL_SCALE,
                "external_weight": EXTERNAL_WEIGHT,
                "train_anchor_counts": {m["context"]: len(m["anchors"]) for m in models},
                "official_target_coverage": covered / max(len(test_targets), 1),
                "uses_test_expression_for_training_or_selection": False}
    info = candidate_g.write_h5ad(args.output_h5ad, obs, state["genes"],
                                  chunk_predictions(state, obs, models, args.chunk_size), metadata, args.chunk_size)
    contract = candidate_g.validate_prediction_schema(args.output_h5ad, args.reference_h5ad,
                                                       args.condition_csv, args.gene_names, args.expected_targets)
    result = {**metadata, **info, "contract": contract, "runtime_seconds": time.time() - started}
    result["status"] = "ready" if contract["status"] == "ready" and info["prediction_finite"] else "blocked"
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default) + "\n")
    return result


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--external-artifact", type=Path, required=True)
    pre.add_argument("--external-sha256", required=True)
    known, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = candidate_g.parse_args()
    args.external_artifact, args.external_sha256 = known.external_artifact, known.external_sha256
    print(json.dumps(run(args), indent=2, sort_keys=True, default=json_default))
