from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from official_h1_candidate_i_pathway_basis import predict as predict_internal
from official_h1_contract import read_condition_targets, read_gene_names

PROTOCOL_ID = "vcc-h1-candidate-j-external-mixture-v1"


def external_model(run, artifact, rank, ridge, shuffle=False):
    cache = run.setdefault("external_model_cache", {})
    key = (rank, ridge, shuffle)
    if key in cache:
        return cache[key]
    delta = artifact["delta"].astype(np.float32)
    targets = artifact["targets"].astype(str).tolist()
    gi = {gene: index for index, gene in enumerate(run["genes"])}
    usable = [index for index, target in enumerate(targets) if target in gi]
    features = run["embedding_matrix"][[gi[targets[index]] for index in usable]]
    if shuffle:
        features = features[np.random.default_rng(run["seed"] + 2903).permutation(len(features))]
    _, _, vt = np.linalg.svd(delta[usable], full_matrices=False)
    basis = vt[:rank]
    coefficients = delta[usable] @ basis.T
    design = np.c_[np.ones(len(features), dtype=np.float32), features]
    penalty = ridge * np.eye(design.shape[1], dtype=np.float32); penalty[0, 0] = 0
    mapping = np.linalg.solve(design.T @ design + penalty, design.T @ coefficients)
    cache[key] = (basis, mapping, features)
    return cache[key]


def predict_external(run, artifact, config, shuffle=False):
    basis, mapping, external_features = external_model(
        run, artifact, config["external_rank"], config["external_ridge"], shuffle
    )
    gi = {gene: index for index, gene in enumerate(run["genes"])}
    cache = {}
    rows = []
    for target in run["val"].target_names_per_row:
        if target not in cache:
            if target not in gi:
                cache[target] = np.zeros(len(run["genes"]), dtype=np.float32)
            else:
                feature = run["embedding_matrix"][gi[target]]
                confidence = max(0.0, float(np.max(external_features @ feature))) ** config["confidence_power"]
                delta = np.r_[1.0, feature] @ mapping @ basis
                cache[target] = confidence * delta
        rows.append(cache[target])
    return np.asarray(rows, dtype=np.float32)


def pairwise_rank_loss(truth, prediction, baseline, top_fraction=0.05):
    truth_delta = np.mean(truth - baseline, axis=0)
    pred_delta = np.mean(prediction - baseline, axis=0)
    count = max(2, int(len(truth_delta) * top_fraction))
    selected = np.argsort(np.abs(truth_delta))[-count:]
    order = np.argsort(truth_delta[selected])
    values = pred_delta[selected][order]
    return float(np.maximum(0.0, 0.01 - np.diff(values)).mean())


def evaluate(config, runs, feature_maps, artifact, folds, salt, shuffle=False):
    rows = []
    from official_h1_candidate_i_pathway_basis import score
    for run, feature_map in zip(runs, feature_maps):
        internal = predict_internal(run, feature_map, config["internal"])
        external_delta = predict_external(run, artifact, config, shuffle)
        prediction = np.maximum(0, internal + config["external_scale"] * external_delta)
        fold_rows = score(run, prediction, folds, salt)
        rank_loss = pairwise_rank_loss(run["val_log"], prediction, run["baseline"])
        rows.extend({**row, "seed": run["seed"], "pairwise_rank_loss": rank_loss} for row in fold_rows)
    return {
        "config": config, "shuffled_external": shuffle,
        "delta_correlation_mean": float(np.mean([r["delta_correlation"] for r in rows])),
        "de_direction_mean": float(np.mean([r["de_direction"] for r in rows])),
        "pairwise_rank_loss_mean": float(np.mean([r["pairwise_rank_loss"] for r in rows])),
        "fold_runs": rows,
    }


def run(args):
    # Reuse Candidate I's fixed training and feature construction; its report is not promoted here.
    import official_h1_candidate_i_pathway_basis as ci
    genes = read_gene_names(args.gene_names)
    train = read_condition_targets(args.train_targets); val = read_condition_targets(args.validation_targets)
    test = read_condition_targets(args.test_targets)
    seeds = [int(x) for x in args.seeds.split(",")]
    symbol_to_gene_id = ci.read_symbol_to_gene_id(args.train_h5ad, genes)
    embedding, mapped = ci.load_embedding_matrix(genes, symbol_to_gene_id, str(args.gene_embedding_npz))
    runs = [ci.train_predict(args, seed, "pca", True, genes, train, val, test, embedding, float(mapped.mean())) for seed in seeds]
    all_targets = sorted({ci.CONTROL, *train, *val, *test})
    feature_maps=[]
    for state in runs:
        values,_=ci.go_bp_ic_features(all_targets,state["train_targets"],str(args.go_node_table),str(args.go_edge_table),args.go_feature_dim)
        feature_maps.append({target:values[i] for i,target in enumerate(all_targets)})
    artifact=np.load(args.external_artifact)
    internal={"basis_rank":32,"ridge":1.0,"de_strength":0.0,"confidence_power":2.0,"scale":1.5,"calibration":0.75}
    configs=[{"internal":internal,"external_rank":rank,"external_ridge":ridge,"external_scale":scale,"confidence_power":power}
             for rank in [16,32] for ridge in [0.1,1.0,10.0] for scale in [0.1,0.25,0.5] for power in [1.0,2.0]]
    results=[evaluate(c,runs,feature_maps,artifact,args.folds,args.fold_salt) for c in configs]
    results.sort(key=lambda x:(x["delta_correlation_mean"],-x["pairwise_rank_loss_mean"]),reverse=True)
    best=results[0]; shuffled=evaluate(best["config"],runs,feature_maps,artifact,args.folds,args.fold_salt,True)
    promoted=best["delta_correlation_mean"]>args.promotion_reference and best["delta_correlation_mean"]>shuffled["delta_correlation_mean"]
    report={"protocol_id":PROTOCOL_ID,"status":"promoted" if promoted else "rejected","test_expression_used":False,
            "official_test_prediction_generated":False,"external_artifact_sha256":args.external_sha256,"best":best,
            "shuffled_external_control":shuffled,"promotion_reference":args.promotion_reference,"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


def parse_args():
    import official_h1_candidate_i_pathway_basis as ci
    p=argparse.ArgumentParser()
    p.add_argument("--train-h5ad",type=Path,default=ci.DEFAULT_ASSETS/"train/adata_Training.h5ad")
    p.add_argument("--validation-h5ad",type=Path,default=ci.DEFAULT_ASSETS/"validation/adata_Validation.h5ad")
    p.add_argument("--train-targets",type=Path,default=ci.DEFAULT_ASSETS/"train/pert_counts_Training.csv")
    p.add_argument("--validation-targets",type=Path,default=ci.DEFAULT_ASSETS/"validation/pert_counts_Validation.csv")
    p.add_argument("--test-targets",type=Path,default=ci.DEFAULT_ASSETS/"test/pert_counts_Test.csv")
    p.add_argument("--gene-names",type=Path,default=ci.DEFAULT_ASSETS/"gene_names.csv")
    p.add_argument("--gene-embedding-npz",type=Path,default=ci.DEFAULT_ROOT/"assets/lingshu_hf_b77f980/gene_embeddings.npz")
    annotations=Path(__file__).resolve().parent/"resources/official-h1-go-bp-v1"
    p.add_argument("--go-node-table",type=Path,default=annotations/"node_table.csv")
    p.add_argument("--go-edge-table",type=Path,default=annotations/"edge_table.csv")
    p.add_argument("--external-artifact",type=Path,required=True); p.add_argument("--external-sha256",required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--seeds",default="20260907,20260908,20260909")
    p.add_argument("--folds",type=int,default=3); p.add_argument("--fold-salt",default="candidate-j-external-folds-20260913")
    p.add_argument("--promotion-reference",type=float,default=0.608694)
    p.add_argument("--n-train-targets",type=int,default=150); p.add_argument("--n-val-targets",type=int,default=50)
    p.add_argument("--max-rows-per-target",type=int,default=32); p.add_argument("--max-steps",type=int,default=2048)
    p.add_argument("--rank",type=int,default=24); p.add_argument("--feature-dim",type=int,default=128)
    p.add_argument("--go-feature-dim",type=int,default=256); p.add_argument("--embedding-projection-seed",type=int,default=20260908)
    p.add_argument("--batch-size",type=int,default=128); p.add_argument("--learning-rate",type=float,default=8e-4)
    p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--direction-weight",type=float,default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()),indent=2,sort_keys=True))
