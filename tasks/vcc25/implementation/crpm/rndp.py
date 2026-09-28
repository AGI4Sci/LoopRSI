"""Regulatory-network diffusion predictor for unseen perturbation targets.

The graph and every fitted parameter are derived from training rows only.  The
implementation intentionally uses NumPy so importing this module is safe in
contract checks; full experiments still belong on an rjob worker.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import control_target_id, estimate_control_baselines, load_npz
from .metrics import regression_metrics


def _name(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def build_coexpression_graph(signatures: np.ndarray, threshold: float = 0.3,
                             neighbors: int = 32, block_size: int = 256
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Build a bounded-memory, positive-correlation, row-stochastic graph."""
    if signatures.ndim != 2 or signatures.shape[0] < 2:
        raise ValueError("signatures must have shape [observations, genes]")
    if not 0 <= threshold < 1 or neighbors < 1:
        raise ValueError("threshold must be in [0, 1) and neighbors must be positive")
    gene_profiles = signatures.astype(np.float32).T
    gene_profiles -= gene_profiles.mean(axis=1, keepdims=True)
    gene_profiles = _unit_rows(gene_profiles)
    n_genes = len(gene_profiles)
    k = min(neighbors, max(1, n_genes - 1))
    indices = np.empty((n_genes, k), dtype=np.int32)
    weights = np.empty((n_genes, k), dtype=np.float32)
    for start in range(0, n_genes, block_size):
        end = min(start + block_size, n_genes)
        corr = gene_profiles[start:end] @ gene_profiles.T
        corr[np.arange(end - start), np.arange(start, end)] = -np.inf
        candidate = np.argpartition(corr, -k, axis=1)[:, -k:]
        values = np.take_along_axis(corr, candidate, axis=1)
        values = np.where(values >= threshold, values, 0.0).astype(np.float32)
        # Isolated genes receive a deterministic self-loop.
        isolated = values.sum(axis=1) == 0
        if isolated.any():
            candidate[isolated, 0] = np.arange(start, end)[isolated]
            values[isolated, 0] = 1.0
        values /= values.sum(axis=1, keepdims=True)
        indices[start:end], weights[start:end] = candidate, values
    return indices, weights


def degree_matched_random_graph(indices: np.ndarray, weights: np.ndarray,
                                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomize edge destinations while preserving every source out-degree."""
    rng = np.random.default_rng(seed)
    flat = indices.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(indices.shape), weights.copy()


def pagerank(seed_nodes: np.ndarray, indices: np.ndarray, weights: np.ndarray,
             alpha: float = 0.15, iterations: int = 30) -> np.ndarray:
    """Batch personalized PageRank using sparse neighbor arrays."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    n_genes = len(indices)
    seeds = np.asarray(seed_nodes, dtype=np.int64)
    if np.any((seeds < 0) | (seeds >= n_genes)):
        raise ValueError("seed node outside graph")
    restart = np.zeros((len(seeds), n_genes), dtype=np.float32)
    restart[np.arange(len(seeds)), seeds] = 1.0
    scores = restart.copy()
    for _ in range(iterations):
        propagated = np.zeros_like(scores)
        for edge in range(indices.shape[1]):
            np.add.at(propagated, (slice(None), indices[:, edge]),
                      scores * weights[:, edge][None, :])
        scores = alpha * restart + (1.0 - alpha) * propagated
    return scores


def pagerank_torch(seed_nodes: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                   alpha: float = 0.15, iterations: int = 30,
                   device: str = "cuda") -> np.ndarray:
    """Sparse-matrix equivalent of pagerank for full official gene panels."""
    import torch

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    n_genes = len(indices)
    seeds = np.asarray(seed_nodes, dtype=np.int64)
    if np.any((seeds < 0) | (seeds >= n_genes)):
        raise ValueError("seed node outside graph")
    sources = np.repeat(np.arange(n_genes, dtype=np.int64), indices.shape[1])
    edge_index = torch.from_numpy(np.stack([indices.ravel(), sources])).to(device)
    edge_weight = torch.from_numpy(weights.ravel()).to(device)
    transition_t = torch.sparse_coo_tensor(
        edge_index, edge_weight, (n_genes, n_genes), device=device).coalesce()
    restart = torch.zeros((len(seeds), n_genes), dtype=torch.float32, device=device)
    restart[torch.arange(len(seeds), device=device), torch.from_numpy(seeds).to(device)] = 1.0
    scores = restart.clone()
    for _ in range(iterations):
        scores = alpha * restart + (1.0 - alpha) * torch.sparse.mm(transition_t, scores.T).T
    return scores.cpu().numpy()


@dataclass
class RNDP:
    indices: np.ndarray
    weights: np.ndarray
    diffusion_basis: np.ndarray
    calibration: np.ndarray
    train_diffusion: np.ndarray
    sparse_residual: np.ndarray
    alpha: float
    iterations: int

    @classmethod
    def fit_graph(cls, signatures: np.ndarray, seed_nodes: np.ndarray,
                  indices: np.ndarray, weights: np.ndarray, rank: int = 64,
                  lasso_alpha: float = 1e-3, pagerank_alpha: float = 0.15,
                  pagerank_iterations: int = 30) -> "RNDP":
        diffusion = pagerank(seed_nodes, indices, weights,
                             pagerank_alpha, pagerank_iterations)
        _, _, vt = np.linalg.svd(diffusion, full_matrices=False)
        basis = vt[:min(rank, len(vt))].astype(np.float32)
        features = diffusion @ basis.T
        calibration = np.linalg.lstsq(features, signatures, rcond=1e-5)[0].astype(np.float32)
        low_rank = features @ calibration
        residual = signatures - low_rank
        sparse = np.sign(residual) * np.maximum(np.abs(residual) - lasso_alpha, 0.0)
        return cls(indices, weights, basis, calibration, _unit_rows(diffusion),
                   sparse.astype(np.float32), pagerank_alpha, pagerank_iterations)

    @classmethod
    def fit(cls, signatures: np.ndarray, seed_nodes: np.ndarray, rank: int = 64,
            lasso_alpha: float = 1e-3, graph_threshold: float = 0.3,
            neighbors: int = 32, pagerank_alpha: float = 0.15,
            pagerank_iterations: int = 30) -> "RNDP":
        indices, weights = build_coexpression_graph(
            signatures, graph_threshold, neighbors)
        return cls.fit_graph(signatures, seed_nodes, indices, weights, rank,
                             lasso_alpha, pagerank_alpha, pagerank_iterations)

    def predict_delta(self, seed_nodes: np.ndarray) -> np.ndarray:
        diffusion = pagerank(seed_nodes, self.indices, self.weights,
                             self.alpha, self.iterations)
        low_rank = (diffusion @ self.diffusion_basis.T) @ self.calibration
        similarity = np.maximum(_unit_rows(diffusion) @ self.train_diffusion.T, 0.0)
        similarity /= np.maximum(similarity.sum(axis=1, keepdims=True), 1e-8)
        return (low_rank + similarity @ self.sparse_residual).astype(np.float32)

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "pagerank_diffusion_nonzero": int(np.count_nonzero(self.train_diffusion)),
            "calibration_weight_norm_positive": float(np.linalg.norm(self.calibration)),
            "sparse_correction_nonzero_count": int(np.count_nonzero(self.sparse_residual)),
        }


def _target_signatures(x: np.ndarray, target: np.ndarray, batch: np.ndarray,
                       baselines: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    delta = x - baselines[batch]
    return np.stack([delta[target == target_id].mean(axis=0) for target_id in target_ids])


def run_npz(args: argparse.Namespace) -> dict:
    split = load_npz(args.data_npz, split_strategy="artifact")
    control = control_target_id(split.target_names)
    baselines, _ = estimate_control_baselines(
        split.x_train, split.target_train, split.batch_train,
        len(split.batch_names), control)
    train_ids = np.setdiff1d(np.unique(split.target_train), [control])
    gene_lookup = {_name(name): i for i, name in enumerate(split.gene_names)}
    seed_nodes = np.asarray([gene_lookup.get(_name(split.target_names[i]), -1) for i in train_ids])
    mapped = seed_nodes >= 0
    if mapped.sum() < 2:
        raise ValueError("fewer than two training targets map to gene names")
    signatures = _target_signatures(split.x_train, split.target_train,
                                    split.batch_train, baselines, train_ids)[mapped]
    model = RNDP.fit(signatures, seed_nodes[mapped], args.calibration_rank,
                     args.lasso_alpha, args.coexpression_threshold,
                     args.neighbors, args.pagerank_alpha, args.pagerank_iterations)
    val_is_control = split.target_val == control
    val_seed = np.asarray([gene_lookup.get(_name(split.target_names[i]), -1)
                           for i in split.target_val], dtype=np.int64)
    missing_mask = (val_seed < 0) & ~val_is_control
    if np.any(missing_mask):
        missing = sorted({_name(split.target_names[i]) for i, bad in
                          zip(split.target_val, missing_mask) if bad})
        raise ValueError(f"validation targets absent from gene panel: {missing[:5]}")
    predicted_delta = np.zeros_like(split.x_val, dtype=np.float32)
    predicted_delta[~val_is_control] = model.predict_delta(val_seed[~val_is_control])
    pred = baselines[split.batch_val] + predicted_delta
    metrics = regression_metrics(split.x_val, pred, baselines[split.batch_val])
    random_indices, random_weights = degree_matched_random_graph(
        model.indices, model.weights, args.seed)
    random_model = RNDP.fit_graph(
        signatures, seed_nodes[mapped], random_indices, random_weights,
        args.calibration_rank, args.lasso_alpha, args.pagerank_alpha,
        args.pagerank_iterations)
    random_delta = np.zeros_like(split.x_val, dtype=np.float32)
    random_delta[~val_is_control] = random_model.predict_delta(val_seed[~val_is_control])
    random_pred = baselines[split.batch_val] + random_delta
    random_metrics = regression_metrics(split.x_val, random_pred, baselines[split.batch_val])
    result = {
        "status": "ok", "candidate_variant": "candidate_rndp",
        "split_strategy": split.split_strategy, "metrics": metrics,
        "diagnostics": {**model.diagnostics(),
            "random_graph_control_pearson_delta": random_metrics["mean_delta_pearson"]},
        "random_graph_metrics": random_metrics,
        "configuration": vars(args) | {"data_npz": str(args.data_npz), "output": str(args.output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pagerank-alpha", type=float, default=0.15)
    parser.add_argument("--pagerank-iterations", type=int, default=30)
    parser.add_argument("--calibration-rank", type=int, default=64)
    parser.add_argument("--lasso-alpha", type=float, default=0.001)
    parser.add_argument("--coexpression-threshold", type=float, default=0.3)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    print(json.dumps(run_npz(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
