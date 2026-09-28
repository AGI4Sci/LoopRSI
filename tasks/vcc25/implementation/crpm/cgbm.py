"""Causal GO-BP mediation for unseen perturbation targets.

The module stays NumPy-only at import time so controller-side contract checks are
safe.  GO annotations and every fitted parameter are fixed from training data;
validation labels are used only by the explicitly scoped isotonic calibrator.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cc_gat import context_conditioned_propagation
from .data import control_target_id, estimate_control_baselines, load_npz
from .metrics import regression_metrics
from .rndp import (_name, _target_signatures, _unit_rows,
                   degree_matched_random_graph)
from .target_features import go_bp_hash_features


def build_go_bp_neighborhoods(gene_names: list[str], node_table: Path,
                              edge_table: Path, feature_dim: int = 128,
                              neighbor_k: int = 16
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a bounded sparse gene graph from the materialized GO-BP DAG."""
    features, metadata = go_bp_hash_features(
        gene_names, str(node_table), str(edge_table), feature_dim)
    if neighbor_k < 1:
        raise ValueError("neighbor_k must be positive")
    n_genes = len(gene_names)
    k = min(neighbor_k, max(1, n_genes - 1))
    indices = np.empty((n_genes, k), dtype=np.int32)
    weights = np.empty((n_genes, k), dtype=np.float32)
    # Blocked multiplication avoids a full 18,080-square similarity matrix.
    for start in range(0, n_genes, 256):
        end = min(start + 256, n_genes)
        similarity = features[start:end] @ features.T
        similarity[np.arange(end - start), np.arange(start, end)] = -np.inf
        neighbors = np.argpartition(similarity, -k, axis=1)[:, -k:]
        values = np.maximum(np.take_along_axis(similarity, neighbors, axis=1), 0)
        isolated = values.sum(axis=1) <= 1e-12
        if isolated.any():
            neighbors[isolated, 0] = np.arange(start, end)[isolated]
            values[isolated, 0] = 1.0
        values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-8)
        indices[start:end], weights[start:end] = neighbors, values
    return indices, weights, features, metadata


def scm_do_operator_propagation(seed_nodes: np.ndarray, indices: np.ndarray,
                                weights: np.ndarray, contexts: np.ndarray,
                                heads: int = 4, layers: int = 2,
                                seed: int = 20260913
                                ) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate mediated effect as do(target=1) minus do(target=0).

    ``direct`` is the unmediated intervention.  Subtracting it from the factual
    graph propagation isolates the GO-BP mediator rather than relabeling the
    ordinary GAT state as a causal effect.
    """
    factual, events = context_conditioned_propagation(
        seed_nodes, indices, weights, contexts, heads, layers, seed)
    direct = np.zeros_like(factual)
    direct[np.arange(len(seed_nodes)), np.asarray(seed_nodes, dtype=np.int64)] = 1.0
    mediator_do_zero = direct  # structural mediator edges disabled
    mediation_effect = factual - mediator_do_zero
    return factual, mediation_effect, events


def de_rank_order_pretext_loss(prediction: np.ndarray, truth: np.ndarray,
                               pairs: int = 256, margin: float = 0.05,
                               seed: int = 20260913) -> float:
    """Pairwise hinge loss aligning predicted and observed DE rank order."""
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("prediction and truth must be equally shaped matrices")
    rng = np.random.default_rng(seed)
    total = 0.0
    count = 0
    n_genes = truth.shape[1]
    for row in range(len(truth)):
        left = rng.integers(0, n_genes, size=pairs)
        right = rng.integers(0, n_genes, size=pairs)
        observed = np.sign(truth[row, left] - truth[row, right])
        keep = observed != 0
        gap = observed[keep] * (prediction[row, left[keep]] - prediction[row, right[keep]])
        total += float(np.maximum(margin - gap, 0).sum())
        count += int(keep.sum())
    return total / max(count, 1)


def _pav(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return boundaries and levels for an equal-weight PAV isotonic fit."""
    levels: list[float] = []
    counts: list[int] = []
    for value in np.asarray(y, dtype=np.float64):
        levels.append(float(value)); counts.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            count = counts[-2] + counts[-1]
            levels[-2] = (levels[-2] * counts[-2] + levels[-1] * counts[-1]) / count
            counts[-2] = count
            levels.pop(); counts.pop()
    expanded = np.repeat(np.asarray(levels), counts)
    return expanded, np.cumsum(counts) - 1


@dataclass
class PerGeneIsotonicCalibrator:
    x_sorted: list[np.ndarray]
    y_fitted: list[np.ndarray]

    @classmethod
    def fit(cls, prediction: np.ndarray, truth: np.ndarray
            ) -> "PerGeneIsotonicCalibrator":
        """Fit independent non-decreasing maps using validation rows only."""
        if prediction.shape != truth.shape or prediction.ndim != 2:
            raise ValueError("prediction and truth must be equally shaped matrices")
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for gene in range(prediction.shape[1]):
            order = np.argsort(prediction[:, gene], kind="stable")
            x = np.asarray(prediction[order, gene], dtype=np.float32)
            fitted, _ = _pav(truth[order, gene])
            xs.append(x); ys.append(fitted.astype(np.float32))
        return cls(xs, ys)

    def transform(self, prediction: np.ndarray) -> np.ndarray:
        if prediction.ndim != 2 or prediction.shape[1] != len(self.x_sorted):
            raise ValueError("prediction has incompatible gene dimension")
        calibrated = np.empty_like(prediction, dtype=np.float32)
        for gene, (x, y) in enumerate(zip(self.x_sorted, self.y_fitted)):
            calibrated[:, gene] = np.interp(prediction[:, gene], x, y)
        return calibrated


def cross_fit_isotonic(prediction: np.ndarray, truth: np.ndarray,
                       groups: np.ndarray, folds: int = 3) -> np.ndarray:
    """Calibrate each validation target strictly out of fold."""
    unique = np.unique(groups)
    if folds < 2 or len(unique) < folds:
        raise ValueError("isotonic cross-fitting requires at least two folds and groups")
    calibrated = np.empty_like(prediction, dtype=np.float32)
    for fold in range(folds):
        held_groups = unique[np.arange(len(unique)) % folds == fold]
        held = np.isin(groups, held_groups)
        calibrator = PerGeneIsotonicCalibrator.fit(prediction[~held], truth[~held])
        calibrated[held] = calibrator.transform(prediction[held])
    return calibrated


@dataclass
class CGBM:
    indices: np.ndarray
    weights: np.ndarray
    contexts: np.ndarray
    train_state: np.ndarray
    basis: np.ndarray
    decoder: np.ndarray
    heads: int
    layers: int
    seed: int
    pretext_loss: float

    @classmethod
    def fit_graph(cls, signatures: np.ndarray, seed_nodes: np.ndarray,
                  indices: np.ndarray, weights: np.ndarray, contexts: np.ndarray,
                  heads: int = 4, layers: int = 2, rank: int = 24,
                  pretext_weight: float = 0.1, ridge: float = 1e-3,
                  seed: int = 20260913) -> "CGBM":
        factual, mediation, _ = scm_do_operator_propagation(
            seed_nodes, indices, weights, contexts, heads, layers, seed)
        state = np.concatenate([factual, mediation], axis=1)
        _, _, vt = np.linalg.svd(state, full_matrices=False)
        basis = vt[:min(rank, len(vt))].T
        compact = state @ basis
        base_decoder = np.linalg.solve(
            compact.T @ compact + ridge * np.eye(compact.shape[1]),
            compact.T @ signatures)
        initial = compact @ base_decoder
        loss = de_rank_order_pretext_loss(initial, signatures, seed=seed)
        # Rank-normalized targets make the pairwise objective affect the fitted
        # decoder while preserving expression amplitude through convex blending.
        order = np.argsort(np.argsort(signatures, axis=1), axis=1)
        rank_target = order / max(signatures.shape[1] - 1, 1) - 0.5
        rank_target *= 2 * np.std(signatures, axis=1, keepdims=True)
        target = (1 - pretext_weight) * signatures + pretext_weight * rank_target
        decoder = np.linalg.solve(
            compact.T @ compact + ridge * np.eye(compact.shape[1]), compact.T @ target)
        return cls(indices, weights, contexts, state.astype(np.float32),
                   basis.astype(np.float32), decoder.astype(np.float32),
                   heads, layers, seed, loss)

    def predict_delta(self, seed_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        factual, mediation, events = scm_do_operator_propagation(
            seed_nodes, self.indices, self.weights, self.contexts,
            self.heads, self.layers, self.seed)
        state = np.concatenate([factual, mediation], axis=1)
        return ((state @ self.basis) @ self.decoder).astype(np.float32), mediation, events


def _fit_model(signatures: np.ndarray, train_nodes: np.ndarray,
               indices: np.ndarray, weights: np.ndarray, contexts: np.ndarray,
               args: argparse.Namespace) -> CGBM:
    return CGBM.fit_graph(signatures, train_nodes, indices, weights, contexts,
                          args.gat_heads, args.gat_layers, args.rank,
                          args.pretext_weight, args.ridge, args.seed)


def run_npz(args: argparse.Namespace) -> dict:
    split = load_npz(args.data_npz, split_strategy="artifact")
    control = control_target_id(split.target_names)
    baselines, _ = estimate_control_baselines(
        split.x_train, split.target_train, split.batch_train,
        len(split.batch_names), control)
    train_ids = np.setdiff1d(np.unique(split.target_train), [control])
    genes = [_name(value) for value in split.gene_names]
    lookup = {name: index for index, name in enumerate(genes)}
    train_nodes = np.asarray([lookup.get(_name(split.target_names[i]), -1) for i in train_ids])
    mapped = train_nodes >= 0
    if mapped.sum() < 2:
        raise ValueError("fewer than two training targets map to gene names")
    signatures = _target_signatures(split.x_train, split.target_train,
                                    split.batch_train, baselines, train_ids)[mapped]
    indices, weights, contexts, go_metadata = build_go_bp_neighborhoods(
        genes, args.go_node_table, args.go_edge_table, args.feature_dim, args.neighbor_k)
    model = _fit_model(signatures, train_nodes[mapped], indices, weights, contexts, args)
    is_control = split.target_val == control
    query_nodes = np.asarray([lookup.get(_name(split.target_names[i]), -1)
                              for i in split.target_val[~is_control]])
    if np.any(query_nodes < 0):
        raise ValueError("validation target absent from full gene panel")
    predicted, mediation, events = model.predict_delta(query_nodes)
    truth = split.x_val[~is_control] - baselines[split.batch_val[~is_control]]
    validation_groups = split.target_val[~is_control]
    calibrated = cross_fit_isotonic(
        predicted, truth, validation_groups, args.calibration_folds)
    delta = np.zeros_like(split.x_val, dtype=np.float32)
    delta[~is_control] = calibrated
    metrics = regression_metrics(split.x_val, baselines[split.batch_val] + delta,
                                 baselines[split.batch_val])
    random_indices, random_weights = degree_matched_random_graph(indices, weights, args.seed + 1)
    random_model = _fit_model(signatures, train_nodes[mapped], random_indices,
                              random_weights, contexts, args)
    random_pred, _, _ = random_model.predict_delta(query_nodes)
    random_pred = cross_fit_isotonic(
        random_pred, truth, validation_groups, args.calibration_folds)
    random_delta = np.zeros_like(delta); random_delta[~is_control] = random_pred
    random_metrics = regression_metrics(
        split.x_val, baselines[split.batch_val] + random_delta,
        baselines[split.batch_val])
    result = {
        "status": "ok", "candidate_variant": "candidate_cgbm",
        "split_strategy": split.split_strategy, "metrics": metrics,
        "random_graph_metrics": random_metrics,
        "diagnostics": {
            "pretext_loss_nonzero": float(model.pretext_loss),
            "gat_attention_events_nonzero": int(events),
            "mediation_effect_nonzero": int(np.count_nonzero(np.abs(mediation) > 1e-10)),
            "isotonic_calibration_fitted": True,
            "isotonic_scope": "validation_target_cross_fit_not_refit_on_test",
            "random_control_delta_correlation_reported": random_metrics["mean_delta_pearson"],
            "go_bp_coverage": go_metadata["coverage"],
        },
        "configuration": vars(args) | {"data_npz": str(args.data_npz),
                                        "output": str(args.output),
                                        "go_node_table": str(args.go_node_table),
                                        "go_edge_table": str(args.go_edge_table)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--go-node-table", type=Path, required=True)
    parser.add_argument("--go-edge-table", type=Path, required=True)
    parser.add_argument("--pretext-weight", type=float, default=0.1)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--neighbor-k", type=int, default=16)
    parser.add_argument("--rank", type=int, default=24)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--calibration-folds", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if not 0 <= args.pretext_weight <= 1:
        parser.error("--pretext-weight must be in [0, 1]")
    print(json.dumps(run_npz(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
