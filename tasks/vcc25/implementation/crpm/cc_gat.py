"""Context-conditioned sparse GRN attention for unseen perturbation targets.

The graph, decoder, and all target representations are built from training data
or target identity.  Torch is deliberately not imported so static contract checks
remain safe on the controller; the experiment itself must run on an rjob worker.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import control_target_id, estimate_control_baselines, load_npz
from .metrics import regression_metrics
from .rndp import _name, _target_signatures, _unit_rows, build_coexpression_graph
from .target_features import hashed_gene_symbol_features


def construct_context_vectors(gene_names: list[str], indices: np.ndarray,
                              context_dim: int = 128) -> np.ndarray:
    """Concatenate target, upstream-neighbor, and pathway-pool identity features."""
    if context_dim < 6:
        raise ValueError("context_dim must be at least 6")
    part = context_dim // 3
    identity = hashed_gene_symbol_features(gene_names, gene_names, part)
    upstream = identity[indices].mean(axis=1)
    # Stable gene-symbol buckets are a label-free pathway-like grouping.  This
    # fallback keeps the mechanism available when no materialized GO asset is
    # configured; it must not be interpreted as curated pathway evidence.
    buckets = np.asarray([sum(map(ord, str(g).upper())) % 64 for g in gene_names])
    bucket_pool = np.stack([
        identity[buckets == bucket].mean(axis=0)
        if np.any(buckets == bucket) else np.zeros(part, dtype=np.float32)
        for bucket in range(64)
    ])
    pathway = bucket_pool[buckets]
    context = np.concatenate([identity, upstream, pathway], axis=1)
    if context.shape[1] < context_dim:
        context = np.pad(context, ((0, 0), (0, context_dim - context.shape[1])))
    return _unit_rows(context[:, :context_dim]).astype(np.float32)


def context_conditioned_propagation(seed_nodes: np.ndarray, indices: np.ndarray,
                                    weights: np.ndarray, contexts: np.ndarray,
                                    heads: int = 4, layers: int = 2,
                                    seed: int = 20260913,
                                    context_order: np.ndarray | None = None
                                    ) -> tuple[np.ndarray, int]:
    """Propagate sparse target signals with attention conditioned on target context."""
    if heads < 1 or layers < 1:
        raise ValueError("heads and layers must be positive")
    n_genes, neighbors = indices.shape
    seeds = np.asarray(seed_nodes, dtype=np.int64)
    if np.any((seeds < 0) | (seeds >= n_genes)):
        raise ValueError("seed node outside graph")
    order = np.arange(len(seeds)) if context_order is None else np.asarray(context_order)
    if sorted(order.tolist()) != list(range(len(seeds))):
        raise ValueError("context_order must be a permutation of query rows")
    query_context = contexts[seeds[order]]
    rng = np.random.default_rng(seed)
    projections = rng.normal(0, 1 / np.sqrt(contexts.shape[1]),
                             (heads, contexts.shape[1])).astype(np.float32)
    signal = np.zeros((len(seeds), n_genes), dtype=np.float32)
    signal[np.arange(len(seeds)), seeds] = 1.0
    events = 0
    for _ in range(layers):
        propagated = np.zeros_like(signal)
        for row in range(len(seeds)):
            active = np.flatnonzero(np.abs(signal[row]) > 1e-12)
            if not len(active):
                continue
            destinations = indices[active]
            base = np.maximum(weights[active], 1e-12)
            logits = np.log(base)
            edge_context = contexts[destinations]
            for projection in projections:
                # Multiplicative interaction is essential: an additive scalar
                # target term would cancel exactly under the neighbor softmax.
                interaction = edge_context * query_context[row][None, None, :]
                logits += np.tanh(interaction @ projection) / heads
            logits -= logits.max(axis=1, keepdims=True)
            attention = np.exp(logits)
            attention /= attention.sum(axis=1, keepdims=True)
            np.add.at(propagated[row], destinations.ravel(),
                      (signal[row, active, None] * attention).ravel())
            events += int(len(active) * neighbors * heads)
        signal = 0.25 * signal + 0.75 * propagated
    return signal, events


def _topk_overlap(y: np.ndarray, scores: np.ndarray, k: int = 100) -> float:
    k = min(k, y.shape[1])
    truth = np.argpartition(np.abs(y), -k, axis=1)[:, -k:]
    ranked = np.argpartition(np.abs(scores), -k, axis=1)[:, -k:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(truth, ranked)]))


@dataclass
class CCGAT:
    indices: np.ndarray
    weights: np.ndarray
    contexts: np.ndarray
    train_signal: np.ndarray
    decoder: np.ndarray
    heads: int
    layers: int
    seed: int

    @classmethod
    def fit(cls, signatures: np.ndarray, seed_nodes: np.ndarray,
            gene_names: list[str], context_dim: int = 128, heads: int = 4,
            layers: int = 2, neighbors: int = 32, threshold: float = 0.3,
            ridge: float = 1e-3, seed: int = 20260913) -> "CCGAT":
        indices, weights = build_coexpression_graph(signatures, threshold, neighbors)
        contexts = construct_context_vectors(gene_names, indices, context_dim)
        signal, _ = context_conditioned_propagation(
            seed_nodes, indices, weights, contexts, heads, layers, seed)
        gram = signal @ signal.T + ridge * np.eye(len(signal), dtype=np.float32)
        # Dual ridge form avoids materializing an 18,080 x 18,080 decoder.
        decoder = np.linalg.solve(gram, signatures)
        return cls(indices, weights, contexts, signal, decoder.astype(np.float32),
                   heads, layers, seed)

    def predict_delta(self, seed_nodes: np.ndarray,
                      context_order: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        signal, events = context_conditioned_propagation(
            seed_nodes, self.indices, self.weights, self.contexts,
            self.heads, self.layers, self.seed, context_order)
        similarity = signal @ self.train_signal.T
        return (similarity @ self.decoder).astype(np.float32), signal, events


def run_npz(args: argparse.Namespace) -> dict:
    split = load_npz(args.data_npz, split_strategy="artifact")
    control = control_target_id(split.target_names)
    baselines, _ = estimate_control_baselines(split.x_train, split.target_train,
                                               split.batch_train, len(split.batch_names), control)
    train_ids = np.setdiff1d(np.unique(split.target_train), [control])
    genes = [_name(x) for x in split.gene_names]
    lookup = {name: i for i, name in enumerate(genes)}
    train_nodes = np.asarray([lookup.get(_name(split.target_names[i]), -1) for i in train_ids])
    mapped = train_nodes >= 0
    if mapped.sum() < 2:
        raise ValueError("fewer than two training targets map to gene names")
    signatures = _target_signatures(split.x_train, split.target_train,
                                    split.batch_train, baselines, train_ids)[mapped]
    model = CCGAT.fit(signatures, train_nodes[mapped], genes, args.context_dim,
                      args.gat_heads, args.gat_layers, args.grn_neighbors,
                      args.grn_threshold, args.ridge, args.seed)
    is_control = split.target_val == control
    query_nodes = np.asarray([lookup.get(_name(split.target_names[i]), -1)
                              for i in split.target_val[~is_control]])
    if np.any(query_nodes < 0):
        raise ValueError("validation target absent from full gene panel")
    predicted, signal, events = model.predict_delta(query_nodes)
    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(len(query_nodes))
    shuffled, _, _ = model.predict_delta(query_nodes, order)
    delta = np.zeros_like(split.x_val, dtype=np.float32)
    shuffled_delta = np.zeros_like(delta)
    delta[~is_control], shuffled_delta[~is_control] = predicted, shuffled
    pred_metrics = regression_metrics(split.x_val, baselines[split.batch_val] + delta,
                                      baselines[split.batch_val])
    shuffled_metrics = regression_metrics(split.x_val, baselines[split.batch_val] + shuffled_delta,
                                          baselines[split.batch_val])
    truth_delta = split.x_val[~is_control] - baselines[split.batch_val[~is_control]]
    pre_rank = _topk_overlap(truth_delta, signal)
    post_rank = _topk_overlap(truth_delta, predicted)
    result = {
        "status": "ok", "candidate_variant": "candidate_cc_gat",
        "split_strategy": split.split_strategy, "metrics": pred_metrics,
        "shuffled_context_metrics": shuffled_metrics,
        "diagnostics": {
            "context_vector_norm_positive": float(np.min(np.linalg.norm(model.contexts, axis=1))),
            "attention_conditioning_events_nonzero": events,
            "propagated_signal_l2_variance_across_targets": float(np.var(np.linalg.norm(signal, axis=1))),
            "shuffled_context_performance_delta": float(pred_metrics["mean_delta_pearson"] - shuffled_metrics["mean_delta_pearson"]),
            "pre_decoder_de_overlap_at_100": pre_rank,
            "post_prediction_de_overlap_at_100": post_rank,
            "pre_decoder_ranking_beats_post_prediction": pre_rank > post_rank,
        },
        "configuration": vars(args) | {"data_npz": str(args.data_npz), "output": str(args.output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--grn-neighbors", type=int, default=32)
    parser.add_argument("--grn-threshold", type=float, default=0.3)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    print(json.dumps(run_npz(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
