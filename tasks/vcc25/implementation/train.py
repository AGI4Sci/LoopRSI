from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from crpm.data import (control_target_id, estimate_control_baselines, limit_rows,
                       load_npz)
from crpm.metrics import grouped_regression_metrics, regression_metrics
from crpm.model import CRPM
from crpm.flow import train_conditional_flow
from crpm.prototype import (fit_hierarchical_prototype,
                            predict_hierarchical_prototype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate the minimal VCC25 CRPM prototype")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset", default="vcc25_smoke.npz")
    p.add_argument("--method", choices=["global_mean", "batch_control_mean", "crpm", "guide_crpm", "hierarchy_crpm", "go_hierarchy_crpm", "flow_matching", "go_hierarchy_flow_matching", "prototype_flow_matching", "go_hierarchy_prototype_flow_matching", "target_prototype"], default="crpm")
    p.add_argument("--split-strategy", choices=["artifact", "heldout_target", "heldout_guide", "heldout_batch"], default="artifact")
    p.add_argument("--heldout-target-fraction", type=float, default=0.2)
    p.add_argument("--heldout-batch-fraction", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20250805)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0, help="0 means no optimizer-step budget")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--shrinkage", type=float, default=1e-3)
    p.add_argument("--guide-shrinkage", type=float, default=2e-3)
    p.add_argument("--guide-shrinkage-type", choices=["constant", "frequency"],
                   default="constant",
                   help="Use constant or inverse-log-frequency guide regularization")
    p.add_argument("--guide-dropout", type=float, default=0.0,
                   help="Probability of hiding the guide residual during training")
    p.add_argument("--use-guide-prior", action="store_true",
                   help="Learn a target-conditioned prior for guide residuals")
    p.add_argument("--heldout-guide-prior-scale", type=float, default=0.0,
                   help="At evaluation, add this multiple of the same-target mean training-guide residual")
    p.add_argument("--hierarchy-embedding-dim", type=int, default=32)
    p.add_argument("--hierarchy-aggregation", choices=["weighted_sum"], default="weighted_sum")
    p.add_argument("--hierarchy-alpha-learnable", action="store_true")
    p.add_argument("--hierarchy-shuffle", action="store_true",
                   help="Negative control: deterministically shuffle guide-to-parent assignments")
    p.add_argument("--hierarchy-graph", type=Path,
                   help="Dataset Adapter materialized hierarchy graph JSON")
    p.add_argument("--hierarchy-dropout", type=float, default=0.0)
    p.add_argument("--hierarchy-depth-weight-power", type=float, default=0.5)
    p.add_argument("--hierarchy-layer-mode", choices=["target_only", "biological_process_only", "full"], default="full")
    p.add_argument("--flow-condition-dim", type=int, default=64)
    p.add_argument("--flow-hidden-dim", type=int, default=256)
    p.add_argument("--flow-depth", type=int, default=2)
    p.add_argument("--flow-inference-steps", type=int, default=8)
    p.add_argument("--delta-loss-weight", type=float, default=0.0,
                   help="Extra weight for genes with strong training-set perturbation deltas")
    p.add_argument("--loss-type", choices=["mse", "huber"], default="mse",
                   help="Elementwise reconstruction loss (default: mse)")
    p.add_argument("--huber-delta", type=float, default=1.0,
                   help="Transition point for --loss-type huber")
    p.add_argument("--target-balanced-sampling", action="store_true",
                   help="Sample cells inversely proportional to their target frequency")
    p.add_argument("--aggregate-training", action="store_true",
                   help="Train on equally weighted target-guide-batch pseudo-bulk means")
    p.add_argument("--aggregate-count-power", type=float, default=0.0,
                   help="Pseudo-bulk sampling weight exponent for group cell counts (0=equal, 1=cell frequency)")
    p.add_argument("--target-sampling-power", type=float, default=0.0,
                   help="Target-frequency exponent: 0 disables, 0.5 uses inverse sqrt, 1 uses inverse frequency")
    p.add_argument("--target-sampling-max-weight", type=float, default=0.0,
                   help="Optional cap applied after mean-one normalization; 0 disables the cap")
    p.add_argument("--no-batch-calibration", action="store_true")
    p.add_argument("--target-batch-interaction", action="store_true",
                   help="Learn target-specific batch response weights")
    p.add_argument("--interaction-shrinkage", type=float, default=0.0,
                   help="Extra L2 shrinkage for target-by-batch interaction weights")
    p.add_argument("--min-controls-per-batch", type=int, default=2)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--group-metrics", action="store_true",
                   help="Include per-target, per-guide, and per-batch evaluation metrics")
    p.add_argument("--prototype-rank", type=int, default=16)
    p.add_argument("--prototype-blend", type=float, default=1.0)
    p.add_argument("--prototype-iterations", type=int, default=5)
    p.add_argument("--prototype-target-pseudocount", type=float, default=100.0)
    p.add_argument("--prototype-batch-pseudocount", type=float, default=100.0)
    p.add_argument("--metrics-out", type=Path, default=Path("artifacts/metrics.json"))
    p.add_argument("--save-checkpoint", type=Path)
    return p.parse_args()


def predict_batches(model: CRPM, x: np.ndarray, target: np.ndarray, batch: np.ndarray,
                    guide: np.ndarray, batch_size: int, device: torch.device,
                    target_guide_prior: torch.Tensor | None = None,
                    use_guide_ids: bool = True) -> np.ndarray:
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            t = torch.from_numpy(target[start:start + batch_size]).to(device)
            b = torch.from_numpy(batch[start:start + batch_size]).to(device)
            g = (torch.from_numpy(guide[start:start + batch_size]).to(device)
                 if use_guide_ids else None)
            chunks.append(model(t, b, g, target_guide_prior=target_guide_prior).cpu().numpy())
    return np.concatenate(chunks)


def guide_parent_mapping(
    guide_train: np.ndarray,
    target_train: np.ndarray,
    guide_val: np.ndarray,
    target_val: np.ndarray,
    n_guides: int,
    control_id: int,
    shuffle: bool,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Build the annotation-only root->target->guide hierarchy."""
    guides = np.concatenate((guide_train, guide_val))
    targets = np.concatenate((target_train, target_val))
    parents = np.full(n_guides, -1, dtype=np.int64)
    for guide_id in range(n_guides):
        observed = np.unique(targets[guides == guide_id])
        if len(observed) != 1:
            raise ValueError(
                f"guide {guide_id} must map to exactly one target, got {observed.tolist()}"
            )
        parents[guide_id] = int(observed[0])
    original = parents.copy()
    if shuffle:
        candidates = np.flatnonzero(parents != control_id)
        if len(candidates) < 2:
            raise ValueError("hierarchy shuffle requires at least two non-control guides")
        rng = np.random.default_rng(seed + 7919)
        shuffled_values = parents[candidates].copy()
        for _ in range(20):
            rng.shuffle(shuffled_values)
            if np.any(shuffled_values != parents[candidates]):
                break
        parents[candidates] = shuffled_values
    return parents, float(np.mean(parents == original))


def load_go_hierarchy(
    path: Path,
    guide_names: np.ndarray,
    depth_weight_power: float,
    layer_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convert an Adapter-materialized GO DAG to per-guide ancestor weights."""
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, dict) or not isinstance(edges, list):
        raise ValueError("hierarchy graph must contain nodes and edges")
    if depth_weight_power < 0:
        raise ValueError("--hierarchy-depth-weight-power must be non-negative")
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    parents: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError("hierarchy graph edge must be [parent, child, edge_type]")
        parent, child, _ = edge
        if parent not in nodes or child not in nodes:
            raise ValueError("hierarchy graph edge references an unknown node")
        adjacency[parent].append(child)
        parents[child].append(parent)
    root = str(graph.get("root", "root"))
    depth = {root: 0}
    queue = [root]
    for node in queue:
        for child in adjacency[node]:
            next_depth = depth[node] + 1
            if child not in depth or next_depth < depth[child]:
                depth[child] = next_depth
                queue.append(child)
    allowed_types = {
        "target_only": {"root", "target", "control_target"},
        "biological_process_only": {"root", "biological_process"},
        "full": {"root", "biological_process", "target", "control_target"},
    }[layer_mode]
    eligible = sorted(
        node for node, value in nodes.items() if value.get("node_type") in allowed_types
    )
    node_index = {node: index for index, node in enumerate(eligible)}
    layer_for_type = {"root": 0, "biological_process": 1, "target": 2, "control_target": 2}
    layers = np.array([layer_for_type[nodes[node]["node_type"]] for node in eligible], dtype=np.int64)
    guide_nodes = {
        value.get("label"): node for node, value in nodes.items() if value.get("node_type") == "guide"
    }
    membership = np.zeros((len(guide_names), len(eligible)), dtype=np.float32)
    ancestor_counts = []
    for guide_id, guide_name in enumerate(guide_names.astype(str)):
        guide_node = guide_nodes.get(guide_name)
        if guide_node is None:
            raise ValueError(f"hierarchy graph has no guide node for {guide_name}")
        ancestors = set()
        stack = list(parents[guide_node])
        while stack:
            node = stack.pop()
            if node in ancestors:
                continue
            ancestors.add(node)
            stack.extend(parents[node])
        selected = [node for node in ancestors if node in node_index]
        ancestor_counts.append(len(selected))
        for layer in (0, 1, 2):
            layer_nodes = [node for node in selected if layers[node_index[node]] == layer]
            if not layer_nodes:
                continue
            weights = np.array(
                [(depth.get(node, 0) + 1.0) ** (-depth_weight_power) for node in layer_nodes],
                dtype=np.float64,
            )
            weights /= max(float(weights.sum()), 1e-12)
            for node, weight in zip(layer_nodes, weights.tolist()):
                membership[guide_id, node_index[node]] = weight
    if np.any(membership.sum(axis=1) <= 0):
        raise ValueError("at least one guide has no usable hierarchy ancestors")
    metadata = dict(graph.get("metadata") or {})
    metadata.update({
        "graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "hierarchy_layer_mode": layer_mode,
        "hierarchy_nodes_used": len(eligible),
        "ancestor_count_min": int(min(ancestor_counts)),
        "ancestor_count_max": int(max(ancestor_counts)),
        "ancestor_count_mean": float(np.mean(ancestor_counts)),
        "layer_membership_nonzero": {
            "root": int(np.count_nonzero(membership[:, layers == 0])),
            "biological_process": int(np.count_nonzero(membership[:, layers == 1])),
            "target": int(np.count_nonzero(membership[:, layers == 2])),
        },
    })
    return membership, layers, metadata


def main() -> int:
    args = parse_args()
    if args.huber_delta <= 0:
        raise ValueError("--huber-delta must be greater than zero")
    if not 0.0 <= args.guide_dropout <= 1.0:
        raise ValueError("--guide-dropout must be between 0 and 1")
    if not 0.0 <= args.heldout_guide_prior_scale <= 1.0:
        raise ValueError("--heldout-guide-prior-scale must be between 0 and 1")
    if not 0.0 <= args.target_sampling_power <= 1.0:
        raise ValueError("--target-sampling-power must be between 0 and 1")
    if args.target_sampling_max_weight < 0:
        raise ValueError("--target-sampling-max-weight must be non-negative")
    if args.interaction_shrinkage < 0:
        raise ValueError("--interaction-shrinkage must be non-negative")
    if not 0.0 <= args.aggregate_count_power <= 1.0:
        raise ValueError("--aggregate-count-power must be between 0 and 1")
    if args.aggregate_count_power > 0 and not args.aggregate_training:
        raise ValueError("--aggregate-count-power requires --aggregate-training")
    if args.aggregate_count_power > 0 and (args.target_balanced_sampling or args.target_sampling_power > 0):
        raise ValueError("aggregate count weighting and target-balanced sampling cannot be combined")
    if args.use_guide_prior and args.method != "guide_crpm":
        raise ValueError("--use-guide-prior requires --method guide_crpm")
    if args.hierarchy_embedding_dim <= 0:
        raise ValueError("--hierarchy-embedding-dim must be positive")
    if not 0.0 <= args.hierarchy_dropout < 1.0:
        raise ValueError("--hierarchy-dropout must be in [0, 1)")
    if min(args.flow_condition_dim, args.flow_hidden_dim, args.flow_depth, args.flow_inference_steps) < 1:
        raise ValueError("flow dimensions, depth, and inference steps must be positive")
    started = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu only for deliberate local use")
    device = torch.device(args.device)
    split = load_npz(args.data_root / args.dataset, args.split_strategy, args.seed,
                     args.heldout_target_fraction, args.heldout_batch_fraction)
    x_train, target_train, batch_train = limit_rows(
        split.x_train, split.target_train, split.batch_train, args.train_limit)
    x_val, target_val, batch_val = limit_rows(
        split.x_val, split.target_val, split.batch_val, args.eval_limit)
    guide_train = split.guide_train[:len(x_train)].astype(np.int64)
    guide_val = split.guide_val[:len(x_val)].astype(np.int64)
    control_id = control_target_id(split.target_names)
    baselines, direct = estimate_control_baselines(
        x_train, target_train, batch_train, len(split.batch_names), control_id,
        args.min_controls_per_batch)
    global_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    global_pred = np.repeat(global_mean[None, :], len(x_val), axis=0)
    baseline_pred = baselines[batch_val]

    history: list[float] = []
    optimizer_steps = 0
    target_sampling_power = args.target_sampling_power
    target_sample_weight_min = 1.0
    target_sample_weight_max = 1.0
    prototype_diagnostics = None
    n_fit_samples = len(x_train)
    aggregate_group_counts = None
    guide_shrinkage_weight_min = 1.0
    guide_shrinkage_weight_max = 1.0
    hierarchy_diagnostics = None
    if args.method == "global_mean":
        pred = global_pred
    elif args.method == "batch_control_mean":
        pred = baseline_pred
    elif args.method in {
        "flow_matching", "go_hierarchy_flow_matching",
        "prototype_flow_matching", "go_hierarchy_prototype_flow_matching",
    }:
        go_flow = args.method in {
            "go_hierarchy_flow_matching", "go_hierarchy_prototype_flow_matching",
        }
        prototype_flow = args.method in {
            "prototype_flow_matching", "go_hierarchy_prototype_flow_matching",
        }
        hierarchy_membership = None
        hierarchy_node_layers = None
        hierarchy_graph_metadata = None
        if go_flow:
            if args.hierarchy_graph is None:
                raise ValueError("--hierarchy-graph is required by a GO hierarchy flow method")
            hierarchy_membership, hierarchy_node_layers, hierarchy_graph_metadata = load_go_hierarchy(
                args.hierarchy_graph, split.guide_names, args.hierarchy_depth_weight_power,
                args.hierarchy_layer_mode,
            )
        train_initial = None
        val_initial = None
        if prototype_flow:
            target_effect, batch_effect, prototype_diagnostics = fit_hierarchical_prototype(
                x_train, target_train, batch_train, baselines,
                len(split.target_names), len(split.batch_names), control_id,
                rank=args.prototype_rank, blend=args.prototype_blend,
                iterations=args.prototype_iterations,
                target_pseudocount=args.prototype_target_pseudocount,
                batch_pseudocount=args.prototype_batch_pseudocount,
            )
            train_initial = predict_hierarchical_prototype(
                baselines, target_effect, batch_effect, target_train, batch_train
            )
            val_initial = predict_hierarchical_prototype(
                baselines, target_effect, batch_effect, target_val, batch_val
            )
        flow = train_conditional_flow(
            x_train=x_train,
            target_train=target_train,
            batch_train=batch_train,
            guide_train=guide_train,
            target_val=target_val,
            batch_val=batch_val,
            guide_val=guide_val,
            baselines=baselines,
            n_targets=len(split.target_names),
            n_guides=len(split.guide_names),
            n_batches=len(split.batch_names),
            device=device,
            seed=args.seed,
            epochs=args.epochs,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            aggregate_count_power=args.aggregate_count_power,
            condition_dim=args.flow_condition_dim,
            hidden_dim=args.flow_hidden_dim,
            depth=args.flow_depth,
            inference_steps=args.flow_inference_steps,
            hierarchy_membership=hierarchy_membership,
            hierarchy_node_layers=hierarchy_node_layers,
            hierarchy_embedding_dim=args.hierarchy_embedding_dim,
            hierarchy_dropout=args.hierarchy_dropout,
            hierarchy_alpha_learnable=args.hierarchy_alpha_learnable,
            train_initial=train_initial,
            val_initial=val_initial,
        )
        pred = flow.predictions
        history = flow.history
        optimizer_steps = flow.optimizer_steps
        n_fit_samples = flow.fit_samples
        mechanism_diagnostics = dict(flow.diagnostics)
        hierarchy_diagnostics = {
            "contract_version": (
                "vcc25.go_hierarchy_prototype_flow_activation.v1"
                if go_flow and prototype_flow
                else "vcc25.prototype_flow_activation.v1"
                if prototype_flow
                else "vcc25.go_hierarchy_flow_activation.v1"
                if go_flow else "vcc25.flow_matching_activation.v1"
            ),
            "candidate_active": True,
            "fixed_command_method": args.method,
            "mechanisms": [{
                "id": (
                    "go_dag_hierarchy_conditioned_prototype_residual_flow_matching"
                    if go_flow and prototype_flow
                    else "prototype_residual_flow_matching"
                    if prototype_flow
                    else "go_dag_hierarchy_conditioned_flow_matching"
                    if go_flow else "conditional_flow_matching"
                ),
                "category": "model",
                "active": True,
                "diagnostics": mechanism_diagnostics,
            }],
            "combination_size": 1 + int(go_flow) + int(prototype_flow),
            "flow_condition_dim": args.flow_condition_dim,
            "flow_hidden_dim": args.flow_hidden_dim,
            "flow_depth": args.flow_depth,
            "flow_inference_steps": args.flow_inference_steps,
        }
        if prototype_flow:
            base_contribution = float(np.abs(val_initial - baseline_pred).sum())
            hierarchy_diagnostics.update({
                "prototype_residual_active": True,
                "prototype_base_contribution_abs_sum": base_contribution,
                "prototype_rank_effective": int(prototype_diagnostics["prototype_rank_effective"]),
                "prototype_blend": float(args.prototype_blend),
            })
        if go_flow:
            hierarchy_diagnostics.update(hierarchy_graph_metadata or {})
    elif args.method == "target_prototype":
        target_effect, batch_effect, prototype_diagnostics = fit_hierarchical_prototype(
            x_train, target_train, batch_train, baselines,
            len(split.target_names), len(split.batch_names), control_id,
            rank=args.prototype_rank, blend=args.prototype_blend,
            iterations=args.prototype_iterations,
            target_pseudocount=args.prototype_target_pseudocount,
            batch_pseudocount=args.prototype_batch_pseudocount)
        pred = predict_hierarchical_prototype(
            baselines, target_effect, batch_effect, target_val, batch_val)
    else:
        hierarchy_active = args.method in {"hierarchy_crpm", "go_hierarchy_crpm"}
        go_hierarchy_active = args.method == "go_hierarchy_crpm"
        guide_aware = args.method in {"guide_crpm", "hierarchy_crpm", "go_hierarchy_crpm"}
        hierarchy_parents = None
        hierarchy_membership = None
        hierarchy_node_layers = None
        hierarchy_graph_metadata = None
        hierarchy_parent_match_rate = 1.0
        if go_hierarchy_active:
            if args.hierarchy_graph is None:
                raise ValueError("--hierarchy-graph is required by go_hierarchy_crpm")
            hierarchy_membership, hierarchy_node_layers, hierarchy_graph_metadata = load_go_hierarchy(
                args.hierarchy_graph, split.guide_names, args.hierarchy_depth_weight_power,
                args.hierarchy_layer_mode,
            )
        elif hierarchy_active:
            hierarchy_parents, hierarchy_parent_match_rate = guide_parent_mapping(
                split.guide_train.astype(np.int64), split.target_train.astype(np.int64),
                split.guide_val.astype(np.int64), split.target_val.astype(np.int64),
                len(split.guide_names), control_id, args.hierarchy_shuffle, args.seed,
            )
        model = CRPM(torch.from_numpy(baselines).to(device), len(split.target_names),
                     args.rank, not args.no_batch_calibration,
                     len(split.guide_names) if guide_aware else 0,
                     args.target_batch_interaction,
                     args.use_guide_prior,
                     (torch.from_numpy(hierarchy_parents).to(device)
                      if hierarchy_parents is not None else None),
                     (torch.from_numpy(hierarchy_membership).to(device)
                      if hierarchy_membership is not None else None),
                     (torch.from_numpy(hierarchy_node_layers).to(device)
                      if hierarchy_node_layers is not None else None),
                     args.hierarchy_embedding_dim,
                     args.hierarchy_alpha_learnable,
                     args.hierarchy_dropout).to(device)
        hierarchy_initial = None
        hierarchy_activation_events = 0
        hierarchy_affected_rows = 0
        hierarchy_contribution_abs_sum = 0.0
        hierarchy_gradient_norm_max = 0.0
        if hierarchy_active:
            hierarchy_initial = torch.cat([
                model.hierarchy_embeddings.weight.detach().flatten(),
                model.hierarchy_projection.weight.detach().flatten(),
                model.hierarchy_alpha.detach().flatten(),
            ]).clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        x_fit, target_fit, batch_fit, guide_fit = x_train, target_train, batch_train, guide_train
        if args.aggregate_training:
            keys = np.stack((target_train, guide_train, batch_train), axis=1)
            order = np.lexsort((batch_train, guide_train, target_train))
            sorted_keys = keys[order]
            starts = np.flatnonzero(np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)])
            counts = np.diff(np.r_[starts, len(order)]).astype(np.float32)
            aggregate_group_counts = counts
            x_fit = np.add.reduceat(x_train[order], starts, axis=0) / counts[:, None]
            x_fit = x_fit.astype(np.float32, copy=False)
            group_keys = sorted_keys[starts]
            target_fit = group_keys[:, 0].astype(np.int64)
            guide_fit = group_keys[:, 1].astype(np.int64)
            batch_fit = group_keys[:, 2].astype(np.int64)
            n_fit_samples = len(x_fit)
        ds = TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(target_fit),
                           torch.from_numpy(batch_fit), torch.from_numpy(guide_fit))
        generator = torch.Generator().manual_seed(args.seed)
        sampler = None
        shuffle = True
        if aggregate_group_counts is not None and args.aggregate_count_power > 0:
            aggregate_weights = np.power(
                aggregate_group_counts.astype(np.float64), args.aggregate_count_power)
            aggregate_weights /= max(float(aggregate_weights.mean()), 1e-12)
            sampler = WeightedRandomSampler(
                torch.from_numpy(aggregate_weights), len(ds), replacement=True,
                generator=generator)
            shuffle = False
        if args.target_balanced_sampling and target_sampling_power == 0:
            target_sampling_power = 1.0
        if target_sampling_power > 0:
            counts = np.bincount(target_fit, minlength=len(split.target_names))
            sample_weights = np.maximum(counts[target_fit], 1).astype(np.float64)
            sample_weights = np.power(sample_weights, -target_sampling_power)
            sample_weights /= max(float(sample_weights.mean()), 1e-12)
            if args.target_sampling_max_weight > 0:
                sample_weights = np.minimum(sample_weights, args.target_sampling_max_weight)
                sample_weights /= max(float(sample_weights.mean()), 1e-12)
            target_sample_weight_min = float(sample_weights.min())
            target_sample_weight_max = float(sample_weights.max())
            sampler = WeightedRandomSampler(
                torch.from_numpy(sample_weights.astype(np.float64)), len(ds),
                replacement=True, generator=generator)
            shuffle = False
        gene_loss_weights = None
        guide_shrinkage_weights = None
        if guide_aware and args.guide_shrinkage_type == "frequency":
            guide_counts = np.bincount(
                guide_train, minlength=len(split.guide_names)).astype(np.float64)
            observed = guide_counts > 0
            weights = np.zeros_like(guide_counts, dtype=np.float64)
            weights[observed] = 1.0 / np.log1p(guide_counts[observed])
            # Preserve the average penalty scale so this ablation changes only
            # its allocation across guides, not the effective coefficient.
            weights[observed] /= max(float(weights[observed].mean()), 1e-12)
            guide_shrinkage_weight_min = float(weights[observed].min())
            guide_shrinkage_weight_max = float(weights[observed].max())
            guide_shrinkage_weights = torch.from_numpy(
                weights.astype(np.float32)).to(device)
        if args.delta_loss_weight > 0:
            train_delta = x_train - baselines[batch_train]
            delta_strength = np.mean(np.abs(train_delta), axis=0, dtype=np.float64)
            delta_strength /= max(float(delta_strength.mean()), 1e-8)
            # Prevent a few noisy genes from dominating while keeping the mean
            # extra weight equal to the requested coefficient.
            delta_strength = np.clip(delta_strength, 0.25, 4.0)
            delta_strength /= max(float(delta_strength.mean()), 1e-8)
            gene_loss_weights = torch.from_numpy(
                (1.0 + args.delta_loss_weight * delta_strength).astype(np.float32)
            ).to(device)
        stop = False
        for _epoch in range(args.epochs):
            model.train(); running = 0.0; seen = 0
            loader = DataLoader(ds, args.batch_size, shuffle=shuffle, sampler=sampler,
                                generator=generator)
            for y, target, batch, guide in loader:
                y, target, batch, guide = y.to(device), target.to(device), batch.to(device), guide.to(device)
                optimizer.zero_grad(set_to_none=True)
                guide_keep_mask = None
                if guide_aware and args.guide_dropout > 0:
                    guide_keep_mask = (
                        torch.rand(len(guide), device=device) >= args.guide_dropout
                    ).to(y.dtype)
                prediction = model(target, batch, guide if guide_aware else None,
                                   guide_keep_mask=guide_keep_mask)
                if args.loss_type == "mse":
                    reconstruction_error = (prediction - y) ** 2
                else:
                    absolute_error = torch.abs(prediction - y)
                    reconstruction_error = torch.where(
                        absolute_error <= args.huber_delta,
                        0.5 * absolute_error.square(),
                        args.huber_delta * (absolute_error - 0.5 * args.huber_delta),
                    )
                if gene_loss_weights is not None:
                    reconstruction_error = reconstruction_error * gene_loss_weights
                loss = torch.mean(reconstruction_error)
                if args.shrinkage > 0:
                    loss = loss + args.shrinkage * model.shrinkage_penalty()
                if guide_aware and args.guide_shrinkage > 0:
                    guide_penalty = model.guide_residuals.weight.square().mean(dim=1)
                    if guide_shrinkage_weights is not None:
                        guide_penalty = guide_penalty * guide_shrinkage_weights
                    loss = loss + args.guide_shrinkage * guide_penalty.mean()
                if model.target_batch_weights is not None and args.interaction_shrinkage > 0:
                    loss = loss + args.interaction_shrinkage * model.target_batch_weights.weight.square().mean()
                loss.backward(); optimizer.step()
                if hierarchy_active:
                    hierarchy_activation_events += 1
                    hierarchy_affected_rows += len(y)
                    hierarchy_contribution_abs_sum += float(
                        getattr(model, "last_hierarchy_abs_sum", 0.0)
                    )
                    grad_sq = 0.0
                    for parameter in (
                        model.hierarchy_embeddings.weight,
                        model.hierarchy_projection.weight,
                    ):
                        if parameter.grad is not None:
                            grad_sq += float(parameter.grad.detach().square().sum().cpu())
                    hierarchy_gradient_norm_max = max(
                        hierarchy_gradient_norm_max, grad_sq ** 0.5
                    )
                running += float(loss.detach()) * len(y); seen += len(y); optimizer_steps += 1
                if args.max_steps > 0 and optimizer_steps >= args.max_steps:
                    stop = True; break
            history.append(running / max(seen, 1))
            if stop: break
        target_guide_prior = None
        if guide_aware and args.heldout_guide_prior_scale > 0:
            target_guide_prior = torch.zeros(
                (len(split.target_names), args.rank), dtype=torch.float32, device=device)
            with torch.no_grad():
                for target_id in np.unique(target_train):
                    train_guides = np.unique(guide_train[target_train == target_id])
                    if len(train_guides) == 0:
                        continue
                    guide_ids = torch.from_numpy(train_guides).to(device)
                    target_guide_prior[target_id] = (
                        args.heldout_guide_prior_scale
                        * model.guide_residuals(guide_ids).mean(dim=0)
                    )
                target_guide_prior[control_id].zero_()
        pred = predict_batches(model, x_val, target_val, batch_val, guide_val,
                               args.batch_size, device, target_guide_prior,
                               use_guide_ids=hierarchy_active or not args.use_guide_prior)
        if hierarchy_active:
            hierarchy_final = torch.cat([
                model.hierarchy_embeddings.weight.detach().flatten(),
                model.hierarchy_projection.weight.detach().flatten(),
                model.hierarchy_alpha.detach().flatten(),
            ])
            mechanism_id = (
                "go_dag_root_process_target_guide_embedding"
                if go_hierarchy_active else "root_target_guide_embedding"
            )
            hierarchy_diagnostics = {
                "contract_version": (
                    "vcc25.go_dag_hierarchy_activation.v1"
                    if go_hierarchy_active else "vcc25.guide_hierarchy_activation.v1"
                ),
                "candidate_active": True,
                "fixed_command_method": args.method,
                "mechanisms": [{
                    "id": mechanism_id,
                    "category": "representation",
                    "active": True,
                    "diagnostics": {
                        "activation_events": hierarchy_activation_events,
                        "affected_rows": hierarchy_affected_rows,
                        "transformed_feature_count": args.hierarchy_embedding_dim,
                        "gradient_norm_max": hierarchy_gradient_norm_max,
                        "parameter_update_l2": float(
                            torch.linalg.vector_norm(hierarchy_final - hierarchy_initial).cpu()
                        ),
                        "contribution_abs_sum": hierarchy_contribution_abs_sum,
                        "embedding_norm": float(
                            torch.linalg.vector_norm(
                                model.hierarchy_embeddings.weight.detach()
                            ).cpu()
                        ),
                    },
                }],
                "combination_size": 1,
                "hierarchy_mode": model.hierarchy_mode,
                "hierarchy_embedding_dim": args.hierarchy_embedding_dim,
                "hierarchy_aggregation": args.hierarchy_aggregation,
                "hierarchy_alpha_learnable": args.hierarchy_alpha_learnable,
                "hierarchy_dropout": args.hierarchy_dropout,
                "hierarchy_layer_scales": model.hierarchy_alpha.detach().cpu().tolist(),
                "num_guides_mapped": int(len(split.guide_names)),
            }
            if go_hierarchy_active:
                hierarchy_diagnostics.update(hierarchy_graph_metadata or {})
                hierarchy_diagnostics["num_guides_mapped"] = int(len(split.guide_names))
            else:
                mapping_bytes = hierarchy_parents.astype("<i8", copy=False).tobytes()
                hierarchy_diagnostics.update({
                    "hierarchy_shuffle": args.hierarchy_shuffle,
                    "guide_parent_match_rate": hierarchy_parent_match_rate,
                    "guide_parent_mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
                })
        if args.save_checkpoint:
            args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_checkpoint)

    result = {
        "status": "ok", "exit_code": 0, "method": args.method,
        "dataset": args.dataset, "data_path": str(args.data_root / args.dataset),
        "device": str(device), "seed": args.seed, "split_strategy": split.split_strategy,
        "evidence_scope": "engineering_only" if args.split_strategy == "artifact" else "scientific_candidate",
        "n_train": len(x_train), "n_eval": len(x_val), "n_genes": x_train.shape[1],
        "aggregate_training": args.aggregate_training, "n_fit_samples": n_fit_samples,
        "aggregate_count_power": args.aggregate_count_power,
        "epochs_requested": args.epochs, "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps, "rank": args.rank,
        "shrinkage": args.shrinkage, "batch_calibration": not args.no_batch_calibration,
        "target_batch_interaction": args.target_batch_interaction,
        "interaction_shrinkage": args.interaction_shrinkage,
        "guide_aware": args.method in {"guide_crpm", "hierarchy_crpm", "go_hierarchy_crpm", "flow_matching", "go_hierarchy_flow_matching", "prototype_flow_matching", "go_hierarchy_prototype_flow_matching"}, "guide_shrinkage": args.guide_shrinkage,
        "guide_shrinkage_type": args.guide_shrinkage_type,
        "guide_shrinkage_weight_min": guide_shrinkage_weight_min,
        "guide_shrinkage_weight_max": guide_shrinkage_weight_max,
        "guide_dropout": args.guide_dropout,
        "use_guide_prior": args.use_guide_prior,
        "heldout_guide_prior_scale": args.heldout_guide_prior_scale,
        "delta_loss_weight": args.delta_loss_weight,
        "loss_type": args.loss_type, "huber_delta": args.huber_delta,
        "target_balanced_sampling": args.target_balanced_sampling,
        "target_sampling_power": target_sampling_power if args.method not in {"global_mean", "batch_control_mean"} else args.target_sampling_power,
        "target_sampling_max_weight": args.target_sampling_max_weight,
        "target_sample_weight_min": target_sample_weight_min if args.method not in {"global_mean", "batch_control_mean"} else 1.0,
        "target_sample_weight_max": target_sample_weight_max if args.method not in {"global_mean", "batch_control_mean"} else 1.0,
        "direct_batch_control_baselines": int(direct.sum()),
        "fallback_batch_control_baselines": int((~direct).sum()),
        "metrics": regression_metrics(x_val, pred, baseline_pred, args.top_k),
        "global_mean_baseline_metrics": regression_metrics(x_val, global_pred, baseline_pred, args.top_k),
        "batch_control_mean_baseline_metrics": regression_metrics(x_val, baseline_pred, baseline_pred, args.top_k),
        "training_loss": history, "runtime_seconds": time.time() - started,
    }
    if hierarchy_diagnostics is not None:
        result["phenotype_activation"] = hierarchy_diagnostics
    if prototype_diagnostics is not None:
        result.update(prototype_diagnostics)
    result["beats_global_mean_mse"] = result["metrics"]["mse"] < result["global_mean_baseline_metrics"]["mse"]
    if args.group_metrics:
        result["group_metrics"] = {
            "target": grouped_regression_metrics(
                x_val, pred, baseline_pred, target_val, split.target_names, args.top_k),
            "guide": grouped_regression_metrics(
                x_val, pred, baseline_pred, guide_val, split.guide_names, args.top_k),
            "batch": grouped_regression_metrics(
                x_val, pred, baseline_pred, batch_val, split.batch_names, args.top_k),
        }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "error", "exit_code": 1, "error": str(exc)}), file=sys.stderr)
        raise
