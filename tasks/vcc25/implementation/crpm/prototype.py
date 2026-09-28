from __future__ import annotations

import numpy as np


def fit_hierarchical_prototype(
    x: np.ndarray,
    target: np.ndarray,
    batch: np.ndarray,
    baselines: np.ndarray,
    n_targets: int,
    n_batches: int,
    control_id: int,
    rank: int = 16,
    blend: float = 1.0,
    iterations: int = 5,
    target_pseudocount: float = 100.0,
    batch_pseudocount: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a two-way target/batch residual model with low-rank target denoising."""
    if rank < 0:
        raise ValueError("prototype rank must be non-negative")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("prototype blend must be between 0 and 1")
    if iterations < 1:
        raise ValueError("prototype iterations must be positive")
    if target_pseudocount < 0 or batch_pseudocount < 0:
        raise ValueError("prototype pseudocounts must be non-negative")

    residual = x.astype(np.float32, copy=False) - baselines[batch]
    n_genes = residual.shape[1]
    target_counts = np.bincount(target, minlength=n_targets).astype(np.float64)
    batch_counts = np.bincount(batch, minlength=n_batches).astype(np.float64)
    target_batch_counts = np.zeros((n_targets, n_batches), dtype=np.float64)
    np.add.at(target_batch_counts, (target, batch), 1.0)

    target_sums = np.zeros((n_targets, n_genes), dtype=np.float64)
    for group_id in np.unique(target):
        target_sums[group_id] = residual[target == group_id].sum(axis=0, dtype=np.float64)
    batch_sums = np.zeros((n_batches, n_genes), dtype=np.float64)
    for group_id in np.unique(batch):
        batch_sums[group_id] = residual[batch == group_id].sum(axis=0, dtype=np.float64)

    target_effect = np.zeros((n_targets, n_genes), dtype=np.float64)
    batch_effect = np.zeros((n_batches, n_genes), dtype=np.float64)
    for _ in range(iterations):
        adjusted_target_sums = target_sums - target_batch_counts @ batch_effect
        target_effect = adjusted_target_sums / np.maximum(
            target_counts[:, None] + target_pseudocount, 1.0)
        target_effect[control_id] = 0.0

        adjusted_batch_sums = batch_sums - target_batch_counts.T @ target_effect
        batch_effect = adjusted_batch_sums / np.maximum(
            batch_counts[:, None] + batch_pseudocount, 1.0)
        # Resolve the additive-model intercept ambiguity while preserving fitted
        # values: move the cell-weighted batch mean into target effects.
        batch_mean = (batch_counts[:, None] * batch_effect).sum(axis=0) / max(batch_counts.sum(), 1.0)
        batch_effect -= batch_mean
        target_effect += batch_mean
        target_effect[control_id] = 0.0

    observed = np.flatnonzero((target_counts > 0) & (np.arange(n_targets) != control_id))
    effective_rank = min(rank, len(observed), n_genes)
    if effective_rank > 0:
        matrix = target_effect[observed]
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        projected = (u[:, :effective_rank] * singular_values[:effective_rank]) @ vt[:effective_rank]
        denoised = target_effect.copy()
        denoised[observed] = ((1.0 - blend) * matrix + blend * projected)
        denoised[control_id] = 0.0
    else:
        singular_values = np.asarray([], dtype=np.float64)
        denoised = target_effect

    total_energy = float(np.square(target_effect[observed]).sum())
    retained_energy = float(np.square(denoised[observed]).sum())
    diagnostics = {
        "prototype_rank_requested": int(rank),
        "prototype_rank_effective": int(effective_rank),
        "prototype_blend": float(blend),
        "prototype_iterations": int(iterations),
        "prototype_target_pseudocount": float(target_pseudocount),
        "prototype_batch_pseudocount": float(batch_pseudocount),
        "prototype_observed_targets": int(len(observed)),
        "prototype_denoised_energy_ratio": retained_energy / max(total_energy, 1e-12),
        "prototype_top_singular_values": singular_values[:10].tolist(),
    }
    return denoised.astype(np.float32), batch_effect.astype(np.float32), diagnostics


def predict_hierarchical_prototype(
    baselines: np.ndarray,
    target_effect: np.ndarray,
    batch_effect: np.ndarray,
    target: np.ndarray,
    batch: np.ndarray,
) -> np.ndarray:
    return baselines[batch] + target_effect[target] + batch_effect[batch]
