from __future__ import annotations

import numpy as np


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(y: np.ndarray, pred: np.ndarray, reference: np.ndarray,
                       top_k: int = 50) -> dict[str, float | int | None]:
    delta_y = y - reference
    delta_pred = pred - reference
    mean_true = delta_y.mean(axis=0)
    mean_pred = delta_pred.mean(axis=0)
    k = min(top_k, y.shape[1])
    true_top = set(np.argpartition(np.abs(mean_true), -k)[-k:].tolist())
    pred_top = set(np.argpartition(np.abs(mean_pred), -k)[-k:].tolist())
    return {
        "mse": float(np.mean((y - pred) ** 2)),
        "delta_mse": float(np.mean((delta_y - delta_pred) ** 2)),
        "mean_profile_pearson": _safe_pearson(y.mean(axis=0), pred.mean(axis=0)),
        "mean_delta_pearson": _safe_pearson(mean_true, mean_pred),
        "delta_energy_ratio": float(np.mean(delta_pred ** 2) / max(np.mean(delta_y ** 2), 1e-12)),
        "top_k_delta_overlap": float(len(true_top & pred_top) / k),
        "top_k": k,
    }


def grouped_regression_metrics(y: np.ndarray, pred: np.ndarray,
                               reference: np.ndarray, group_ids: np.ndarray,
                               group_names: np.ndarray,
                               top_k: int = 50) -> list[dict]:
    """Compute the existing metrics independently for every observed group."""
    if not (len(y) == len(pred) == len(reference) == len(group_ids)):
        raise ValueError("Grouped metric inputs must have the same row count")
    rows = []
    for group_id in np.unique(group_ids.astype(np.int64)):
        mask = group_ids == group_id
        raw_name = group_names[int(group_id)]
        if isinstance(raw_name, bytes):
            name = raw_name.decode("utf-8", errors="replace")
        else:
            name = str(raw_name)
        rows.append({
            "group_id": int(group_id),
            "group_name": name,
            "n_cells": int(mask.sum()),
            **regression_metrics(y[mask], pred[mask], reference[mask], top_k),
        })
    return rows
