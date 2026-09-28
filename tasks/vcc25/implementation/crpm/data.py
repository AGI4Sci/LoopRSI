from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "x_train", "x_val", "target_train", "target_val", "batch_train",
    "batch_val", "guide_train", "guide_val", "gene_names", "target_names",
    "guide_names", "batch_names",
}


@dataclass
class Split:
    x_train: np.ndarray
    x_val: np.ndarray
    target_train: np.ndarray
    target_val: np.ndarray
    batch_train: np.ndarray
    batch_val: np.ndarray
    guide_train: np.ndarray
    guide_val: np.ndarray
    gene_names: np.ndarray
    target_names: np.ndarray
    guide_names: np.ndarray
    batch_names: np.ndarray
    split_strategy: str


def load_npz(path: str | Path, split_strategy: str = "artifact", seed: int = 0,
             heldout_target_fraction: float = 0.2,
             heldout_batch_fraction: float = 0.2) -> Split:
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = REQUIRED_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"NPZ is missing required keys: {sorted(missing)}")
        values = {key: archive[key] for key in REQUIRED_KEYS}
        test_keys = {"x_test", "target_test", "batch_test", "guide_test"}
        test_values = ({key: archive[key] for key in test_keys}
                       if test_keys.issubset(archive.files) else {})

    if values["x_train"].ndim != 2 or values["x_val"].ndim != 2:
        raise ValueError("x_train and x_val must be two-dimensional")
    if values["x_train"].shape[1] != values["x_val"].shape[1]:
        raise ValueError("train and validation gene dimensions differ")

    if split_strategy == "artifact":
        label = "artifact_row_engineering_only"
        metadata_path = Path(path).with_suffix(".json")
        if metadata_path.exists():
            import json
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label = str(metadata.get("split_strategy", label))
        return Split(**values, split_strategy=label)
    if split_strategy not in {"heldout_target", "heldout_guide", "heldout_batch"}:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    x_parts = [values["x_train"], values["x_val"]]
    target_parts = [values["target_train"], values["target_val"]]
    batch_parts = [values["batch_train"], values["batch_val"]]
    guide_parts = [values["guide_train"], values["guide_val"]]
    if test_values and len(test_values["x_test"]):
        x_parts.append(test_values["x_test"])
        target_parts.append(test_values["target_test"])
        batch_parts.append(test_values["batch_test"])
        guide_parts.append(test_values["guide_test"])
    x = np.concatenate(x_parts)
    target = np.concatenate(target_parts).astype(np.int64)
    batch = np.concatenate(batch_parts).astype(np.int64)
    guide = np.concatenate(guide_parts).astype(np.int64)
    control_id = control_target_id(values["target_names"])
    rng = np.random.default_rng(seed)
    if split_strategy == "heldout_target":
        candidates = np.setdiff1d(np.unique(target), np.asarray([control_id]))
        n_holdout = max(1, int(round(len(candidates) * heldout_target_fraction)))
        heldout = rng.choice(candidates, size=min(n_holdout, len(candidates)), replace=False)
        val_mask = np.isin(target, heldout)
        label = "heldout_target_scientific_candidate"
    elif split_strategy == "heldout_guide":
        # A guide holdout is only informative when its target still has another
        # guide in training. Hold out exactly one guide for every non-control
        # target with >=2 guides; singleton-guide targets remain in training.
        heldout_list = []
        for target_id in np.setdiff1d(np.unique(target), np.asarray([control_id])):
            target_guides = np.unique(guide[target == target_id])
            if len(target_guides) >= 2:
                heldout_list.append(int(rng.choice(target_guides)))
        if not heldout_list:
            raise ValueError("No target has multiple guides; paired held-out-guide split is impossible")
        heldout = np.asarray(sorted(heldout_list), dtype=np.int64)
        val_mask = np.isin(guide, heldout)
        label = "paired_heldout_guide_scientific_candidate"
    else:
        candidates = np.unique(batch)
        if len(candidates) < 2:
            raise ValueError("Batch holdout requires at least two batches")
        n_holdout = max(1, int(round(len(candidates) * heldout_batch_fraction)))
        n_holdout = min(n_holdout, len(candidates) - 1)
        heldout = rng.choice(candidates, size=n_holdout, replace=False)
        val_mask = np.isin(batch, heldout)
        label = "heldout_batch_scientific_candidate"
    if not val_mask.any() or val_mask.all():
        raise ValueError(f"{split_strategy} produced an empty partition")
    return Split(
        x_train=x[~val_mask], x_val=x[val_mask],
        target_train=target[~val_mask], target_val=target[val_mask],
        batch_train=batch[~val_mask], batch_val=batch[val_mask],
        guide_train=guide[~val_mask], guide_val=guide[val_mask],
        gene_names=values["gene_names"], target_names=values["target_names"],
        guide_names=values["guide_names"], batch_names=values["batch_names"],
        split_strategy=label,
    )


def control_target_id(target_names: np.ndarray) -> int:
    normalized = np.char.lower(np.char.strip(target_names.astype(str)))
    hits = np.flatnonzero(normalized == "non-targeting")
    if len(hits) != 1:
        raise ValueError("Expected exactly one target category named 'non-targeting'")
    return int(hits[0])


def limit_rows(x: np.ndarray, target: np.ndarray, batch: np.ndarray, limit: int) -> tuple[np.ndarray, ...]:
    n = len(x) if limit <= 0 else min(limit, len(x))
    return x[:n], target[:n].astype(np.int64), batch[:n].astype(np.int64)


def estimate_control_baselines(x: np.ndarray, target: np.ndarray, batch: np.ndarray,
                               n_batches: int, control_id: int,
                               min_controls: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return per-batch control means and whether each is directly estimated."""
    controls = target == control_id
    if not controls.any():
        raise ValueError("Training split contains no non-targeting control cells")
    global_control = x[controls].mean(axis=0, dtype=np.float64).astype(np.float32)
    baselines = np.repeat(global_control[None, :], n_batches, axis=0)
    direct = np.zeros(n_batches, dtype=bool)
    for batch_id in range(n_batches):
        mask = controls & (batch == batch_id)
        if int(mask.sum()) >= min_controls:
            baselines[batch_id] = x[mask].mean(axis=0, dtype=np.float64)
            direct[batch_id] = True
    return baselines, direct
