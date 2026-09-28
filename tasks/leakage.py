"""Domain-independent split overlap checks used by generated Task Adapters."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _selected(record: Any, keys: list[str] | None) -> Any:
    if keys is None or not isinstance(record, Mapping):
        return record
    return {key: record.get(key) for key in keys}


def split_integrity_report(
    train_records: Iterable[Any], validation_records: Iterable[Any], *,
    seed: int, strategy: str, sample_id: Callable[[Any], Any] | None = None,
    content_keys: list[str] | None = None,
    group_id: Callable[[Any], Any] | None = None,
    time_value: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Report cross-split duplicates without assuming a domain or file format."""
    train, validation = list(train_records), list(validation_records)
    if not train or not validation:
        raise ValueError("both train and validation splits must be non-empty")
    train_content = {_hash(_selected(item, content_keys)) for item in train}
    validation_content = {_hash(_selected(item, content_keys)) for item in validation}
    content_overlap = train_content.intersection(validation_content)
    if sample_id is None:
        identity_overlap: set[str] = set()
        identity_check = "not_available"
    else:
        train_ids = {_canonical(sample_id(item)) for item in train}
        validation_ids = {_canonical(sample_id(item)) for item in validation}
        identity_overlap = train_ids.intersection(validation_ids)
        identity_check = "passed" if not identity_overlap else "failed"
    if group_id is None:
        group_overlap: set[str] = set()
        group_check = "not_applicable"
    else:
        train_groups = {_canonical(group_id(item)) for item in train}
        validation_groups = {_canonical(group_id(item)) for item in validation}
        group_overlap = train_groups.intersection(validation_groups)
        group_check = "passed" if not group_overlap else "failed"
    time_order_valid: bool | None = None
    if time_value is not None:
        train_times = [time_value(item) for item in train]
        validation_times = [time_value(item) for item in validation]
        time_order_valid = max(train_times) <= min(validation_times)
    passed = (
        not identity_overlap and not content_overlap and not group_overlap
        and time_order_valid is not False
    )
    fingerprint = hashlib.sha256(_canonical({
        "seed": seed, "strategy": strategy,
        "train": sorted(train_content), "validation": sorted(validation_content),
    }).encode("utf-8")).hexdigest()
    return {
        "status": "passed" if passed else "failed",
        "strategy": strategy, "seed": int(seed),
        "train_count": len(train), "validation_count": len(validation),
        "identity_check": identity_check, "identity_overlap": len(identity_overlap),
        "content_hash_check": "passed" if not content_overlap else "failed",
        "content_hash_overlap": len(content_overlap),
        "group_check": group_check, "group_overlap": len(group_overlap),
        "time_order_valid": time_order_valid,
        "split_fingerprint": fingerprint,
    }
