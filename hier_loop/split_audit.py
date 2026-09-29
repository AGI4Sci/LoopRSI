"""Target-identifier-only split audit for the VCC25 search contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class SplitLeakageError(ValueError):
    """Raised when target identifiers overlap across declared splits."""


def _targets(values: Iterable[str]) -> frozenset[str]:
    result = frozenset(str(value).strip() for value in values if str(value).strip())
    if not result:
        raise SplitLeakageError("each target split must be non-empty")
    return result


def _hashes(values: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values))


@dataclass(frozen=True)
class SplitAudit:
    train_target_sha256: tuple[str, ...]
    validation_target_sha256: tuple[str, ...]
    test_target_sha256: tuple[str, ...]

    @classmethod
    def from_targets(
        cls,
        *,
        train: Iterable[str],
        validation: Iterable[str],
        test: Iterable[str],
    ) -> "SplitAudit":
        splits = {
            "train": _targets(train),
            "validation": _targets(validation),
            "test": _targets(test),
        }
        pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
        overlapping = {
            f"{left}:{right}": sorted(splits[left] & splits[right])
            for left, right in pairs
            if splits[left] & splits[right]
        }
        if overlapping:
            raise SplitLeakageError(
                "target split overlap detected: "
                + ", ".join(f"{pair}={len(values)}" for pair, values in sorted(overlapping.items()))
            )
        return cls(
            train_target_sha256=_hashes(splits["train"]),
            validation_target_sha256=_hashes(splits["validation"]),
            test_target_sha256=_hashes(splits["test"]),
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": "vcc25-split-audit/v1",
            "contents": "target_identifiers_only",
            "target_splits_disjoint": True,
            "splits": {
                "train": {
                    "count": len(self.train_target_sha256),
                    "target_sha256": list(self.train_target_sha256),
                },
                "validation": {
                    "count": len(self.validation_target_sha256),
                    "target_sha256": list(self.validation_target_sha256),
                },
                "test": {
                    "count": len(self.test_target_sha256),
                    "target_sha256": list(self.test_target_sha256),
                },
            },
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination
