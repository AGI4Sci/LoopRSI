"""Strict evaluation authority records for VCC25 candidate selection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_AUTHORITIES = frozenset({"search_validation", "final_official"})
REWARD_METRIC_SOURCE = "real"


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


@dataclass(frozen=True)
class EvaluationRecord:
    authority: str
    candidate_pcc: float
    candidate_hash: str
    full_validation: bool
    metric_source: str
    selection_feedback_allowed: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        authority = value.get("authority")
        if authority not in ALLOWED_AUTHORITIES:
            raise ValueError(
                "authority must be one of: " + ", ".join(sorted(ALLOWED_AUTHORITIES))
            )
        candidate_hash = value.get("candidate_hash")
        if not isinstance(candidate_hash, str) or not candidate_hash.strip():
            raise ValueError("candidate_hash is required")
        full_validation = value.get("full_validation")
        if not isinstance(full_validation, bool):
            raise ValueError("full_validation must be boolean")
        metric_source = value.get("metric_source")
        if not isinstance(metric_source, str) or not metric_source.strip():
            raise ValueError("metric_source is required")
        feedback = value.get("selection_feedback_allowed")
        if not isinstance(feedback, bool):
            raise ValueError("selection_feedback_allowed must be boolean")
        if authority == "final_official" and feedback:
            raise ValueError("final_official requires selection_feedback_allowed=false")
        return cls(
            authority=str(authority),
            candidate_pcc=_finite_float("candidate_pcc", value.get("candidate_pcc")),
            candidate_hash=candidate_hash.strip(),
            full_validation=full_validation,
            metric_source=metric_source.strip(),
            selection_feedback_allowed=feedback,
        )


def promotion_eligible(record: EvaluationRecord) -> bool:
    """Return whether the record has the contract needed for promotion review."""

    return (
        record.authority == "search_validation"
        and record.full_validation
        and record.metric_source == REWARD_METRIC_SOURCE
    )


def validation_reward(record: EvaluationRecord, baseline: float) -> float | None:
    """Return candidate-minus-baseline only for full real validation evidence."""

    baseline_pcc = _finite_float("baseline", baseline)
    if not promotion_eligible(record):
        return None
    return record.candidate_pcc - baseline_pcc
