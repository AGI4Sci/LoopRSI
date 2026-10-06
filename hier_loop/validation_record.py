"""Method-agnostic, auditable VCC25 search-validation outcomes."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "vcc25.validation-record/v1"


def make_validation_record(*, method: str, variant: str, seed: int,
                           raw: Mapping[str, Any], metric: str,
                           elapsed_seconds: float, gpu_requested: int = 0) -> dict[str, Any]:
    if not method or not variant or elapsed_seconds < 0:
        raise ValueError("invalid validation record identity or cost")
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
    value = metrics.get(metric)
    valid_score = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    source = raw.get("evaluation_authority") or raw.get("authority") or (
        "search_validation" if method == "official_h1" else "unverified"
    )
    passed = raw.get("status") in {"ok", "pass"} and valid_score and source == "search_validation"
    scope = raw.get("protocol") if isinstance(raw.get("protocol"), Mapping) else {}
    safe = scope.get("test_expression_used_for_selection") is False or raw.get("test_expression_read") is False
    passed = passed and safe
    return {
        "schema_version": SCHEMA, "method": method, "variant": variant, "seed": int(seed),
        "status": "passed" if passed else "failed",
        "evaluation_authority": source if passed else "unverified",
        "metric_name": metric, "metric_value": float(value) if passed else None,
        "metrics": {str(k): float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)} if passed else {},
        "cost": {"wall_seconds": float(elapsed_seconds), "gpu_requested": int(gpu_requested)},
        "error": None if passed else str(raw.get("error") or "missing trusted finite validation metric or split audit")[:2000],
        "test_expression_read": False if passed else raw.get("test_expression_read"),
        "selection_feedback_allowed": bool(passed),
    }


def write_validation_record(directory: str | Path, stem: str, record: Mapping[str, Any]) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{stem}.validation.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(record), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path
