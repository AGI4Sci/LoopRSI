#!/usr/bin/env python3
"""Fail-closed archive gate and failure taxonomy for controller trials."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RETRYABLE_FAILURES = {
    "api_timeout", "api_rate_limited", "rjob_submit_transient", "rjob_log_unavailable",
}


def classify_failure(record: dict[str, Any]) -> dict[str, Any]:
    """Return a stable, domain-agnostic failure class and retry policy."""
    status = str(record.get("status", "failed")).lower()
    text = " ".join(str(record.get(key, "")) for key in ("error", "reason", "result_contract_error")).lower()
    if status == "timeout":
        failure_class = "execution_timeout"
    elif "phenotype" in text or "activation" in text:
        failure_class = "method_inactive"
    elif "protocol" in text or "fixed_context" in text or "dataset" in text or "seed" in text:
        failure_class = "protocol_drift"
    elif "standard_result" in text or (
        "result" in text and ("missing" in text or "contract" in text or "json" in text)
    ):
        failure_class = "result_invalid"
    elif "acceptance_criteria_failed" in text:
        failure_class = "below_baseline"
    elif "quota" in text or "enqueue" in text or "temporar" in text:
        failure_class = "rjob_submit_transient"
    elif status in {"skipped", "simulated"}:
        failure_class = "not_executed"
    else:
        failure_class = "training_failed"
    bucket = {
        "training_failed": "failed_train",
        "below_baseline": "rejected",
    }.get(failure_class, "invalid")
    return {
        "failure_class": failure_class,
        "outcome": {
            "training_failed": "training_failed",
            "result_invalid": "result_invalid",
            "method_inactive": "method_inactive",
            "below_baseline": "below_acceptance",
        }.get(failure_class, failure_class),
        "retryable": failure_class in RETRYABLE_FAILURES,
        "archive_bucket": bucket,
    }


def archive_decision(
    record: dict[str, Any], standardized: dict[str, Any] | None,
    *, expected_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide archive eligibility without understanding domain metric names."""
    reasons: list[str] = []
    if record.get("status") != "ok":
        reasons.append("execution_status_not_ok")
    if record.get("result_contract_status") != "ok" or not isinstance(standardized, dict):
        reasons.append("standard_result_missing_or_invalid")
    if isinstance(standardized, dict):
        if standardized.get("status") != "ok":
            reasons.append("standard_result_status_not_ok")
        if (standardized.get("resource_usage") or {}).get("within_budget") is not True:
            reasons.append("budget_not_proven")
        protocol = standardized.get("protocol") or {}
        for key, wanted in (expected_context or {}).items():
            if wanted is not None and protocol.get(key) != wanted:
                reasons.append(f"fixed_context_mismatch:{key}")
        features = standardized.get("features")
        if isinstance(features, dict) and features.get("status") == "invalid":
            reasons.append("feature_activation_not_proven")
    acceptance = record.get("acceptance")
    if isinstance(acceptance, dict) and acceptance.get("passed") is not True:
        reasons.append("acceptance_criteria_failed")
    if reasons:
        failure = classify_failure({**record, "error": " ".join(reasons)})
        return {"eligible": False, "decision": "reject", "reasons": reasons, **failure}
    return {
        "eligible": True, "decision": "accept", "reasons": [],
        "failure_class": None, "outcome": "accepted", "retryable": False,
        "archive_bucket": "accepted",
    }


def load_standardized(record: dict[str, Any]) -> dict[str, Any] | None:
    path = record.get("standardized_result")
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
