#!/usr/bin/env python3
"""Select trial results using only result/v2 metadata.

This module deliberately does not contain task names, metric names, or domain
thresholds. All semantics come from metric role/direction and protocol fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task_contract import ContractError, SCHEMA_DIR, validate_schema, write_json


def primary_record(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    primaries = [
        (name, record) for name, record in result.get("metrics", {}).items()
        if isinstance(record, dict) and record.get("role") == "primary"
    ]
    if len(primaries) != 1:
        raise ContractError("result must contain exactly one primary metric")
    return primaries[0]


def eligibility(
    result: dict[str, Any],
    require_baseline: bool = True,
    require_multiseed: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if result.get("status") != "ok":
        reasons.append("status_not_ok")
    try:
        _, primary = primary_record(result)
    except ContractError:
        return False, ["invalid_primary_metadata"]
    if require_baseline and primary.get("beats_baseline") is not True:
        reasons.append("primary_did_not_prove_baseline_improvement")
    constraints = [
        record for record in result.get("metrics", {}).values()
        if isinstance(record, dict) and record.get("role") == "constraint"
    ]
    if any(record.get("constraint_satisfied") is not True for record in constraints):
        reasons.append("constraint_not_satisfied_or_unknown")
    if (result.get("resource_usage") or {}).get("within_budget") is not True:
        reasons.append("budget_not_satisfied_or_unknown")
    stability = ((result.get("protocol") or {}).get("stability") or {})
    if require_multiseed and stability.get("status") != "multi_seed":
        reasons.append("multi_seed_stability_required")
    return not reasons, reasons


def select_results(
    results: list[dict[str, Any]],
    require_baseline: bool = True,
    require_multiseed: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    for result in results:
        validate_schema(result, SCHEMA_DIR / "result_v2.schema.json")
        eligible, reasons = eligibility(result, require_baseline, require_multiseed)
        decisions.append({
            "task": result["task"], "trial_id": result["trial_id"],
            "eligible": eligible, "reasons": reasons,
        })
        if eligible:
            grouped.setdefault(result["task"], []).append(result)

    winners: dict[str, Any] = {}
    for task, candidates in grouped.items():
        directions = {primary_record(candidate)[1]["direction"] for candidate in candidates}
        if len(directions) != 1:
            raise ContractError(f"eligible results for task {task!r} have inconsistent directions")
        direction = directions.pop()
        winner = sorted(
            candidates,
            key=lambda candidate: primary_record(candidate)[1]["value"],
            reverse=direction == "maximize",
        )[0]
        metric_name, metric = primary_record(winner)
        winners[task] = {
            "trial_id": winner["trial_id"],
            "primary_metric": metric_name,
            "value": metric["value"],
            "direction": direction,
            "beats_baseline": metric["beats_baseline"],
            "within_budget": winner["resource_usage"]["within_budget"],
            "stability": winner["protocol"]["stability"],
        }
    return {
        "status": "ok",
        "policy": {
            "require_baseline": require_baseline,
            "require_multiseed": require_multiseed,
            "constraint_policy": "all constraint_satisfied metadata must be true",
            "budget_policy": "within_budget metadata must be true",
        },
        "winners": winners,
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select result/v2 trials without domain knowledge")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-missing-baseline", action="store_true")
    parser.add_argument("--require-multiseed", action="store_true")
    args = parser.parse_args()
    try:
        results = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
        output = select_results(
            results,
            require_baseline=not args.allow_missing_baseline,
            require_multiseed=args.require_multiseed,
        )
        write_json(args.out, output)
        print(args.out)
        return 0
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
