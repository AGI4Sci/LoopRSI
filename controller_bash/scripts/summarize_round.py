#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metric(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "missing"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable", "error": str(exc)}


def extract_primary_metric(metric: dict[str, Any]) -> tuple[str | None, float | int | None, str | None]:
    if metric.get("schema_version") == "omni-ar-result/v2":
        records = metric.get("metrics") or {}
        for name, record in records.items():
            if isinstance(record, dict) and record.get("role") == "primary":
                value = record.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return name, value, record.get("direction")
    primary = metric.get("primary_metric")
    if isinstance(primary, dict):
        value = primary.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return primary.get("name"), value, primary.get("direction")
    if isinstance(primary, (int, float)) and not isinstance(primary, bool):
        return "primary_metric", primary, None
    for key in ("score", "metric"):
        value = metric.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, value, None
    return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--suggestion", type=Path, required=True)
    parser.add_argument("--trial-results", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    suggestion = json.loads(args.suggestion.read_text(encoding="utf-8")) if args.suggestion.exists() else {}
    trial_results = json.loads(args.trial_results.read_text(encoding="utf-8")) if args.trial_results.exists() else {"results": []}

    rows = []
    for result in trial_results.get("results", []):
        metric = load_metric(result.get("standardized_result") or result.get("output", ""))
        metric_name, metric_value, metric_direction = extract_primary_metric(metric)
        rows.append(
            {
                "trial_index": result.get("trial_index"),
                "status": result.get("status"),
                "name": (result.get("trial") or {}).get("name"),
                "purpose": (result.get("trial") or {}).get("purpose"),
                "primary_metric_name": metric_name,
                "primary_metric": metric_value,
                "primary_metric_direction": metric_direction,
                "metric_status": metric.get("status") if isinstance(metric, dict) else None,
                "beats_baseline": next((
                    value.get("beats_baseline")
                    for value in (metric.get("metrics") or {}).values()
                    if isinstance(value, dict) and value.get("role") == "primary"
                ), None) if isinstance(metric, dict) else None,
                "within_budget": (metric.get("resource_usage") or {}).get("within_budget")
                if isinstance(metric, dict) else None,
                "stability": ((metric.get("protocol") or {}).get("stability") or {}).get("status")
                if isinstance(metric, dict) else None,
                "acceptance_passed": (result.get("acceptance") or {}).get("passed"),
                "output": result.get("output"),
            }
        )

    summary = {
        "round": args.round,
        "verdict": suggestion.get("verdict"),
        "num_trials": trial_results.get("num_trials", 0),
        "trial_mode": trial_results.get("mode"),
        "rows": rows,
        "priority_order": suggestion.get("priority_order", []),
        "risks": suggestion.get("risks", []),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Heuresis-Codex Round {args.round}",
        "",
        f"- Verdict: {summary['verdict'] or 'n/a'}",
        f"- Trial mode: `{summary['trial_mode']}`",
        f"- Trials: `{summary['num_trials']}`",
        "",
        "| trial | status | name | primary_metric | direction | beats baseline | budget | stability | acceptance | metric_status | output |",
        "|---:|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {trial_index} | {status} | {name} | {primary_metric_name}={primary_metric} | {primary_metric_direction} | {beats_baseline} | {within_budget} | {stability} | {acceptance_passed} | {metric_status} | {output} |".format(
                **{k: "" if v is None else v for k, v in row.items()}
            )
        )
    if summary["priority_order"]:
        lines += ["", "## Priority", "", ", ".join(str(x) for x in summary["priority_order"])]
    if summary["risks"]:
        lines += ["", "## Risks", ""]
        lines += [f"- {risk}" for risk in summary["risks"]]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
