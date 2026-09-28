#!/usr/bin/env python3
"""Migrate completed legacy trials that were mislabeled as training failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def migrate_gate(gate: Any, *, status: Any = None) -> bool:
    if not isinstance(gate, dict):
        return False
    reasons = set(gate.get("reasons") or [])
    if reasons != {"acceptance_criteria_failed"}:
        return False
    if status not in {None, "ok"}:
        return False
    gate.update({
        "failure_class": "below_baseline", "retryable": False,
        "archive_bucket": "rejected",
    })
    return True


def migrate_non_search_entry(entry: dict[str, Any]) -> bool:
    """Remove verification/control records from the strategy failure bucket."""
    bucket = str(entry.get("bucket") or "")
    if entry.get("strategy_bucket") != "failed_train":
        return False
    gate = entry.get("archive_gate") if isinstance(entry.get("archive_gate"), dict) else {}
    if bucket == "verification":
        outcome = "activation_verified"
    elif bucket == "control":
        outcome = "non_scientific_control"
    elif bucket == "rejected" and (
        gate.get("failure_class") == "below_baseline"
        or "acceptance_criteria_failed" in set(gate.get("reasons") or [])
    ):
        outcome = "below_acceptance"
    else:
        return False
    entry["strategy_bucket"] = outcome
    entry["outcome"] = outcome
    metadata = entry.get("strategy_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        entry["strategy_metadata"] = metadata
    metadata.update({
        "bucket": outcome,
        "excluded_from_search_archive": True,
    })
    metadata.pop("omniepic_failure_mode", None)
    return True


def migrate_project(project_dir: Path, *, apply: bool) -> dict[str, Any]:
    project_dir = project_dir.resolve()
    archive_path = project_dir / "records/archive.json"
    lineage_path = project_dir / "records/lineage.json"
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    archive = _load(archive_path)
    changed_ids: list[str] = []
    for entry in archive.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        changed = migrate_non_search_entry(entry)
        if migrate_gate(entry.get("archive_gate"), status="ok"):
            entry["bucket"] = "rejected"
            entry["score"] = None
            changed = True
        if changed:
            changed_ids.append(str(entry.get("candidate_id")))
    changed_files: dict[str, str] = {}
    traces_changed = 0
    for trace_path in sorted((project_dir / "rounds").rglob("execution_trace.json")):
        trace = _load(trace_path)
        changed = False
        for job in trace.get("jobs") or []:
            if isinstance(job, dict) and migrate_gate(
                job.get("archive_gate"), status=job.get("status"),
            ):
                changed = True
        if changed:
            traces_changed += 1
            if apply:
                _write(trace_path, trace)
                changed_files[str(trace_path)] = _sha(trace_path)
    lineage = _load(lineage_path) if lineage_path.is_file() else None
    if isinstance(lineage, dict):
        changed_set = set(changed_ids)
        for node in lineage.get("nodes") or []:
            if isinstance(node, dict) and str(node.get("candidate_id")) in changed_set:
                node["bucket"] = "rejected"
    if apply and changed_ids:
        _write(archive_path, archive)
        changed_files[str(archive_path)] = _sha(archive_path)
        if isinstance(lineage, dict):
            _write(lineage_path, lineage)
            changed_files[str(lineage_path)] = _sha(lineage_path)
    record = {
        "schema_version": "omni-ar-category-migration/v1",
        "status": "applied" if apply else "dry_run",
        "project_dir": str(project_dir), "candidate_ids": changed_ids,
        "candidate_count": len(changed_ids), "execution_traces_changed": traces_changed,
        "changed_files": changed_files,
        "mapping": {
            "below_acceptance": {
                "from": {"failure_class": "train_failed", "archive_bucket": "failed_train"},
                "to": {"failure_class": "below_baseline", "archive_bucket": "rejected"},
            },
            "non_search_evidence": {
                "from": {"strategy_bucket": "failed_train"},
                "to": {"strategy_bucket": "activation_verified_or_control", "selectable": False},
            },
        },
    }
    if apply:
        _write(project_dir / "records/category_migration.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        migrate_project(args.project_dir, apply=args.apply), ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
