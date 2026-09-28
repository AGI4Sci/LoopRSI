from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ProjectRecords:
    """One auditable ledger for planning, code routing, trials and lineage."""

    schema_version = "omni-ar-project-records/v1"

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.root = self.project_dir / "records"
        self.rounds_dir = self.root / "rounds"
        self.project_path = self.root / "project.json"
        self.archive_path = self.root / "archive.json"
        self.lineage_path = self.root / "lineage.json"
        self.events_path = self.root / "events.jsonl"
        self.stages_path = self.root / "stages.json"

    def initialize(self, metadata: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.project_path.exists():
            write_json(self.project_path, {
                "schema_version": self.schema_version,
                "status": "running",
                "created_at": now(),
                "updated_at": now(),
                "metadata": metadata,
                "termination": None,
                "final_package": None,
            })
        if not self.archive_path.exists():
            write_json(self.archive_path, {
                "schema_version": "omni-ar-archive/v1",
                "entries": [],
            })
        if not self.lineage_path.exists():
            write_json(self.lineage_path, {
                "schema_version": "omni-ar-lineage/v1",
                "nodes": [], "edges": [],
            })
        if not self.stages_path.exists():
            write_json(self.stages_path, {
                "schema_version": "omni-ar-stages/v1", "stages": {},
            })
        else:
            project = json.loads(self.project_path.read_text(encoding="utf-8"))
            project.update({"status": "running", "updated_at": now(), "termination": None})
            write_json(self.project_path, project)
        self.event("project_initialized", metadata)

    def stage(self, name: str, status: str, **details: Any) -> None:
        payload = json.loads(self.stages_path.read_text(encoding="utf-8"))
        previous = (payload.get("stages") or {}).get(name) or {}
        value = {**previous, **details, "status": status, "updated_at": now()}
        if status == "running":
            value.pop("finished_at", None)
            value.pop("error", None)
            value["attempts"] = int(previous.get("attempts", 0)) + 1
            value.setdefault("started_at", now())
        if status in {"completed", "failed", "skipped"}:
            value["finished_at"] = now()
        payload.setdefault("stages", {})[name] = value
        write_json(self.stages_path, payload)
        self.event("stage_updated", {"name": name, **value})

    def completed_rounds(self) -> int:
        count = 0
        while True:
            path = self.rounds_dir / f"round_{count:03d}.json"
            if not path.is_file():
                break
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("status") != "completed":
                break
            count += 1
        return count

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "schema_version": "omni-ar-event/v1",
                "timestamp": now(), "kind": kind, "payload": payload,
            }, ensure_ascii=False) + "\n")

    def archive(self) -> list[dict[str, Any]]:
        if not self.archive_path.exists():
            return []
        return json.loads(self.archive_path.read_text(encoding="utf-8")).get("entries", [])

    def add_candidate(self, candidate: dict[str, Any]) -> None:
        archive = self.archive()
        archive = [item for item in archive if item.get("candidate_id") != candidate["candidate_id"]]
        archive.append(candidate)
        archive.sort(key=lambda item: (int(item.get("round", 0)), str(item.get("candidate_id"))))
        write_json(self.archive_path, {
            "schema_version": "omni-ar-archive/v1", "entries": archive,
        })
        lineage = json.loads(self.lineage_path.read_text(encoding="utf-8"))
        lineage["nodes"] = [
            item for item in lineage.get("nodes", [])
            if item.get("candidate_id") != candidate["candidate_id"]
        ]
        lineage["nodes"].append({
            key: candidate.get(key) for key in (
                "candidate_id", "round", "generation", "bucket", "score",
                "proposal_id", "proposal_path", "trial_id", "route",
                "candidate_worktree", "code_patch", "code_parent_id",
                "code_snapshot", "changed_paths",
                "dataset_lineage", "parameters", "control_roles",
            )
        })
        lineage["edges"] = [
            edge for edge in lineage.get("edges", [])
            if edge.get("child") != candidate["candidate_id"]
        ]
        lineage["edges"].extend({
            "parent": parent, "child": candidate["candidate_id"],
        } for parent in candidate.get("parent_ids", []))
        write_json(self.lineage_path, lineage)
        self.event("candidate_recorded", candidate)

    def record_round(self, round_number: int, value: dict[str, Any]) -> Path:
        path = self.rounds_dir / f"round_{round_number:03d}.json"
        write_json(path, value)
        self.event("round_completed", {
            "round": round_number, "record": str(path),
            "status": value.get("status"),
        })
        return path

    def finalize(self, termination: dict[str, Any], final_package: Path) -> None:
        project = json.loads(self.project_path.read_text(encoding="utf-8"))
        project.update({
            "status": "failed" if termination.get("error") else "completed",
            "updated_at": now(),
            "termination": termination,
            "final_package": str(final_package),
        })
        write_json(self.project_path, project)
        self.event("project_finalized", {
            "termination": termination, "final_package": str(final_package),
        })

    def manifest(self) -> dict[str, Any]:
        files = [
            path for path in self.root.rglob("*")
            if path.is_file() and path != self.events_path
        ]
        return {
            "schema_version": self.schema_version,
            "root": str(self.root),
            "files": [{
                "path": str(path.relative_to(self.project_dir)),
                "sha256": sha256(path), "bytes": path.stat().st_size,
            } for path in sorted(files)],
        }
