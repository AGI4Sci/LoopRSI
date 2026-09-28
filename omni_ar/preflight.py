from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .project_records import write_json


class PreflightError(RuntimeError):
    pass


def gpu_budget_allowed(task_gpu_count: int, authorized_gpu_count: int, legacy: bool = False) -> bool:
    """CPU-only tasks require no GPU authorization or CUDA probe."""
    return legacy or (task_gpu_count == 0 and authorized_gpu_count == 0) or authorized_gpu_count >= task_gpu_count


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def run_preflight(
    repository: Path, rough: dict[str, Any], task_spec: Path, output: Path,
    *, planning_mode: str, execution_mode: str, coding_mode: str,
    project_dir: Path | None = None,
) -> Path:
    """Validate everything that can be checked without starting research work."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "ok" if ok else "failed", "detail": detail})

    policy = rough.get("execution_policy") or {}
    legacy = rough.get("schema_version") == "omni-ar-rough-idea/v1"
    add("rough_idea_confirmed", rough.get("confirmation", {}).get("status") == "confirmed", "QA confirmation")
    add("task_spec_exists", task_spec.is_file(), str(task_spec))
    expected_task_sha = (rough.get("provenance") or {}).get("task_spec_sha256")
    add("task_spec_unchanged", not expected_task_sha or _sha256(task_spec) == expected_task_sha, "task specification hash")

    spec = yaml.safe_load(task_spec.read_text(encoding="utf-8"))
    task_gpu_count = int((spec.get("resources") or {}).get("gpu_count", 0))
    dataset_ref = str((spec.get("data") or {}).get("dataset_ref") or "")
    try:
        from datasets.registry import DatasetRegistry
        registry = DatasetRegistry(repository / "datasets")
        manifest = registry.manifest_path(dataset_ref)
        dataset_ok = bool(dataset_ref) and manifest.is_file()
        expected_manifest_sha = (rough.get("dataset") or {}).get("manifest_sha256")
        dataset_ok = dataset_ok and (not expected_manifest_sha or _sha256(manifest) == expected_manifest_sha)
        add("dataset_binding", dataset_ok, dataset_ref or "missing dataset_ref")
    except Exception as exc:  # noqa: BLE001
        add("dataset_binding", False, f"{type(exc).__name__}: {exc}")

    if planning_mode == "live":
        env = {**_env_file(repository / "Heuresis_PJLAB-boyue/.env"), **os.environ}
        has_key = bool(env.get("BOYUE_API_KEY") or env.get("BOYUE_API_KEYS"))
        authorized = bool(policy.get("external_services")) or legacy
        add("external_services_authorized", authorized, "QA execution policy")
        add("boyue_credentials_configured", has_key, "credential presence only; value not recorded")

    if execution_mode in {"rjob", "rjob_dry_run"}:
        project = (project_dir or output.parent.parent).resolve()
        inside_repository = project == repository.resolve() or repository.resolve() in project.parents
        add("rjob_output_location", inside_repository, str(project))

    if execution_mode == "rjob":
        defaults = _env_file(repository / "controller_bash/configs/execution.env")
        merged = {**defaults, **os.environ}
        authorized = bool(policy.get("submit_rjob")) or legacy
        add("rjob_authorized", authorized, "QA execution policy")
        add("rjob_cli", shutil.which("rjob") is not None, "rjob executable")
        shared = Path(merged.get("OMNI_AR_SHARED_ROOT", "")) if merged.get("OMNI_AR_SHARED_ROOT") else None
        add("shared_storage", bool(shared and shared.is_dir() and os.access(shared, os.W_OK)), str(shared or "missing"))
        add("rjob_namespace", bool(merged.get("RJOB_NAMESPACE") and merged.get("RJOB_CHARGED_GROUP")), "namespace and charged group")
        authorized_gpu_count = int(policy.get("max_gpus", 0))
        add(
            "gpu_budget",
            gpu_budget_allowed(task_gpu_count, authorized_gpu_count, legacy),
            f"task requests {task_gpu_count} GPU(s); QA authorizes {authorized_gpu_count}",
        )

        # Submit one small CUDA probe only after all static rjob checks pass.
        if task_gpu_count == 0:
            add("gpu_probe", True, "not required for CPU-only task")
        elif not [item for item in checks if item["status"] == "failed"]:
            probe_dir = (project_dir or output.parent.parent) / "preflight" / "gpu_probe"
            probe_record = probe_dir / "gpu_probe.execution.json"
            reusable = False
            if probe_record.is_file():
                try:
                    reusable = json.loads(probe_record.read_text(encoding="utf-8")).get("status") == "ok"
                except (OSError, json.JSONDecodeError):
                    reusable = False
            if not reusable:
                probe_dir.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(
                    [sys.executable, str(repository / "controller_bash/scripts/run_gpu_probe.py"),
                     "--output-dir", str(probe_dir)],
                    cwd=repository, text=True, capture_output=True, check=False,
                )
                (probe_dir / "preflight.stdout.log").write_text(proc.stdout, encoding="utf-8")
                (probe_dir / "preflight.stderr.log").write_text(proc.stderr, encoding="utf-8")
                reusable = proc.returncode == 0 and probe_record.is_file()
            add("gpu_probe", reusable, str(probe_record))

    if coding_mode == "apply":
        authorized = bool(policy.get("code_changes")) or legacy
        add("code_changes_authorized", authorized, "QA execution policy")
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, check=False)
        add("git_snapshot_base", git.returncode == 0, "current Git HEAD")

    failed = [item for item in checks if item["status"] != "ok"]
    payload = {
        "schema_version": "omni-ar-preflight/v1",
        "status": "ok" if not failed else "failed",
        "planning_mode": planning_mode, "execution_mode": execution_mode,
        "coding_mode": coding_mode, "checks": checks,
        "blockers": [item["name"] for item in failed],
    }
    write_json(output, payload)
    if failed:
        raise PreflightError("startup checks failed: " + ", ".join(payload["blockers"]))
    return output
