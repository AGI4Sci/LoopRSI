#!/usr/bin/env python3
"""Create reproducibility snapshots and run the OMNI-AR stability acceptance suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from task_contract import find_repository_root, load_task_spec, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    return {
        "command": command, "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout": proc.stdout[-12000:], "stderr": proc.stderr[-12000:],
        "passed": proc.returncode == 0,
    }


def git_output(repository: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repository, text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def working_tree_fingerprint(repository: Path) -> dict[str, Any]:
    porcelain = git_output(repository, "status", "--porcelain=v1", "-z")
    entries = [entry for entry in porcelain.split("\0") if entry]
    digest = hashlib.sha256()
    for entry in sorted(entries):
        digest.update(entry.encode("utf-8", errors="surrogateescape"))
        raw_path = entry[3:].split(" -> ")[-1]
        path = repository / raw_path
        if path.is_file():
            digest.update(sha256(path).encode())
    return {
        "clean": not entries, "changed_entry_count": len(entries),
        "content_fingerprint": digest.hexdigest(),
        "note": "A Git tag is safe only after clean=true; this fingerprint preserves the current dirty snapshot identity.",
    }


def dataset_snapshot(repository: Path, spec: dict[str, Any]) -> dict[str, Any]:
    from datasets.registry import DatasetRegistry

    data = spec.get("data") or {}
    ref = str(data.get("dataset_ref", ""))
    if not ref:
        return {"dataset_ref": None, "status": "not_configured"}
    registry = DatasetRegistry(repository / "datasets")
    artifact = (data.get("bindings") or {}).get("runtime", {}).get("artifact") or data.get("default_artifact")
    binding = registry.binding(ref, artifact=artifact)
    checks = []
    manifest = registry.manifest(ref)
    for resolved in binding["resolved"]["artifacts"]:
        path = Path(resolved["path"])
        definition = (manifest.get("artifacts") or {}).get(resolved["name"], {})
        expected_size, expected_hash = definition.get("size_bytes"), definition.get("sha256")
        checks.append({
            "artifact": resolved["name"], "path": str(path), "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "expected_size_bytes": expected_size,
            "size_matches": path.is_file() and (expected_size is None or path.stat().st_size == expected_size),
            "sha256": sha256(path) if path.is_file() and expected_hash else None,
            "expected_sha256": expected_hash,
        })
        checks[-1]["hash_matches"] = expected_hash is None or checks[-1]["sha256"] == expected_hash
    return {
        "dataset_ref": ref, "artifact": artifact,
        "manifest": str(registry.manifest_path(ref)),
        "manifest_sha256": binding["manifest_sha256"],
        "checks": checks,
        "verified": bool(checks) and all(row["exists"] and row["size_matches"] and row["hash_matches"] for row in checks),
    }


def create_snapshot(task_path: Path) -> dict[str, Any]:
    task_path = task_path.resolve()
    spec = load_task_spec(task_path)
    repository = find_repository_root(task_path)
    important = [
        task_path, repository / spec["adapter"]["module"],
        repository / "controller_bash/configs/execution.env",
        task_path.with_name("regression_suite.yaml"),
    ]
    markers = sorted(task_path.parent.glob("*activation_verified.json"))
    return {
        "schema_version": "omni-ar-reproducibility-snapshot/v1",
        "created_at_unix": int(time.time()),
        "task": spec["task"]["name"],
        "git": {
            "commit": git_output(repository, "rev-parse", "HEAD"),
            "branch": git_output(repository, "branch", "--show-current"),
            "suggested_tag": f"stability-{spec['task']['name']}-20260819",
            "working_tree": working_tree_fingerprint(repository),
        },
        "runtime": {
            "python_executable": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "files": {str(path.relative_to(repository)): sha256(path) for path in important if path.is_file()},
        "activation_markers": {str(path.relative_to(repository)): sha256(path) for path in markers},
        "dataset": dataset_snapshot(repository, spec),
        "fixed_trial_defaults": spec["adapter"]["trial_defaults"],
        "resource_limits": spec.get("resources") or {},
    }


def clean_environment(repository: Path) -> dict[str, str]:
    keep = {"PATH", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update({
        "PYTHONPATH": str(repository), "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": tempfile.mkdtemp(prefix="omni-ar-clean-tmp-"),
        "OMNI_AR_PYTHON": sys.executable,
    })
    return env


def run_checks(task_path: Path, output_dir: Path) -> dict[str, Any]:
    task_path, output_dir = task_path.resolve(), output_dir.resolve()
    spec = load_task_spec(task_path)
    repository = find_repository_root(task_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = create_snapshot(task_path)
    write_json(output_dir / "reproducibility_snapshot.json", snapshot)
    python = sys.executable
    checks = {
        "cpu_unit_tests": run([
            python, "-m", "unittest", "discover", "-s", "controller_bash/tests", "-p", "test_*.py",
        ], cwd=repository),
        "adapter_unit_tests": run([
            python, "-m", "unittest", "discover", "-s", "tasks/tests", "-p", "test_*.py",
        ], cwd=repository),
        "initialization_unit_tests": run([
            python, "-m", "unittest", "discover", "-s", "omni_ar/tests", "-p", "test_*.py",
        ], cwd=repository),
        "compileall": run([
            python, "-m", "compileall", "-q", "controller_bash", "datasets", "omni_ar", "tasks",
        ], cwd=repository),
        "controller_plumbing": run([
            python, "controller_bash/scripts/run_research.py", "--task", str(task_path),
            "--execution-mode", "rjob_dry_run", "--output-dir", str(output_dir / "rjob_dry_run"),
        ], cwd=repository),
        "clean_environment_task_contract": run([
            python, "controller_bash/scripts/task_contract.py", "validate-task", "--task-spec", str(task_path),
        ], cwd=repository, env=clean_environment(repository)),
        "clean_environment_dataset_validation": run([
            python, "-m", "datasets.datasetctl", "validate", "--dataset", str((spec.get("data") or {}).get("dataset_ref")),
        ], cwd=repository, env=clean_environment(repository)),
    }
    passed = snapshot["dataset"].get("verified", True) and all(row["passed"] for row in checks.values())
    summary = {
        "schema_version": "omni-ar-stability-run/v1", "status": "ok" if passed else "failed",
        "task": spec["task"]["name"], "passed": passed,
        "snapshot": str(output_dir / "reproducibility_snapshot.json"), "checks": checks,
        "gpu_execution": "not_run_by_this_command; use stability/profiles.yaml in order",
    }
    write_json(output_dir / "stability_summary.json", summary)
    (output_dir / "stability_report.md").write_text(render_report(summary, snapshot), encoding="utf-8")
    return summary


def render_report(summary: dict[str, Any], snapshot: dict[str, Any]) -> str:
    rows = ["| Check | Passed | Seconds |", "|---|---:|---:|"]
    rows += [f"| {name} | {'yes' if row['passed'] else 'no'} | {row['elapsed_seconds']} |" for name, row in summary["checks"].items()]
    return "\n".join([
        "# OMNI-AR stability report", "", f"Overall: **{summary['status']}**", "", *rows, "",
        f"Git commit: `{snapshot['git']['commit']}`", "",
        f"Working tree clean: `{snapshot['git']['working_tree']['clean']}`", "",
        f"Dataset verified: `{snapshot['dataset'].get('verified')}`", "",
        "GPU tiers are intentionally separate and must be run in the order defined in `stability/profiles.yaml`.", "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--task", required=True, type=Path)
    snap.add_argument("--output", required=True, type=Path)
    check = sub.add_parser("check")
    check.add_argument("--task", required=True, type=Path)
    check.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        write_json(args.output, create_snapshot(args.task))
        print(args.output)
        return 0
    summary = run_checks(args.task, args.output_dir)
    print(args.output_dir / "stability_summary.json")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
