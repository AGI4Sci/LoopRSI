#!/usr/bin/env python3
"""Run a task-owned regression suite through the generic controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from task_contract import (
    ContractError,
    SCHEMA_DIR,
    extract_source,
    find_repository_root,
    load_json,
    load_task_adapter,
    load_task_spec,
    resolved_workspace,
    validate_proposal,
    validate_schema,
    write_json,
)


def load_env_defaults(path: Path) -> dict[str, str]:
    defaults: dict[str, str] = {}
    if not path.exists():
        return defaults
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        defaults[key.strip()] = value.strip()
    return defaults


def load_regression_suite(path: Path, task_name: str) -> dict[str, Any]:
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict):
        raise ContractError("regression suite must be a YAML mapping")
    validate_schema(suite, SCHEMA_DIR / "regression_suite.schema.json")
    if suite["task_name"] != task_name:
        raise ContractError("regression suite task_name does not match task spec")
    for index, case in enumerate(suite["cases"]):
        try:
            validate_schema(case["proposal"], SCHEMA_DIR / "experiment_proposal.schema.json")
        except ContractError as exc:
            raise ContractError(f"cases[{index}].proposal: {exc}") from exc
    return suite


def proposal_from_suite(suite: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "omni-ar-proposal/v2",
        "proposal_id": suite["suite_id"],
        "task_name": suite["task_name"],
        "round": 0,
        "verdict": "Run the frozen task regression suite without research search.",
        "evidence_gaps": [],
        "experiment_proposals": [case["proposal"] for case in suite["cases"]],
        "risks": [],
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controller_fingerprint(repository: Path) -> dict[str, str]:
    relative_paths = (
        "controller_bash/prompts/codex_apply_suggestion.md",
        "controller_bash/prompts/codex_build_prototype.md",
        "controller_bash/scripts/heuresis_suggest.py",
        "controller_bash/scripts/run_trials.py",
        "controller_bash/scripts/execution_backends.py",
        "controller_bash/scripts/task_contract.py",
    )
    return {path: sha256(repository / path) for path in relative_paths}


def replay_historical_anchors(
    suite: dict[str, Any], repository: Path
) -> dict[str, Any]:
    anchors = []
    for anchor in suite["historical_anchors"]:
        artifact = (repository / anchor["artifact"]).resolve()
        if artifact != repository and repository not in artifact.parents:
            raise ContractError(f"historical artifact escapes repository: {anchor['artifact']}")
        checks = []
        if artifact.exists():
            payload = load_json(artifact)
            for check in anchor["checks"]:
                try:
                    actual = extract_source(payload, check["source"])
                    difference = abs(float(actual) - float(check["expected"]))
                    passed = difference <= check["absolute_tolerance"]
                    error = None
                except (ContractError, TypeError, ValueError) as exc:
                    actual, difference, passed, error = None, None, False, str(exc)
                checks.append({
                    **check, "actual": actual, "absolute_difference": difference,
                    "passed": passed, "error": error,
                })
        else:
            checks.append({"passed": False, "error": f"artifact missing: {artifact}"})
        anchors.append({
            "anchor_id": anchor["anchor_id"], "artifact": str(artifact),
            "passed": bool(checks) and all(check["passed"] for check in checks),
            "checks": checks,
        })
    return {"passed": all(anchor["passed"] for anchor in anchors), "anchors": anchors}


def compiled_case_coverage(
    suite: dict[str, Any], spec: dict[str, Any], spec_path: Path
) -> dict[str, Any]:
    adapter = load_task_adapter(spec, spec_path)
    defaults = spec["adapter"]["trial_defaults"]
    rows = []
    for case in suite["cases"]:
        trial = adapter.proposal_to_trial(case["proposal"], defaults, case["case_id"])
        compiled = trial["cli_overrides"]
        mismatches = {
            key: {"expected": expected, "actual": compiled.get(key)}
            for key, expected in case["expected_parameters"].items()
            if compiled.get(key) != expected
        }
        rows.append({
            "case_id": case["case_id"], "passed": not mismatches,
            "entrypoint": trial["entrypoint"], "compiled_parameters": compiled,
            "mismatches": mismatches,
        })
    return {"passed": all(row["passed"] for row in rows), "cases": rows}


def execution_environment(
    base: dict[str, str], repository: Path, spec: dict[str, Any], spec_path: Path,
    task_name: str, mode: str,
) -> dict[str, str]:
    env = dict(base)
    project_root, implementation_dir = resolved_workspace(spec, spec_path)
    project_relative = project_root.relative_to(repository)
    implementation_relative = implementation_dir.relative_to(repository)
    sync_paths = list(dict.fromkeys(("tasks", "controller_bash", str(project_relative))))
    dataset_ref = str((spec.get("data") or {}).get("dataset_ref", ""))
    dataset_pack = None
    dataset_binding = None
    if dataset_ref:
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        from datasets.registry import DatasetRegistry

        registry = DatasetRegistry(repository / "datasets")
        runtime = spec["data"].get("bindings", {}).get("runtime", {})
        artifact = runtime.get("artifact") or spec["data"].get("default_artifact")
        split = runtime.get("split")
        dataset_binding = registry.binding(dataset_ref, artifact=artifact, split=split)
        dataset_pack = Path(dataset_binding["pack_root"])
        dataset_support = [
            "datasets/__init__.py", "datasets/adapter_contract.py", "datasets/registry.py",
            "datasets/registry.yaml", "datasets/generic_adapter.py",
            str((dataset_pack / "adapter.py").relative_to(repository)),
            str(registry.manifest_path(dataset_ref).relative_to(repository)),
            str((dataset_pack / "SKILL.md").relative_to(repository)),
        ]
        modes_file = str(registry.manifest(dataset_ref).get("usage_modes_file", "usage_modes.json"))
        dataset_support.append(str((dataset_pack / modes_file).relative_to(repository)))
        sync_paths.extend(path for path in dataset_support if (repository / path).exists())
        for resolved in dataset_binding["resolved"]["artifacts"]:
            path = Path(resolved["path"])
            sync_paths.append(str(path.relative_to(repository)))
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                sync_paths.append(str(sidecar.relative_to(repository)))
        for artifact_name in spec["data"].get("runtime_artifacts", []):
            supplemental = registry.binding(dataset_ref, artifact=str(artifact_name))
            for resolved in supplemental["resolved"]["artifacts"]:
                path = Path(resolved["path"])
                sync_paths.append(str(path.relative_to(repository)))
        sync_paths = list(dict.fromkeys(sync_paths))
    shared_root = env.get("OMNI_AR_SHARED_ROOT", "").strip()
    if mode in {"rjob", "rjob_dry_run"} and not shared_root:
        raise ContractError("OMNI_AR_SHARED_ROOT is required for rjob execution")
    env.update({
        "TASK_SPEC": str(spec_path),
        "TRIAL_EXECUTION_MODE": mode,
        "MAX_TRIALS_PER_ROUND": str((spec.get("resources") or {}).get("max_trials_per_round", 12)),
        "PROJECT_ROOT": str(project_root),
        "IMPLEMENTATION_DIR": str(implementation_dir),
        "TRIAL_WORKDIR": str(implementation_dir),
        "TRIAL_PYTHONPATH": str(repository),
        "RJOB_SYNC_SOURCE": str(repository),
        "RJOB_SYNC_PATHS": ":".join(sync_paths),
        "RJOB_SHARED_FOLDER": str(Path(shared_root) / "rjob_packages" / f"{task_name}_regression")
        if shared_root else "",
        "RJOB_FOLDER": "",
        "TRIAL_WORKDIR_IN_PACKAGE": str(implementation_relative),
        "TRIAL_PYTHONPATH_IN_PACKAGE": os.path.relpath(repository, implementation_dir),
        "COMMAND_IMPLEMENTATION_DIR_IN_PACKAGE": ".",
        "PROJECT_ROOT_IN_PACKAGE": os.path.relpath(project_root, implementation_dir),
        "DATASET_REF": dataset_ref,
        "DATASET_MANIFEST_SHA256": dataset_binding["manifest_sha256"] if dataset_binding else "",
        "DATASET_DIR": str(dataset_pack) if dataset_pack else "",
        "DATASET_DIR_IN_PACKAGE": str(dataset_pack.relative_to(repository)) if dataset_pack else "",
    })
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one task through the generic OMNI-AR controller")
    parser.add_argument("--task", required=True, type=Path, help="path to tasks/<task>/task_spec.yaml")
    parser.add_argument(
        "--execution-mode", choices=("rjob_dry_run", "rjob", "simulated", "skip"),
        default="rjob_dry_run",
    )
    parser.add_argument("--suite", type=Path, help="defaults to regression_suite.yaml beside task spec")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--execution-config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/execution.env",
    )
    args = parser.parse_args()

    try:
        spec_path = args.task.resolve()
        spec = load_task_spec(spec_path)
        repository = find_repository_root(spec_path)
        suite_path = (args.suite or spec_path.with_name("regression_suite.yaml")).resolve()
        suite = load_regression_suite(suite_path, spec["task"]["name"])
        if suite["policy"]["allow_search"]:
            raise ContractError("regression entry refuses suites with allow_search=true")
        proposal = proposal_from_suite(suite)
        validate_proposal(proposal, spec, spec_path)
        project_root, _ = resolved_workspace(spec, spec_path)
        output_dir = (args.output_dir or project_root / "regression_runs" / suite["suite_id"]).resolve()
        if args.execution_mode in {"rjob", "rjob_dry_run"} and (
            output_dir != repository and repository not in output_dir.parents
        ):
            raise ContractError("rjob output-dir must stay inside RJOB_SYNC_SOURCE/repository")
        round_dir = output_dir / "state/round_0"
        artifact_dir = output_dir / "artifacts"
        proposal_path = round_dir / "regression_proposal.json"
        results_path = round_dir / "trial_results.json"
        summary_path = output_dir / "regression_summary.json"
        write_json(proposal_path, proposal)

        before = controller_fingerprint(repository)
        defaults = load_env_defaults(args.execution_config.resolve())
        base_env = {**defaults, **os.environ}
        env = execution_environment(
            base_env, repository, spec, spec_path, spec["task"]["name"], args.execution_mode
        )
        command = [
            sys.executable, str(repository / "controller_bash/scripts/run_trials.py"),
            "--suggestion", str(proposal_path), "--round-dir", str(round_dir),
            "--artifact-dir", str(artifact_dir), "--out", str(results_path),
        ]
        proc = subprocess.run(command, cwd=repository, env=env, text=True, capture_output=True, check=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "controller.stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
        (output_dir / "controller.stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")

        coverage = compiled_case_coverage(suite, spec, spec_path)
        historical = replay_historical_anchors(suite, repository)
        after = controller_fingerprint(repository)
        prompt_core_unchanged = before == after
        trials = load_json(results_path) if results_path.exists() else {"results": []}
        expected_trials = len(suite["cases"])
        controller_passed = (
            proc.returncode == 0
            and trials.get("num_trials") == expected_trials
            and all(row.get("result_contract_status") == "ok" for row in trials.get("results", []))
        )
        passed = controller_passed and coverage["passed"] and historical["passed"] and prompt_core_unchanged
        summary = {
            "status": "ok" if passed else "failed",
            "schema_version": "omni-ar-regression-run/v1",
            "task": spec["task"]["name"],
            "task_spec": str(spec_path),
            "suite": str(suite_path),
            "suite_policy": suite["policy"],
            "execution_mode": args.execution_mode,
            "dataset": {
                "ref": env.get("DATASET_REF"),
                "manifest_sha256": env.get("DATASET_MANIFEST_SHA256"),
                "pack_root": env.get("DATASET_DIR"),
            },
            "controller": {
                "entrypoint": str(repository / "controller_bash/scripts/run_trials.py"),
                "returncode": proc.returncode,
                "passed": controller_passed,
                "num_trials": trials.get("num_trials"),
            },
            "compiled_case_coverage": coverage,
            "historical_replay": historical,
            "generic_prompt_and_core_unchanged": prompt_core_unchanged,
            "controller_fingerprint": after,
            "artifacts": {
                "proposal": str(proposal_path), "trial_results": str(results_path),
                "stdout": str(output_dir / "controller.stdout.log"),
                "stderr": str(output_dir / "controller.stderr.log"),
            },
        }
        write_json(summary_path, summary)
        print(summary_path)
        return 0 if passed else 1
    except (ContractError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
