#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_repository_hint = os.environ.get("ROOT_DIR") or os.getcwd()
if _repository_hint not in sys.path:
    sys.path.insert(0, _repository_hint)

from task_contract import (
    ContractError,
    find_repository_root,
    load_task_spec,
    load_task_adapter,
    normalize_result,
    resolved_workspace,
    write_json,
)
from ai4ai.plugin_manifest import load_task_plugin_manifest
from execution_backends import (
    ExecutionBackendError,
    ExecutionRequest,
    RjobConfig,
    RjobExecutionBackend,
    classify_rjob_status,
    parse_actual_rjob_name,
    recover_json_result_from_rjob_log,
)
from stability_gate import archive_decision, load_standardized


def configure_dataset_runtime_env(mode: str, trials: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Make a task's logical dataset binding authoritative for every backend."""
    raw_spec = os.environ.get("TASK_SPEC")
    if not raw_spec:
        return None
    spec_path = Path(raw_spec).resolve()
    spec = load_task_spec(spec_path)
    data = spec.get("data") or {}
    dataset_ref = str(data.get("dataset_ref", ""))
    if not dataset_ref:
        return None
    repository = find_repository_root(spec_path)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from datasets.registry import DatasetRegistry

    registry = DatasetRegistry(repository / "datasets")
    runtime = data.get("bindings", {}).get("runtime", {})
    binding = registry.binding(
        dataset_ref,
        artifact=runtime.get("artifact") or data.get("default_artifact"),
        split=runtime.get("split"),
    )
    pack = Path(binding["pack_root"])
    os.environ.update({
        "DATASET_REF": dataset_ref,
        "DATASET_MANIFEST_SHA256": binding["manifest_sha256"],
        "DATASET_DIR": str(pack),
        "DATASET_DIR_IN_PACKAGE": str(pack.relative_to(repository)),
    })
    project_root, implementation = resolved_workspace(spec, spec_path)
    os.environ.update({
        "PROJECT_ROOT": str(project_root),
        "IMPLEMENTATION_DIR": str(implementation),
    })
    if mode in {"rjob", "rjob_dry_run"}:
        manifest_path = repository / "tasks" / str(spec["task"]["name"]) / "task_plugin.yaml"
        if manifest_path.is_file():
            plugin = load_task_plugin_manifest(manifest_path)
            declared = plugin.section("runtime").get("support_files") or []
            support = [str(item) for item in declared]
        else:
            support = [
                "tasks", "controller_bash", str(project_root.relative_to(repository)),
                "datasets/__init__.py", "datasets/adapter_contract.py", "datasets/registry.py",
                "datasets/registry.yaml", "datasets/generic_adapter.py",
            ]
        support.extend([
            str((pack / "adapter.py").relative_to(repository)),
            str(registry.manifest_path(dataset_ref).relative_to(repository)),
            str((pack / "SKILL.md").relative_to(repository)),
        ])
        modes_file = str(registry.manifest(dataset_ref).get("usage_modes_file", "usage_modes.json"))
        support.append(str((pack / modes_file).relative_to(repository)))
        requested_bindings = [binding]
        requested_artifacts = {
            str((trial.get("cli_overrides") or {}).get("prepared_artifact", ""))
            for trial in trials if isinstance(trial, dict)
        }
        requested_artifacts.update(
            str(name) for name in data.get("runtime_artifacts", []) if name
        )
        for artifact in sorted(requested_artifacts - {""}):
            requested_bindings.append(registry.binding(dataset_ref, artifact=artifact))
        for requested in requested_bindings:
            for item in requested["resolved"]["artifacts"]:
                path = Path(item["path"])
                support.append(str(path.relative_to(repository)))
                sidecar = path.with_suffix(".json")
                if sidecar.exists():
                    support.append(str(sidecar.relative_to(repository)))
        os.environ.update({
            "RJOB_SYNC_SOURCE": str(repository),
            "RJOB_SYNC_PATHS": ":".join(dict.fromkeys(path for path in support if (repository / path).exists())),
            "TRIAL_WORKDIR_IN_PACKAGE": str(implementation.relative_to(repository)),
            "TRIAL_PYTHONPATH_IN_PACKAGE": os.path.relpath(repository, implementation),
            "COMMAND_IMPLEMENTATION_DIR_IN_PACKAGE": ".",
            "PROJECT_ROOT_IN_PACKAGE": str(project_root.relative_to(repository)),
        })
    return binding


def trials_from_suggestion(suggestion: dict[str, Any]) -> list[dict[str, Any]]:
    if suggestion.get("schema_version") != "omni-ar-proposal/v2":
        trials = suggestion.get("sweep_trials", [])
        if not isinstance(trials, list):
            raise ValueError("suggestion.sweep_trials must be a list")
        return trials
    task_spec_path = os.environ.get("TASK_SPEC")
    if not task_spec_path:
        raise ValueError("structured proposals require TASK_SPEC")
    spec_path = Path(task_spec_path).resolve()
    spec = load_task_spec(spec_path)
    adapter = load_task_adapter(spec, spec_path)
    experiments = suggestion.get("experiment_proposals")
    if not isinstance(experiments, list):
        raise ValueError("suggestion.experiment_proposals must be a list")
    defaults = spec["adapter"]["trial_defaults"]
    runnable = list(experiments)
    for request in suggestion.get("implementation_requests") or []:
        trial = request.get("trial_proposal") or {}
        try:
            adapter.proposal_to_trial(trial, defaults, "implemented_candidate")
        except Exception:
            continue
        if trial not in runnable:
            runnable.append(trial)
    return [
        adapter.proposal_to_trial(experiment, defaults, f"experiment_{index:03d}")
        for index, experiment in enumerate(runnable)
    ]


def command_string_for(trial: dict[str, Any], output: Path) -> str:
    trial_json = json.dumps(trial, ensure_ascii=False)
    if "command" in trial:
        template = str(trial["command"])
    elif "entrypoint" in trial:
        task_spec_path = os.environ.get("TASK_SPEC")
        if not task_spec_path:
            raise ValueError("trial.entrypoint requires TASK_SPEC")
        spec = load_task_spec(Path(task_spec_path).resolve())
        entrypoint = str(trial["entrypoint"])
        if entrypoint in spec["entrypoints"]:
            template = spec["entrypoints"][entrypoint]
        elif entrypoint in spec["adapter"]["actions"]:
            template = spec["adapter"]["actions"][entrypoint]
        else:
            raise ValueError(f"unknown task entrypoint: {entrypoint}")
        overrides = trial.get("cli_overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("trial.cli_overrides must be an object")
        parts = [template]
        for key, value in overrides.items():
            flag = "--" + str(key).replace("_", "-")
            if value is True:
                parts.append(flag)
            elif value is not False and value is not None:
                encoded = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
                parts.extend([flag, shlex.quote(encoded)])
        template = " ".join(parts)
    else:
        template = os.environ.get("TRIAL_COMMAND_TEMPLATE", "")
    if not template:
        raise ValueError("trial.command or TRIAL_COMMAND_TEMPLATE must be set")
    task_spec_path = Path(os.environ["TASK_SPEC"]).resolve() if os.environ.get("TASK_SPEC") else None
    repository_root = task_spec_path
    project_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    implementation_dir = Path(os.environ.get("IMPLEMENTATION_DIR", ".")).resolve()
    if task_spec_path:
        spec = load_task_spec(task_spec_path)
        project_root, implementation_dir = resolved_workspace(spec, task_spec_path)
        repository_root = find_repository_root(task_spec_path)
    command = template.format(
        output=str(output), result_path=str(output), trial_json=trial_json,
        repository_root=str(repository_root), project_root=str(project_root),
        implementation_dir=str(implementation_dir),
        python_executable=(
            shlex.quote(sys.executable)
            if os.environ.get("TRIAL_EXECUTION_MODE") == "local"
            else "python3"
        ),
    )
    if os.environ.get("TRIAL_EXECUTION_MODE") in {"rjob", "rjob_dry_run"}:
        command = strip_proxy_setup_commands(command)
        command = normalize_project_data_paths(command)
        command = normalize_nested_submit_script(command)
        command = unwrap_rjob_submit_command(command)
        command = normalize_package_paths(command)
    return command


def normalize_package_paths(command: str) -> str:
    """Translate control-node absolute paths to paths inside an rjob package."""
    if not (os.environ.get("RJOB_SHARED_FOLDER") or os.environ.get("RJOB_FOLDER")):
        return command
    command_impl = os.environ.get("COMMAND_IMPLEMENTATION_DIR_IN_PACKAGE", ".")
    mappings = [
        (os.environ.get("DATASET_DIR", ""), os.environ.get("DATASET_DIR_IN_PACKAGE", "")),
        (os.environ.get("IMPLEMENTATION_DIR", ""), command_impl),
        (os.environ.get("PROJECT_ROOT", ""), os.environ.get("PROJECT_ROOT_IN_PACKAGE", "")),
    ]
    for host_path, package_path in mappings:
        if host_path and package_path:
            command = command.replace(host_path, package_path)
    # Heuresis may quote paths copied from an older machine/case.  Once the
    # trial is packaged, those mounts do not exist; translate the two scoped
    # project locations to the package layout as well.
    dataset_in_package = os.environ.get("DATASET_DIR_IN_PACKAGE", "")
    if dataset_in_package:
        command = re.sub(
            r"/mnt/shared-storage[^\s'\"\\]*/datasets",
            dataset_in_package,
            command,
        )
    command = re.sub(
        r"/mnt/shared-storage[^\s'\"\\]*/implementation",
        command_impl,
        command,
    )
    workdir_in_package = os.environ.get("TRIAL_WORKDIR_IN_PACKAGE", "")
    if workdir_in_package:
        command = re.sub(
            rf"(?m)^cd\s+{re.escape(workdir_in_package)}\s*$",
            "cd .",
            command,
        )
    return command


def strip_proxy_setup_commands(command: str) -> str:
    lines = []
    for raw_line in command.splitlines():
        line = raw_line.strip()
        if line in {"proxy_on", "proxyon"}:
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def normalize_project_data_paths(command: str) -> str:
    project_root = os.environ.get("PROJECT_ROOT", "")
    dataset_dir = os.environ.get("DATASET_DIR", "")
    if project_root and dataset_dir:
        legacy_data = str(Path(project_root) / "data")
        command = re.sub(
            rf"(?<![\w.-]){re.escape(legacy_data)}(?=/|$)",
            dataset_dir,
            command,
        )
    return command


def normalize_nested_submit_script(command: str) -> str:
    """Avoid submitting an rjob from inside the rjob worker.

    Heuresis sometimes suggests a user-facing command such as
    `cd <implementation> && bash scripts/submit_rjob.sh`. The controller itself
    is already the rjob executor, so the worker should run the payload smoke
    script directly. This keeps the rule project-agnostic while handling the
    common submit-script naming convention.
    """
    implementation = os.environ.get("IMPLEMENTATION_DIR") or os.environ.get("TRIAL_WORKDIR", "")
    if not implementation:
        return command
    submit_script = str(Path(implementation) / "scripts" / "submit_rjob.sh")
    smoke_script = str(Path(implementation) / "scripts" / "run_smoke.sh")
    if not Path(smoke_script).exists():
        return command

    patterns = [
        f"cd {shlex.quote(implementation)} && bash scripts/submit_rjob.sh",
        f"cd {implementation} && bash scripts/submit_rjob.sh",
        f"bash {shlex.quote(submit_script)}",
        f"bash {submit_script}",
    ]
    for pattern in patterns:
        if command.strip() == pattern:
            return "bash scripts/run_smoke.sh"
    return command


def unwrap_rjob_submit_command(command: str) -> str:
    normalized = command.replace("\\\n", " ")
    try:
        parts = shlex.split(normalized)
    except ValueError:
        return command
    try:
        start = next(i for i in range(len(parts) - 1) if parts[i : i + 2] == ["rjob", "submit"])
    except StopIteration:
        return command
    if "--" not in parts[start:]:
        return command
    idx = parts.index("--", start)
    inner = parts[idx + 1 :]
    if not inner:
        return command
    inner = normalize_inner_worker_command(inner)
    return " ".join(shlex.quote(part) for part in inner)


def normalize_inner_worker_command(parts: list[str]) -> list[str]:
    if len(parts) >= 2 and parts[0] == "bash":
        script = Path(parts[1])
        if not script.is_absolute() and len(script.parts) == 1:
            workdir_raw = os.environ.get("TRIAL_WORKDIR") or os.environ.get("IMPLEMENTATION_DIR", "")
            if workdir_raw:
                workdir = Path(workdir_raw)
                direct = workdir / script
                nested = workdir / "scripts" / script
                if not direct.exists() and nested.exists():
                    parts = parts.copy()
                    parts[1] = str(Path("scripts") / script)
    return parts


def command_for(trial: dict[str, Any], output: Path) -> list[str]:
    command = command_string_for(trial, output)
    return ["bash", "-lc", command]


def safe_job_name(round_name: str, idx: int, trial: dict[str, Any]) -> str:
    raw_name = str(trial.get("name") or f"trial-{idx}")
    digest = hashlib.sha1(json.dumps(trial, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:8]
    base = f"hcx-{round_name}-t{idx:03d}-{raw_name}-{digest}".lower()
    base = re.sub(r"[^a-z0-9-]+", "-", base).strip("-")
    return base[:62].rstrip("-") or f"hcx-{digest}"


def simulated_metric_for(trial: dict[str, Any], idx: int, output: Path) -> dict[str, Any]:
    name = str(trial.get("name") or f"trial_{idx}")
    command = str(trial.get("command") or "")
    return {
        "status": "simulated_ok",
        "simulated": True,
        "simulation_basis": "controller_plumbing_only",
        "primary_metric": None,
        "plumbing_token": hashlib.sha1(name.encode("utf-8")).hexdigest()[:12],
        "trial": trial,
        "command": command,
        "note": "No scientific metric is synthesized. This artifact only verifies controller plumbing.",
    }


def standardize_trial_result(
    record: dict[str, Any],
    raw_output: Path,
    suggestion: dict[str, Any],
) -> None:
    task_spec_path = os.environ.get("TASK_SPEC")
    if not task_spec_path or not raw_output.exists():
        return
    standard_output = raw_output.with_suffix(".result.json")
    try:
        spec = load_task_spec(Path(task_spec_path).resolve())
        raw = json.loads(raw_output.read_text(encoding="utf-8"))
        command = record.get("command", "")
        if isinstance(command, list):
            command = " ".join(shlex.quote(str(part)) for part in command)
        standardized = normalize_result(
            raw=raw,
            spec=spec,
            raw_path=raw_output,
            proposal_id=str(suggestion.get("proposal_id", "legacy-proposal")),
            trial_name=str((record.get("trial") or {}).get("name", f"trial_{record.get('trial_index', 0)}")),
            command=str(command),
            resource_request=(record.get("trial") or {}).get("resource_request"),
        )
        write_json(standard_output, standardized)
        record["standardized_result"] = str(standard_output)
        record["result_contract_status"] = "ok"
    except (ContractError, json.JSONDecodeError) as exc:
        record["standardized_result"] = str(standard_output)
        record["result_contract_status"] = "failed"
        record["result_contract_error"] = str(exc)
        if record.get("status") == "ok":
            record["status"] = "failed"


def evaluate_acceptance(raw_output: Path, criteria: dict[str, Any]) -> dict[str, Any]:
    operators = {
        ">": lambda actual, target: actual > target,
        ">=": lambda actual, target: actual >= target,
        "<": lambda actual, target: actual < target,
        "<=": lambda actual, target: actual <= target,
        "==": lambda actual, target: actual == target,
    }
    raw = json.loads(raw_output.read_text(encoding="utf-8"))
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    checks = []
    for metric_name, rule in criteria.items():
        actual = metrics.get(metric_name, raw.get(metric_name))
        operator_name = rule["operator"]
        target = rule["value"]
        passed = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and operators[operator_name](actual, target)
        )
        checks.append({
            "metric": metric_name,
            "actual": actual,
            "operator": operator_name,
            "target": target,
            "passed": passed,
        })
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suggestion", type=Path, required=True)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    suggestion = json.loads(args.suggestion.read_text(encoding="utf-8"))
    trials = trials_from_suggestion(suggestion)
    max_trials = int(os.environ.get("MAX_TRIALS_PER_ROUND", "12"))
    trials = trials[:max_trials]
    mode = os.environ.get("TRIAL_EXECUTION_MODE", "skip")
    dataset_binding = configure_dataset_runtime_env(mode, trials)

    trial_root = (
        args.artifact_dir
        / os.environ.get("TRIAL_ARTIFACT_SUBDIR", "controller_trials")
        / args.round_dir.name
    ).resolve()
    trial_root.mkdir(parents=True, exist_ok=True)
    results = []
    rjob_backend = None
    if mode in {"rjob", "rjob_dry_run"}:
        try:
            rjob_backend = RjobExecutionBackend(RjobConfig.from_env(trial_root))
        except ExecutionBackendError as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
            return 2

    for idx, trial in enumerate(trials):
        if not isinstance(trial, dict):
            results.append({"trial_index": idx, "status": "invalid", "error": "trial must be an object"})
            continue
        output = trial_root / f"trial_{idx:03d}.json"
        stdout_path = trial_root / f"trial_{idx:03d}.stdout.log"
        stderr_path = trial_root / f"trial_{idx:03d}.stderr.log"
        record = {
            "trial_index": idx,
            "mode": mode,
            "trial": trial,
            "command": command_for(trial, output),
            "output": str(output),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        if mode in {"skip", "dry_run"} or os.environ.get("DRY_RUN", "0") == "1":
            record["status"] = "skipped"
            record["reason"] = "TRIAL_EXECUTION_MODE=skip/dry_run"
            output.write_text(json.dumps({"status": "skipped", "trial": trial}, indent=2), encoding="utf-8")
        elif mode == "simulated":
            metric = simulated_metric_for(trial, idx, output)
            output.write_text(json.dumps(metric, indent=2, ensure_ascii=False), encoding="utf-8")
            stdout_path.write_text(
                f"simulated trial {idx}: {trial.get('name', '')}\noutput={output}\n",
                encoding="utf-8",
            )
            stderr_path.write_text("", encoding="utf-8")
            record["status"] = "ok"
            record["returncode"] = 0
            record["elapsed_sec"] = 0.0
            record["simulated"] = True
        elif mode == "local":
            start = time.time()
            env = os.environ.copy()
            impl = os.environ.get("IMPLEMENTATION_DIR", "")
            root = os.environ.get("ROOT_DIR", "")
            python_paths = [path for path in (root, impl, env.get("PYTHONPATH", "")) if path]
            env["PYTHONPATH"] = ":".join(python_paths)
            proc = subprocess.run(record["command"], cwd=impl or None, env=env, text=True, capture_output=True, check=False)
            stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
            stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
            record["returncode"] = proc.returncode
            record["elapsed_sec"] = time.time() - start
            record["status"] = "ok" if proc.returncode == 0 else "failed"
        elif mode in {"rjob", "rjob_dry_run"}:
            assert rjob_backend is not None
            try:
                record = rjob_backend.execute(
                    ExecutionRequest(
                        job_name=safe_job_name(args.round_dir.name, idx, trial),
                        command=command_string_for(trial, output),
                        worker_script=trial_root / f"trial_{idx:03d}_worker.sh",
                        output=output,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        trial_root=trial_root,
                        resource_request=trial.get("resource_request") or {},
                        trial=trial,
                    ),
                    dry_run=mode == "rjob_dry_run",
                )
                record["trial_index"] = idx
            except ExecutionBackendError as exc:
                record.update(status="failed", error=str(exc))
                write_json(output, {
                    "status": "failed", "error": str(exc), "trial": trial,
                    "execution_backend": "rjob",
                })
        else:
            record["status"] = "unsupported"
            record["error"] = f"unsupported TRIAL_EXECUTION_MODE={mode}; use skip, dry_run, simulated, local, rjob_dry_run, or rjob"
        standardize_trial_result(record, output, suggestion)
        criteria = trial.get("acceptance_criteria") or {}
        if record.get("status") == "ok" and output.exists() and isinstance(criteria, dict):
            try:
                record["acceptance"] = evaluate_acceptance(output, criteria)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                record["acceptance"] = {"passed": False, "error": str(exc), "checks": []}
        overrides = (record.get("trial") or trial).get("cli_overrides") or {}
        expected_context = {
            "dataset_ref": overrides.get("dataset_ref"),
            "requested_split": overrides.get("split_strategy"),
            "prepared_artifact": overrides.get("prepared_artifact"),
            "seed": overrides.get("seed"),
        }
        record["archive_gate"] = archive_decision(
            record,
            load_standardized(record),
            expected_context={key: value for key, value in expected_context.items() if value is not None},
        )
        results.append(record)

    payload = {
        "round_dir": str(args.round_dir),
        "mode": mode,
        "num_trials": len(trials),
        "results": results,
        "dataset_binding": dataset_binding,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out)
    return 0 if all(r.get("status") in {"ok", "skipped"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
