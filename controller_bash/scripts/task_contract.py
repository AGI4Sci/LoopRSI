#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


class ContractError(ValueError):
    pass


def enforce_autonomous_candidate_guard(proposal: dict[str, Any]) -> None:
    """Fail closed for runs that explicitly forbid historical candidate fallback."""
    if os.environ.get("REQUIRE_AUTONOMOUS_CANDIDATE") != "1":
        return
    allowed = {"autonomous_research_candidate"}
    trials = list(proposal.get("experiment_proposals") or [])
    trials.extend(
        request.get("trial_proposal")
        for request in (proposal.get("implementation_requests") or [])
        if isinstance(request, dict) and isinstance(request.get("trial_proposal"), dict)
    )
    for index, trial in enumerate(trials):
        parameters = trial.get("parameters") if isinstance(trial, dict) else None
        variant = (parameters or {}).get("candidate_variant") if isinstance(parameters, dict) else None
        if variant not in allowed:
            raise ContractError(
                "strict autonomous admission requires every executable trial to use "
                f"an autonomous candidate_variant in {sorted(allowed)!r}; "
                f"trial[{index}] requested {variant!r}"
            )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_task_spec(path: Path) -> dict[str, Any]:
    path = resolve_task_spec_path(path)
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ContractError("task spec must be a YAML mapping")
    validate_schema(obj, SCHEMA_DIR / "task_spec.schema.json")
    validate_task_policy(obj, path)
    return obj


def resolve_task_spec_path(path: Path) -> Path:
    """Resolve a task spec relative to ROOT_DIR when invoked outside the worktree."""
    if path.is_absolute():
        return path.resolve()
    configured = os.environ.get("ROOT_DIR")
    if configured:
        candidate = (Path(configured).resolve() / path).resolve()
        if candidate.is_file():
            return candidate
    return path.resolve()


def validate_schema(obj: Any, schema_path: Path) -> None:
    schema = load_json(schema_path)
    relative_ref = schema.get("$ref") if isinstance(schema, dict) else None
    if isinstance(relative_ref, str) and not relative_ref.startswith(("#", "http://", "https://")):
        schema = load_json((schema_path.parent / relative_ref).resolve())
    errors = sorted(Draft202012Validator(schema).iter_errors(obj), key=lambda error: list(error.path))
    if errors:
        details = []
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            details.append(f"{location}: {error.message}")
        raise ContractError("schema validation failed: " + "; ".join(details))


def resolved_workspace(spec: dict[str, Any], spec_path: Path) -> tuple[Path, Path]:
    base = spec_path.parent
    if spec["workspace"]["base"] == "repository":
        base = find_repository_root(spec_path)
    project_root = (base / spec["workspace"]["project_root"]).resolve()
    implementation_dir = (project_root / spec["workspace"]["implementation_dir"]).resolve()
    if implementation_dir != project_root and project_root not in implementation_dir.parents:
        raise ContractError("workspace.implementation_dir must stay inside project_root")
    return project_root, implementation_dir


def find_repository_root(path: Path) -> Path:
    """Resolve the repository independently of the caller's working directory."""
    configured = os.environ.get("ROOT_DIR")
    candidates = []
    if not path.is_absolute() and configured:
        candidates.append(Path(configured).resolve() / path)
    candidates.append(path.resolve())
    for candidate in candidates:
        for parent in [candidate.parent, *candidate.parents]:
            if (parent / "controller_bash").is_dir() and (parent / "tasks").is_dir():
                return parent
    if configured:
        return Path(configured).resolve()
    raise ContractError(f"cannot locate repository root for {path.resolve()}")


def load_task_adapter(spec: dict[str, Any], spec_path: Path) -> Any:
    repository = find_repository_root(spec_path)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    module_path = scoped_path(spec["adapter"]["module"], repository)
    if not module_path.is_file():
        raise ContractError(f"adapter module does not exist: {module_path}")
    module_spec = importlib.util.spec_from_file_location(
        f"omni_ar_adapter_{spec['task']['name']}", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ContractError(f"cannot load adapter module: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    adapter_class = getattr(module, spec["adapter"]["class_name"], None)
    if not isinstance(adapter_class, type):
        raise ContractError(f"adapter class not found: {spec['adapter']['class_name']}")
    try:
        return adapter_class()
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"cannot initialize task adapter: {exc}") from exc


def scoped_path(raw: str, root: Path) -> Path:
    candidate = (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise ContractError(f"path escapes its configured scope: {raw}")
    return candidate


def overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_task_policy(spec: dict[str, Any], spec_path: Path) -> None:
    adapter = load_task_adapter(spec, spec_path)
    adapter_class = type(adapter)
    missing_actions = [
        action for action in spec["adapter"]["actions"]
        if not callable(getattr(adapter_class, action, None))
    ]
    if missing_actions:
        raise ContractError(f"adapter class is missing actions: {missing_actions}")
    _, implementation_dir = resolved_workspace(spec, spec_path)
    editable = [scoped_path(path, implementation_dir) for path in spec["search"]["editable_paths"]]
    protected = [scoped_path(path, implementation_dir) for path in spec["search"]["protected_paths"]]
    for edit_path in editable:
        for protected_path in protected:
            if overlaps(edit_path, protected_path):
                raise ContractError(
                    f"editable/protected paths overlap: {edit_path} and {protected_path}"
                )
    repository = find_repository_root(spec_path)
    interfaces = [
        scoped_path(path, repository)
        for path in spec["search"].get("interface_editable_paths", [])
    ]
    task_project, _ = resolved_workspace(spec, spec_path)
    outside_task = [path for path in interfaces if path != task_project and task_project not in path.parents]
    if outside_task:
        raise ContractError(
            "search.interface_editable_paths must stay inside the task project: "
            + ", ".join(str(path) for path in outside_task)
        )
    metric_names = [spec["metrics"]["primary"]["name"]]
    metric_names.extend(metric["name"] for metric in spec["metrics"]["secondary"])
    if len(metric_names) != len(set(metric_names)):
        raise ContractError("metric names must be unique")
    primary_role = spec["metrics"]["primary"].get("role", "primary")
    if primary_role != "primary":
        raise ContractError("metrics.primary role must be 'primary'")
    for definition in spec["metrics"]["secondary"]:
        if definition.get("role", "diagnostic") == "primary":
            raise ContractError("secondary metrics cannot have role 'primary'")


def validate_runtime_binding(spec: dict[str, Any], spec_path: Path) -> None:
    project_root, implementation_dir = resolved_workspace(spec, spec_path)
    bindings = {
        "PROJECT_ROOT": project_root,
        "IMPLEMENTATION_DIR": implementation_dir,
    }
    for variable, expected in bindings.items():
        configured = os.environ.get(variable)
        if configured and Path(configured).resolve() != expected:
            raise ContractError(
                f"{variable}={Path(configured).resolve()} disagrees with task spec {expected}"
            )


def path_is_editable(raw: str, spec: dict[str, Any], spec_path: Path) -> bool:
    repository = find_repository_root(spec_path)
    _, implementation_dir = resolved_workspace(spec, spec_path)
    interface_raw = set(spec["search"].get("interface_editable_paths", []))
    if raw in interface_raw:
        candidate = scoped_path(raw, repository)
        return candidate.is_file() or candidate.parent.is_dir()
    candidate = scoped_path(raw, implementation_dir)
    editable = [scoped_path(path, implementation_dir) for path in spec["search"]["editable_paths"]]
    protected = [scoped_path(path, implementation_dir) for path in spec["search"]["protected_paths"]]
    allowed = any(candidate == root or root in candidate.parents for root in editable)
    denied = any(overlaps(candidate, root) for root in protected)
    return allowed and not denied


def editable_workspace_path(raw: str, spec: dict[str, Any], spec_path: Path) -> str:
    """Map a policy path to a repository-relative Coding Agent path."""
    if not path_is_editable(raw, spec, spec_path):
        raise ContractError(f"path is not editable: {raw}")
    repository = find_repository_root(spec_path)
    if raw in set(spec["search"].get("interface_editable_paths", [])):
        return str(scoped_path(raw, repository).relative_to(repository))
    _, implementation_dir = resolved_workspace(spec, spec_path)
    return str(scoped_path(raw, implementation_dir).relative_to(repository))


def validate_proposal(proposal: dict[str, Any], spec: dict[str, Any], spec_path: Path) -> None:
    version = proposal.get("schema_version")
    schema_name = "proposal_v2.schema.json" if version == "omni-ar-proposal/v2" else "proposal.schema.json"
    validate_schema(proposal, SCHEMA_DIR / schema_name)
    if proposal["task_name"] != spec["task"]["name"]:
        raise ContractError(
            f"proposal task_name={proposal['task_name']!r} does not match task spec {spec['task']['name']!r}"
        )
    if version == "omni-ar-proposal/v2":
        enforce_autonomous_candidate_guard(proposal)
        experiments = proposal["experiment_proposals"]
        if not experiments and not (proposal.get("implementation_requests") or []):
            raise ContractError("proposal requires an experiment or implementation request")
        max_trials = int((spec.get("resources") or {}).get("max_trials_per_round", len(experiments)))
        if len(experiments) > max_trials:
            raise ContractError(f"proposal has {len(experiments)} experiments, task limit is {max_trials}")
        adapter = load_task_adapter(spec, spec_path)
        defaults = spec["adapter"]["trial_defaults"]
        adapter_path = spec["adapter"]["module"]
        implementation_methods = {
            request.get("trial_proposal", {}).get("parameters", {}).get("method")
            for request in proposal.get("implementation_requests") or []
            if adapter_path in set(request.get("allowed_paths") or [])
        }
        for index, experiment in enumerate(experiments):
            try:
                adapter.proposal_to_trial(experiment, defaults, f"experiment_{index:03d}")
            except Exception as exc:  # noqa: BLE001
                method = experiment.get("parameters", {}).get("method")
                if method not in implementation_methods:
                    raise ContractError(f"experiment_proposals[{index}] cannot be compiled: {exc}") from exc
            request = experiment["resource_request"]
            task_resources = spec.get("resources") or {}
            if request.get("gpu_count", 0) > task_resources.get("gpu_count", 0):
                raise ContractError(f"experiment_proposals[{index}] exceeds task GPU limit")
            if request.get("max_runtime_minutes", 1) > task_resources.get("max_runtime_minutes", 1):
                raise ContractError(f"experiment_proposals[{index}] exceeds task runtime limit")
        for index, implementation in enumerate(proposal.get("implementation_requests") or []):
            invalid = sorted({
                path for path in implementation["allowed_paths"]
                if not path_is_editable(path, spec, spec_path)
            })
            if invalid:
                raise ContractError(
                    f"implementation_requests[{index}] references non-editable or protected paths: {invalid}"
                )
            if not implementation.get("activation_diagnostics"):
                raise ContractError(
                    f"implementation_requests[{index}] requires activation diagnostics"
                )
            trial = implementation["trial_proposal"]
            try:
                adapter.proposal_to_trial(
                    trial, defaults, f"implementation_{index:03d}"
                )
            except Exception as exc:  # noqa: BLE001
                if adapter_path not in set(implementation["allowed_paths"]):
                    raise ContractError(
                        f"implementation_requests[{index}].trial_proposal needs a new "
                        f"task capability, but task-local Adapter {adapter_path!r} is not allowed"
                    ) from exc
            request = trial["resource_request"]
            task_resources = spec.get("resources") or {}
            if request.get("gpu_count", 0) > task_resources.get("gpu_count", 0):
                raise ContractError(
                    f"implementation_requests[{index}] exceeds task GPU limit"
                )
            if request.get("max_runtime_minutes", 1) > task_resources.get(
                "max_runtime_minutes", 1
            ):
                raise ContractError(
                    f"implementation_requests[{index}] exceeds task runtime limit"
                )
    else:
        paths: list[str] = []
        for variant in proposal["idea_variants"]:
            paths.extend(change["file_hint"] for change in variant["required_code_changes"])
        for task in proposal["codex_tasks"]:
            paths.extend(task["allowed_paths"])
        invalid = sorted({path for path in paths if not path_is_editable(path, spec, spec_path)})
        if invalid:
            raise ContractError(f"proposal references non-editable or protected paths: {invalid}")


def extract_source(obj: Any, source: str) -> Any:
    current = obj
    for part in source.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ContractError(f"raw result does not contain metric source {source!r}")
    return current


def finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def compare_values(actual: float, operator: str, target: float) -> bool:
    operators = {
        ">": lambda left, right: left > right,
        ">=": lambda left, right: left >= right,
        "<": lambda left, right: left < right,
        "<=": lambda left, right: left <= right,
        "==": lambda left, right: left == right,
    }
    return operators[operator](actual, target)


def metric_value(
    raw: dict[str, Any],
    definition: dict[str, Any],
    baseline_metrics: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    value = extract_source(raw, definition["source"])
    if not finite_number(value):
        raise ContractError(f"metric {definition['name']!r} is not a finite number: {value!r}")
    baseline = baseline_metrics.get(definition["name"])
    if not finite_number(baseline):
        baseline = None
    beats_baseline = None
    if baseline is not None:
        beats_baseline = value > baseline if definition["direction"] == "maximize" else value < baseline
    constraint_satisfied = None
    if definition.get("constraint"):
        rule = definition["constraint"]
        constraint_satisfied = compare_values(value, rule["operator"], rule["value"])
    elif role == "constraint" and beats_baseline is not None:
        constraint_satisfied = beats_baseline
    return {
        "value": value,
        "direction": definition["direction"],
        "role": role,
        "baseline_value": baseline,
        "beats_baseline": beats_baseline,
        "constraint_satisfied": constraint_satisfied,
    }


def normalized_baselines(raw: dict[str, Any], definitions: list[dict[str, Any]]) -> dict[str, Any]:
    source = raw.get("baseline_metrics")
    if not isinstance(source, dict):
        source = {}
    return {
        definition["name"]: source.get(definition["name"])
        if finite_number(source.get(definition["name"])) else None
        for definition in definitions
    }


def normalized_resource_usage(
    raw: dict[str, Any],
    spec: dict[str, Any],
    resource_request: dict[str, Any] | None,
) -> dict[str, Any]:
    supplied = raw.get("resource_usage")
    usage = dict(supplied) if isinstance(supplied, dict) else {}
    runtime = usage.get("runtime_seconds", raw.get("elapsed_sec", raw.get("runtime_seconds")))
    usage["runtime_seconds"] = runtime if finite_number(runtime) and runtime >= 0 else None
    for key in ("gpu_count", "cpu", "memory_mb"):
        value = usage.get(key)
        usage[key] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    task_budget = spec.get("resources") or {}
    request = dict(resource_request or {})
    budget = {
        "gpu_count": request.get("gpu_count", task_budget.get("gpu_count")),
        "cpu": request.get("cpu"),
        "memory_mb": request.get("memory_mb"),
        "max_runtime_minutes": request.get(
            "max_runtime_minutes", task_budget.get("max_runtime_minutes")
        ),
    }
    checks: list[bool] = []
    if usage["runtime_seconds"] is not None and finite_number(budget["max_runtime_minutes"]):
        checks.append(usage["runtime_seconds"] <= budget["max_runtime_minutes"] * 60)
    for key in ("gpu_count", "cpu", "memory_mb"):
        if usage[key] is not None and finite_number(budget[key]):
            checks.append(usage[key] <= budget[key])
    usage["budget"] = budget
    usage["within_budget"] = all(checks) if checks else None
    return usage


def normalized_protocol(
    raw: dict[str, Any], primary_value: float | int | None
) -> dict[str, Any]:
    supplied = raw.get("protocol")
    protocol = dict(supplied) if isinstance(supplied, dict) else {}
    seed = protocol.get("seed", raw.get("seed"))
    protocol["seed"] = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
    stability = protocol.get("stability")
    if not isinstance(stability, dict):
        if primary_value is None:
            stability = {
                "status": "unavailable", "num_seeds": 0,
                "mean": None, "std": None, "min": None, "max": None,
            }
        else:
            stability = {
                "status": "single_seed", "num_seeds": 1,
                "mean": primary_value, "std": None,
                "min": primary_value, "max": primary_value,
            }
    protocol["stability"] = stability
    executed = raw.get("executed_parameters")
    if isinstance(executed, dict):
        protocol["executed_parameters"] = dict(executed)
    return protocol


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_features(
    raw: dict[str, Any], spec: dict[str, Any], raw_path: Path,
) -> dict[str, Any]:
    """Record selected feature assets and their activation in every new result.

    Dataset manifests are authoritative for builder versions and immutable
    feature files.  Runtime diagnostics are authoritative for activation.  A
    declared feature never becomes active merely because its files exist.
    """
    data = spec.get("data") or {}
    dataset_ref = str(data.get("dataset_ref") or "")
    protocol = raw.get("protocol") if isinstance(raw.get("protocol"), dict) else {}
    executed = raw.get("executed_parameters")
    executed = dict(executed) if isinstance(executed, dict) else {}
    explicit = raw.get("features") if isinstance(raw.get("features"), dict) else {}
    requested_ids = explicit.get("selected_ids") or executed.get("feature_ids")
    if isinstance(requested_ids, str):
        requested_ids = [requested_ids]
    requested = {str(value) for value in (requested_ids or [])}
    feature_id = executed.get("feature_id")
    if isinstance(feature_id, str) and feature_id:
        requested.add(feature_id)
    # An explicit feature selection is authoritative.  In particular,
    # ``feature_id=raw`` must remain a feature-free control even when its
    # method also has a registered feature binding.  Method-based inference
    # only supports older task results that did not record feature_id yet.
    infer_from_method = not requested
    method = str(executed.get("method") or raw.get("method") or "")
    runtime_diagnostics = raw.get("activation_diagnostics")
    runtime_diagnostics = dict(runtime_diagnostics) if isinstance(runtime_diagnostics, dict) else {}
    backend = raw.get("backend_metadata")
    backend = dict(backend) if isinstance(backend, dict) else {}
    phenotype = backend.get("phenotype_activation")
    if isinstance(phenotype, dict):
        runtime_diagnostics = {**runtime_diagnostics, **phenotype}
    usage = raw.get("resource_usage")
    usage = dict(usage) if isinstance(usage, dict) else {}

    records: list[dict[str, Any]] = []
    manifest_hash = os.environ.get("DATASET_MANIFEST_SHA256") or None
    if dataset_ref:
        try:
            repository = find_repository_root(raw_path.resolve())
        except ContractError:
            configured = os.environ.get("TASK_SPEC")
            repository = find_repository_root(Path(configured).resolve()) if configured else None
        if repository is not None:
            try:
                from datasets.registry import DatasetRegistry

                registry = DatasetRegistry(repository / "datasets")
                manifest = registry.manifest(dataset_ref)
                manifest_path = registry.manifest_path(dataset_ref)
                manifest_hash = manifest_hash or _sha256_file(manifest_path)
                pack = registry.pack_root(dataset_ref)
                for name, declaration in sorted((manifest.get("features") or {}).items()):
                    if not isinstance(declaration, dict):
                        continue
                    bindings = list(declaration.get("task_bindings") or [])
                    matched = [
                        binding for binding in bindings
                        if isinstance(binding, dict) and str(binding.get("method") or "") == method
                    ]
                    selected = name in requested or (infer_from_method and bool(matched))
                    if not selected:
                        continue
                    root = (pack / str(declaration.get("root") or ".")).resolve()
                    if root != pack and pack not in root.parents:
                        raise ContractError(f"feature root escapes Dataset Pack: {name}")
                    spec_path = root / str(declaration.get("spec") or "feature_spec.yaml")
                    feature_spec = (
                        yaml.safe_load(spec_path.read_text(encoding="utf-8"))
                        if spec_path.is_file() else {}
                    )
                    feature_spec = feature_spec if isinstance(feature_spec, dict) else {}
                    validation_path = root / "validation.json"
                    validation = (
                        json.loads(validation_path.read_text(encoding="utf-8"))
                        if validation_path.is_file() else {}
                    )
                    validation = validation if isinstance(validation, dict) else {}
                    static_diagnostics = validation.get("activation_diagnostics")
                    static_diagnostics = (
                        dict(static_diagnostics) if isinstance(static_diagnostics, dict) else {}
                    )
                    files: dict[str, Any] = {}
                    if root.is_dir():
                        for path in sorted(root.rglob("*")):
                            if path.is_file() and path.name not in {"build.stdout.log", "build.stderr.log"}:
                                files[str(path.relative_to(root))] = {
                                    "sha256": _sha256_file(path), "size_bytes": path.stat().st_size,
                                }
                    required = list(feature_spec.get("activation_diagnostics") or [])
                    observed = {
                        key: runtime_diagnostics.get(key, static_diagnostics.get(key))
                        for key in required
                        if key in runtime_diagnostics or key in static_diagnostics
                    }
                    candidate_active = runtime_diagnostics.get("candidate_active")
                    activation_flag = (
                        candidate_active is True
                        or runtime_diagnostics.get("feature_active") is True
                        or explicit.get("enabled") is True
                    )
                    # Some diagnostics have an expected zero state (for
                    # example split/content overlap). Presence plus the
                    # Builder's successful validation is the generic rule;
                    # the task runtime must still emit an explicit active flag.
                    active = bool(
                        selected and activation_flag
                        and validation.get("status", "ok") == "ok"
                        and all(key in observed for key in required)
                    )
                    records.append({
                        "feature_id": str(name), "selected": True, "enabled": active,
                        "activation_status": "active" if active else "missing_or_inactive",
                        "builder": dict(feature_spec.get("builder") or declaration.get("builder") or {}),
                        "fit_scope": feature_spec.get("fit_scope"),
                        "input_hashes": dict(feature_spec.get("inputs") or {}),
                        "feature_file_hashes": files,
                        "data_usage_scope": {
                            "dataset_ref": dataset_ref,
                            "protocol": protocol.get("name"), "split": protocol.get("split"),
                            "requested_split": protocol.get("requested_split"),
                            "split_fingerprint": protocol.get("split_fingerprint"),
                            "n_train": protocol.get("n_train", usage.get("train_samples")),
                            "n_eval": protocol.get("n_eval", usage.get("eval_samples")),
                            "feature_branch": executed.get("feature_branch"),
                        },
                        "required_activation_diagnostics": required,
                        "activation_diagnostics": observed,
                        "task_binding": matched[0] if matched else None,
                    })
            except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
                return {
                    "status": "invalid", "dataset_ref": dataset_ref,
                    "dataset_manifest_sha256": manifest_hash, "records": [],
                    "error": f"feature provenance could not be resolved: {exc}",
                }
    if records:
        status = "active" if all(item["enabled"] for item in records) else "invalid"
    else:
        status = "not_requested"
    return {
        "status": status, "dataset_ref": dataset_ref or None,
        "dataset_manifest_sha256": manifest_hash, "records": records,
    }


def normalize_result(
    raw: dict[str, Any],
    spec: dict[str, Any],
    raw_path: Path,
    proposal_id: str,
    trial_name: str,
    command: str = "",
    resource_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_status = str(raw.get("status", "failed")).lower()
    status_aliases = {
        "simulated_ok": "simulated",
        "rjob_dry_run": "skipped",
        "pending": "skipped",
    }
    status = status_aliases.get(raw_status, raw_status)
    if status not in {"ok", "failed", "skipped", "simulated"}:
        status = "failed"
    definitions = [spec["metrics"]["primary"], *spec["metrics"]["secondary"]]
    baselines = normalized_baselines(raw, definitions)
    metrics: dict[str, Any] = {}
    if status == "ok":
        for index, definition in enumerate(definitions):
            role = "primary" if index == 0 else definition.get("role", "diagnostic")
            metrics[definition["name"]] = metric_value(raw, definition, baselines, role)
    primary_value = None
    if metrics:
        primary_value = next(
            record["value"] for record in metrics.values() if record["role"] == "primary"
        )
    raw_artifacts = raw.get("artifacts")
    artifacts = dict(raw_artifacts) if isinstance(raw_artifacts, dict) else {}
    artifacts["raw_result"] = str(raw_path)
    if command:
        artifacts["command"] = command
    protocol = normalized_protocol(raw, primary_value)
    data_spec = spec.get("data") or {}
    if data_spec.get("dataset_ref"):
        protocol["dataset_ref"] = data_spec["dataset_ref"]
        protocol["dataset_manifest_sha256"] = os.environ.get("DATASET_MANIFEST_SHA256") or None
    result = {
        "schema_version": "omni-ar-result/v2",
        "status": status,
        "task": spec["task"]["name"],
        "trial_id": f"{proposal_id}:{trial_name}",
        "metrics": metrics,
        "baseline_metrics": baselines,
        "resource_usage": normalized_resource_usage(raw, spec, resource_request),
        "protocol": protocol,
        "artifacts": artifacts,
        "features": normalized_features(raw, spec, raw_path),
    }
    if status == "failed":
        result["error"] = str(raw.get("error", f"raw status was {raw_status!r}"))
    validate_result(result, spec)
    return result


def validate_result(result: dict[str, Any], spec: dict[str, Any]) -> None:
    version = result.get("schema_version")
    if version == "omni-ar-result/v1":
        validate_schema(result, SCHEMA_DIR / "result.schema.json")
        if result["task_name"] != spec["task"]["name"]:
            raise ContractError("result task_name does not match task spec")
        return
    validate_schema(result, SCHEMA_DIR / "result_v2.schema.json")
    if result["task"] != spec["task"]["name"]:
        raise ContractError("result task does not match task spec")
    if result["status"] == "ok":
        expected = spec["metrics"]["primary"]
        primaries = [
            (name, record) for name, record in result["metrics"].items()
            if record["role"] == "primary"
        ]
        if len(primaries) != 1:
            raise ContractError("ok result must contain exactly one primary metric")
        name, primary = primaries[0]
        if name != expected["name"] or primary["direction"] != expected["direction"]:
            raise ContractError("result primary metric does not match task spec")
        expected_constraints = {
            definition["name"] for definition in spec["metrics"]["secondary"]
            if definition.get("role", "diagnostic") == "constraint"
        }
        actual_constraints = {
            name for name, record in result["metrics"].items()
            if record["role"] == "constraint"
        }
        if actual_constraints != expected_constraints:
            raise ContractError("result constraint metrics do not match task spec")
        definitions = {
            definition["name"]: definition
            for definition in [spec["metrics"]["primary"], *spec["metrics"]["secondary"]]
        }
        for name, record in result["metrics"].items():
            baseline = result["baseline_metrics"].get(name)
            if record["baseline_value"] != baseline:
                raise ContractError(f"metric {name!r} baseline metadata is inconsistent")
            expected_beats = None
            if baseline is not None:
                expected_beats = (
                    record["value"] > baseline if record["direction"] == "maximize"
                    else record["value"] < baseline
                )
            if record["beats_baseline"] is not expected_beats:
                raise ContractError(f"metric {name!r} baseline comparison is inconsistent")
            rule = definitions[name].get("constraint")
            if rule and record["constraint_satisfied"] is not compare_values(
                record["value"], rule["operator"], rule["value"]
            ):
                raise ContractError(f"metric {name!r} constraint metadata is inconsistent")
    stability = result["protocol"]["stability"]
    stability_status = stability["status"]
    if stability_status == "unavailable" and (
        stability["num_seeds"] != 0
        or any(stability[key] is not None for key in ("mean", "std", "min", "max"))
    ):
        raise ContractError("unavailable stability must have zero seeds and null statistics")
    if stability_status == "single_seed" and (
        stability["num_seeds"] != 1 or stability["std"] is not None
    ):
        raise ContractError("single_seed stability must have one seed and null std")
    if stability_status == "multi_seed" and (
        stability["num_seeds"] < 2 or stability["std"] is None
    ):
        raise ContractError("multi_seed stability must have at least two seeds and a std")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OMNI-AR task, proposal, and result contracts")
    sub = parser.add_subparsers(dest="command", required=True)

    task_parser = sub.add_parser("validate-task")
    task_parser.add_argument("--task-spec", type=Path, required=True)
    task_parser.add_argument("--out", type=Path)

    proposal_parser = sub.add_parser("validate-proposal")
    proposal_parser.add_argument("--task-spec", type=Path, required=True)
    proposal_parser.add_argument("--proposal", type=Path, required=True)

    normalize_parser = sub.add_parser("normalize-result")
    normalize_parser.add_argument("--task-spec", type=Path, required=True)
    normalize_parser.add_argument("--raw-result", type=Path, required=True)
    normalize_parser.add_argument("--proposal-id", required=True)
    normalize_parser.add_argument("--trial-name", required=True)
    normalize_parser.add_argument("--command-text", default="")
    normalize_parser.add_argument("--out", type=Path, required=True)

    result_parser = sub.add_parser("validate-result")
    result_parser.add_argument("--task-spec", type=Path, required=True)
    result_parser.add_argument("--result", type=Path, required=True)

    args = parser.parse_args()
    try:
        spec = load_task_spec(args.task_spec.resolve())
        if args.command == "validate-task":
            validate_runtime_binding(spec, args.task_spec.resolve())
            payload = {"status": "ok", "task": spec["task"], "task_spec": str(args.task_spec.resolve())}
            if args.out:
                write_json(args.out, payload)
            print(json.dumps(payload, ensure_ascii=False))
        elif args.command == "validate-proposal":
            validate_proposal(load_json(args.proposal), spec, args.task_spec.resolve())
            print(json.dumps({"status": "ok", "proposal": str(args.proposal)}, ensure_ascii=False))
        elif args.command == "normalize-result":
            result = normalize_result(
                load_json(args.raw_result), spec, args.raw_result, args.proposal_id,
                args.trial_name, args.command_text,
            )
            write_json(args.out, result)
            print(args.out)
        else:
            validate_result(load_json(args.result), spec)
            print(json.dumps({"status": "ok", "result": str(args.result)}, ensure_ascii=False))
    except (ContractError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
