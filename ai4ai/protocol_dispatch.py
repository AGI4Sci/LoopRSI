"""Manifest-driven CandidateAdapter/Evaluator dispatcher."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .plugin_manifest import TaskPluginManifest
from .plugin_protocols import EvaluatorRequest
from .plugin_registry import load_entrypoint


class ProtocolDispatchError(RuntimeError):
    pass


def dispatch_native_protocol(
    manifest: TaskPluginManifest, package: dict[str, Any], context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one opt-in protocol lifecycle from manifest entrypoints."""
    context = dict(context or {})
    candidate_entrypoint = manifest.section("candidate").get("adapter")
    evaluator_entrypoint = manifest.section("evaluator").get("plugin")
    if not candidate_entrypoint or not evaluator_entrypoint:
        raise ProtocolDispatchError("manifest must declare candidate and evaluator entrypoints")
    candidate = load_entrypoint(str(candidate_entrypoint))()
    evaluator = load_entrypoint(str(evaluator_entrypoint))()
    if not candidate.validate_package(package).passed:
        raise ProtocolDispatchError("candidate package validation failed")

    if package.get("mode") == "existing_artifact_replay":
        prepare = getattr(candidate, "prepare_existing_artifact", None)
        materialize = getattr(candidate, "materialize_existing_artifact", None)
        if not callable(prepare) or not callable(materialize):
            raise ProtocolDispatchError("candidate does not support existing artifact replay")
        request = prepare(package["prediction_artifact"], package["contract_artifact"])
        candidate_result = materialize(request)
    else:
        request = candidate.prepare_execution(package, context)
        candidate_result = candidate.execute(request)
    candidate_validation = candidate.validate_artifacts(candidate_result)
    if not candidate_validation.passed:
        raise ProtocolDispatchError(f"candidate artifact validation failed: {candidate_validation.errors}")

    artifacts = list(candidate_result.produced_artifacts)
    prediction = package.get("prediction_artifact") or (artifacts[0] if artifacts else "")
    contract = package.get("contract_artifact") or prediction
    evaluator_request = EvaluatorRequest(
        task_id=manifest.task_id,
        prediction_artifact=str(prediction),
        contract_artifact=str(contract),
        output_location=str(package.get("output_location") or request.output_location),
        config=dict(package.get("evaluator_config") or {}),
        required_metrics=tuple(package.get("required_metrics") or ()),
        existing_evaluator_artifact=package.get("official_result_artifact"),
        existing_cell_eval_artifact=package.get("cell_eval_artifact"),
    )
    evaluator_input = evaluator.validate_input(evaluator_request)
    if not evaluator_input.passed:
        raise ProtocolDispatchError(f"evaluator input validation failed: {evaluator_input.errors}")
    evaluator_result = evaluator.evaluate(evaluator_request)
    normalized = evaluator.normalize(evaluator_result)
    payload = {
        "mode": "native_protocol",
        "task": manifest.task_id,
        "candidate": {
            "entrypoint": str(candidate_entrypoint),
            "return_code": candidate_result.return_code,
            "artifacts": list(candidate_result.produced_artifacts),
            "manifest": dict(candidate_result.artifact_manifest),
            "metadata": dict(candidate_result.execution_metadata),
        },
        "evaluator": {
            "entrypoint": str(evaluator_entrypoint),
            "status": normalized.status,
            "metrics": dict(normalized.metrics),
            "artifacts": dict(normalized.artifacts),
            "metric_completeness": normalized.metric_completeness,
            "validation": dict(normalized.validation),
            "metadata": dict(normalized.metadata),
            "archive_record": dict(evaluator.archive_record(normalized)),
        },
    }
    return _with_archive_and_lineage(payload, "native_protocol")


def dispatch_legacy_compatibility(
    manifest: TaskPluginManifest, package: dict[str, Any], context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke an explicitly declared task-local legacy compatibility shim."""
    entrypoint = manifest.section("compatibility").get("legacy_execution")
    if not entrypoint:
        raise ProtocolDispatchError("task does not declare a legacy compatibility shim")
    shim = load_entrypoint(str(entrypoint))()
    payload = shim.execute(package, dict(context or {}))
    if not isinstance(payload, dict):
        raise ProtocolDispatchError("legacy compatibility shim must return a mapping")
    payload["mode"] = "legacy"
    payload["compatibility_entrypoint"] = str(entrypoint)
    return _with_archive_and_lineage(payload, "legacy")


def _with_archive_and_lineage(payload: dict[str, Any], execution_mode: str) -> dict[str, Any]:
    evaluator = dict(payload.get("evaluator") or {})
    archive = dict(evaluator.get("archive_record") or {})
    archive["execution_mode"] = execution_mode
    payload["archive"] = archive
    payload["lineage"] = {
        "task": payload.get("task"),
        "execution_mode": execution_mode,
        "candidate_entrypoint": (payload.get("candidate") or {}).get("entrypoint"),
        "evaluator_entrypoint": evaluator.get("entrypoint"),
        "status": evaluator.get("status"),
    }
    return payload


def load_and_dispatch(manifest_path: str | Path, package_path: str | Path) -> dict[str, Any]:
    from .plugin_manifest import load_task_plugin_manifest
    manifest = load_task_plugin_manifest(manifest_path)
    package = json.loads(Path(package_path).read_text(encoding="utf-8"))
    return dispatch_native_protocol(manifest, package)


def dispatch_protocol(
    manifest_path: str | Path,
    package_path: str | Path,
    execution_mode: str = "native_protocol",
) -> dict[str, Any]:
    from .plugin_manifest import load_task_plugin_manifest
    manifest = load_task_plugin_manifest(manifest_path)
    package = json.loads(Path(package_path).read_text(encoding="utf-8"))
    if execution_mode == "native_protocol":
        return dispatch_native_protocol(manifest, package)
    if execution_mode == "legacy":
        return dispatch_legacy_compatibility(manifest, package)
    raise ProtocolDispatchError(f"unsupported execution mode: {execution_mode}")
