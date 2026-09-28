from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets.feature_builders import FeatureBuilderRegistry
from datasets.registry import DatasetRegistry


class FeaturePlanningError(ValueError):
    pass


def run_coding_fallback(
    repository: str | Path, source: str | Path, *, dataset_id: str, version: str,
    blueprint: dict[str, Any], authorized: bool, mode: str = "plan",
    agent: str = "codex", model: str | None = None,
) -> dict[str, Any]:
    """Use the existing isolated Coding Agent path when no Builder matches."""
    if not authorized:
        raise FeaturePlanningError("Coding Agent feature fallback requires explicit code-change authorization")
    from .universal_adapter import generate_universal_adapter

    return generate_universal_adapter(
        Path(repository), Path(source), dataset_id=dataset_id, version=version,
        blueprint=blueprint, mode=mode, agent=agent, model=model,
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def plan_features(
    repository: str | Path, dataset_ref: str, *, rough_idea: str = "",
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic feature plan from registered data capabilities."""
    repository = Path(repository).resolve()
    datasets = DatasetRegistry(repository / "datasets")
    manifest = datasets.manifest(dataset_ref)
    modality = str((manifest.get("dataset") or {}).get("modality") or "unknown")
    builder_registry = FeatureBuilderRegistry(repository)
    compatible = builder_registry.compatible(modality)
    declared = dict(manifest.get("features") or {})
    feature_options = ["raw", *sorted(declared)]
    declared_builder_ids = {
        str((record.get("builder") or {}).get("id"))
        for record in declared.values() if isinstance(record, dict)
    }
    recommended_builds = [
        {
            "builder_id": str(entry["id"]),
            "status": (
                "already_declared" if str(entry["id"]) in declared_builder_ids
                else "authorization_required" if entry.get("external_authorization_required")
                else "ready"
            ),
            "materialization": entry.get("materialization", "builder_defined"),
        }
        for entry in compatible
    ]
    combinations: list[dict[str, Any]] = [{"id": "raw", "feature_ids": []}]
    for name in sorted(declared):
        record = declared[name] if isinstance(declared[name], dict) else {}
        bindings = list(record.get("task_bindings") or [])
        combinations.append({
            "id": name, "feature_ids": [name],
            **({"task_bindings": bindings} if bindings else {}),
        })
        for binding in bindings:
            combinations.append({
                "id": f"{name}:{binding['id']}", "feature_ids": [name],
                "task_method": binding["method"],
                "task_parameters": dict(binding.get("parameters") or {}),
                "role": str(binding.get("role") or "search_candidate"),
            })
        if modality == "audio_text" and name == "audio_text_basic_v1":
            combinations.extend([
                {"id": f"{name}:text", "feature_ids": [name], "feature_branch": "text"},
                {"id": f"{name}:audio", "feature_ids": [name], "feature_branch": "audio"},
                {"id": f"{name}:combined", "feature_ids": [name], "feature_branch": "combined"},
            ])
    fallback = None
    if not compatible:
        fallback = {
            "route": "coding_agent", "status": "requires_generation",
            "handler": "omni_ar.feature_planner.run_coding_fallback",
            "requires_user_confirmation": True,
            "acceptance_gates": [
                "interface_complete", "data_readable", "fixed_split_no_leakage",
                "bounded_smoke", "standard_result", "activation_diagnostics",
                "fail_closed_registration",
            ],
        }
    plan_body = {
        "dataset_ref": dataset_ref, "modality": modality,
        "rough_idea": rough_idea.strip(),
        "available_builders": [
            {key: value for key, value in entry.items() if key not in {"module", "class"}}
            for entry in compatible
        ],
        "recommended_builds": recommended_builds,
        "declared_features": declared, "feature_options": feature_options,
        "search_combinations": combinations, "fallback": fallback,
        "selection_policy": {
            "prefer_validated_existing": True,
            "external_sources_require_confirmation": True,
            "unknown_semantics_require_confirmation": True,
            "fit_scope_required": "training_partition_only_or_identifiers_only",
        },
    }
    plan_key = _canonical_hash({
        "manifest": manifest, "builders": builder_registry.entries(),
        "rough_idea": rough_idea.strip(), "planner_version": 1,
    })
    result = {
        "schema_version": "omni-ar-feature-plan/v1", "status": "ok",
        "plan_id": plan_key[:16], "cache_key": plan_key, **plan_body,
    }
    if output is not None:
        path = Path(output).resolve()
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("cache_key") == plan_key:
                previous["cache_status"] = "reused"
                return previous
        path.parent.mkdir(parents=True, exist_ok=True)
        result["cache_status"] = "created"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
