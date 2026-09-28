from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from datasets.datasetctl import import_dataset
from datasets.profiler import profile_dataset
from datasets.registry import DatasetRegistry


class TaskBootstrapError(ValueError):
    pass


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9-]+", "-", value.lower().replace("_", "-")).strip("-")
    if not result or not re.fullmatch(r"[a-z][a-z0-9-]*", result):
        raise TaskBootstrapError("dataset id must start with a letter and contain lowercase letters, digits, or hyphens")
    return result


def _class_name(dataset_id: str) -> str:
    return "".join(part.capitalize() for part in dataset_id.split("-")) + "TaskAdapter"


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _task_spec(dataset_id: str, dataset_ref: str, profile: dict[str, Any]) -> dict[str, Any]:
    supported = {
        ("classification", "image"), ("classification", "text"),
        ("classification", "tabular"), ("classification", "audio"),
        ("classification", "audio_text"),
        ("regression", "tabular"), ("regression", "time_series"),
        ("object_detection", "object_detection"),
        ("reconstruction", "single_cell"),
    }
    if (profile["task_type"], profile["modality"]) not in supported:
        raise TaskBootstrapError(
            "automatic executable baseline has no reviewed template for "
            f"detected {profile['task_type']}/{profile['modality']}"
        )
    method = str(profile["recommended_baseline"])
    module = f"tasks/{dataset_id}/adapter.py"
    # Trial workers run from the task's implementation directory. Generated
    # adapters live one level above it, including for hyphenated task names.
    command = "{python_executable} ../adapter.py"
    modality_defaults = {
        "text": {"validation_fraction": 0.2, "alpha": 1.0},
        "image": {"validation_fraction": 0.2, "pixel_stride": 1, "image_size": 16},
        "tabular": {"validation_fraction": 0.2, "ridge_alpha": 1.0},
        "time_series": {"validation_fraction": 0.2, "moving_average_window": 1},
        "audio": {"validation_fraction": 0.2},
        "audio_text": {
            "validation_fraction": 0.2, "alpha": 1.0,
            "fusion_text_weight": 0.5,
        },
        "object_detection": {"validation_fraction": 0.2},
        "single_cell": {"validation_fraction": 0.2, "max_train_samples": 2000, "max_eval_samples": 500},
    }[profile["modality"]]
    actions = {
        name: f"{command} {name} --output {{result_path}}"
        for name in ("prepare_data", "validate_data", "run_baseline", "run_trial", "evaluate", "summarize_results")
    }
    return {
        "schema_version": "omni-ar-task/v2",
        "task": {
            "name": dataset_id.replace("-", "_"), "type": profile["task_type"],
            "modality": profile["modality"],
            "description": (
                f"Automatically bootstrapped {profile['modality']} "
                f"{profile['task_type']} task for {dataset_ref}."
            ),
        },
        "workspace": {
            "base": "repository", "project_root": f"tasks/{dataset_id}",
            "implementation_dir": "implementation",
        },
        "adapter": {
            "module": module, "class_name": _class_name(dataset_id),
            "trial_defaults": {
                "dataset_ref": dataset_ref, "method": method, "seed": 42,
                **modality_defaults,
                "max_train_samples": modality_defaults.get("max_train_samples", 0),
                "max_eval_samples": modality_defaults.get("max_eval_samples", 0),
            },
            "actions": actions,
        },
        "entrypoints": {"train": actions["run_trial"], "evaluate": actions["evaluate"]},
        "data": {
            "dataset_ref": dataset_ref, "default_artifact": "raw",
            "bindings": {"runtime": {"artifact": "raw"}},
        },
        "metrics": {
            "primary": {
                **profile["recommended_metrics"][0],
                "source": f"metrics.{profile['recommended_metrics'][0]['name']}",
            },
            "secondary": [
                {**metric, "source": f"metrics.{metric['name']}"}
                for metric in profile["recommended_metrics"][1:]
            ],
        },
        "search": {
            "path_scope": "implementation", "editable_paths": ["model.py"],
            "protected_paths": ["frozen"], "interface_editable_paths": [module],
        },
        "resources": {"gpu_count": 0, "gpu_type": "CPU", "max_runtime_minutes": 10, "max_trials_per_round": 3},
    }


def bootstrap_dataset_task(
    repository: str | Path,
    source: str | Path,
    *,
    dataset_id: str | None = None,
    version: str = "1",
    copy_mode: str = "copy",
    run_baseline: bool = True,
) -> dict[str, Any]:
    repository = Path(repository).resolve()
    source_path = Path(source).expanduser().resolve()
    profile = profile_dataset(source_path)
    candidates = list(profile.get("task_candidates") or [])
    selected_candidate = next(
        (
            item for item in candidates
            if item.get("id") == profile.get("selected_task_candidate")
        ),
        candidates[0] if candidates else None,
    )
    if selected_candidate and not selected_candidate.get("reviewed_template", False):
        raise TaskBootstrapError(
            "the selected task has no reviewed Adapter template; "
            "a Coding Agent implementation must pass the Adapter contract before registration"
        )
    chosen_id = _slug(dataset_id or source_path.stem)
    dataset_ref = f"{chosen_id}@{version}"
    registry = DatasetRegistry(repository / "datasets")
    existing = next((item for item in registry.entries() if item.get("ref") == dataset_ref), None)
    if existing is None:
        args = argparse.Namespace(
            source=str(source_path), id=chosen_id, version=version, name=chosen_id,
            modality=profile["modality"], format=profile["format"], copy_mode=copy_mode,
            dry_run=False,
        )
        imported = import_dataset(args, registry, profile=profile)
    else:
        imported = {"status": "reused", "dataset_ref": dataset_ref, "pack_root": str(registry.pack_root(dataset_ref))}
        registered_profile = registry.manifest(dataset_ref).get("profile")
        comparable_registered = dict(registered_profile or {})
        comparable_supplied = dict(profile)
        comparable_registered.pop("source", None)
        comparable_supplied.pop("source", None)
        if comparable_registered != comparable_supplied:
            raise TaskBootstrapError(f"registered {dataset_ref} profile differs from the supplied source")

    pack_root = registry.pack_root(dataset_ref)
    manifest_path = registry.manifest_path(dataset_ref)
    manifest = registry.manifest(dataset_ref)
    if (profile.get("fields") or {}).get("test_features"):
        mode_name = "official_test"
    elif profile["modality"] == "time_series":
        mode_name = "chronological_holdout"
    elif profile["modality"] == "object_detection":
        mode_name = "deterministic_image_holdout"
    elif profile["modality"] == "single_cell":
        mode_name = "deterministic_heldout_cell"
    elif profile["task_type"] == "regression":
        mode_name = "deterministic_random_holdout"
    else:
        mode_name = "deterministic_stratified_holdout"
    usage_modes = {
        mode_name: {
            "description": (
                "Use the dataset-provided train/test arrays."
                if mode_name == "official_test"
                else f"Use the reviewed {mode_name} protocol implemented by the generic Task Adapter."
            ),
            "artifact": manifest["default_artifact"],
            "split": "test" if mode_name == "official_test" else "validation",
        }
    }
    manifest["usage_modes_file"] = "usage_modes.json"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (pack_root / "usage_modes.json").write_text(
        json.dumps(usage_modes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    task_dir = repository / "tasks" / chosen_id
    if task_dir.exists() and not (task_dir / ".omni_ar_generated_task").exists():
        raise TaskBootstrapError(f"refusing to overwrite non-generated task directory: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)
    implementation = task_dir / "implementation"
    frozen = implementation / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    class_name = _class_name(chosen_id)
    adapter_source = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "REPOSITORY = Path(__file__).resolve().parents[2]\n"
        "if str(REPOSITORY) not in sys.path:\n"
        "    sys.path.insert(0, str(REPOSITORY))\n\n"
        "from tasks.adapter_contract import adapter_main\n"
        "from tasks.generic.adapter import GenericSupervisedTaskAdapter\n\n"
        f"class {class_name}(GenericSupervisedTaskAdapter):\n"
        f"    task_name = {chosen_id.replace('-', '_')!r}\n"
        f"    dataset_ref = {dataset_ref!r}\n\n"
        "if __name__ == '__main__':\n"
        f"    raise SystemExit(adapter_main({class_name}()))\n"
    )
    (task_dir / "adapter.py").write_text(adapter_source, encoding="utf-8")
    compile(adapter_source, str(task_dir / "adapter.py"), "exec")
    (task_dir / "__init__.py").write_text("\"\"\"Automatically bootstrapped task package.\"\"\"\n", encoding="utf-8")
    (task_dir / ".omni_ar_generated_task").write_text("omni-ar-generated-task/v1\n", encoding="utf-8")
    (implementation / "model.py").write_text(
        '"""Editable model hooks for future Coding Agent candidates.\n\n'
        "The validated baseline stays in tasks.generic.adapter.\n"
        '"""\n', encoding="utf-8",
    )
    (frozen / "README.md").write_text("Generated task records in this directory are protected.\n", encoding="utf-8")
    spec = _task_spec(chosen_id, dataset_ref, profile)
    spec_path = task_dir / "task_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (task_dir / "task_context.md").write_text(
        f"# {chosen_id}\n\nDataset: `{dataset_ref}`\n\n"
        f"Detected modality/task: `{profile['modality']}/{profile['task_type']}`.\n\n"
        f"Baseline: `{profile['recommended_baseline']}`.\n",
        encoding="utf-8",
    )

    from controller_bash.scripts.task_contract import load_task_adapter, load_task_spec

    validated_spec = load_task_spec(spec_path)
    adapter = load_task_adapter(validated_spec, spec_path)
    validation = adapter.dispatch("validate_data", dict(validated_spec["adapter"]["trial_defaults"]))
    baseline = None
    baseline_candidates: list[dict[str, Any]] = []
    baseline_reused = False
    baseline_path = frozen / "baseline_result.json"
    bootstrap_path = frozen / "bootstrap.json"
    baseline_suite_fingerprint = _canonical_sha({
        "dataset_manifest": registry.manifest(dataset_ref),
        "profile": profile,
        "generic_adapter_sha256": hashlib.sha256(
            (repository / "tasks/generic/adapter.py").read_bytes()
        ).hexdigest(),
    })
    if run_baseline:
        previous = None
        if bootstrap_path.exists() and baseline_path.exists():
            previous = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        if (
            previous
            and previous.get("baseline_suite_fingerprint") == baseline_suite_fingerprint
            and previous.get("baseline_candidates")
        ):
            baseline_candidates = list(previous["baseline_candidates"])
            baseline_reused = True
        else:
            methods = list(profile.get("baseline_candidates") or [profile["recommended_baseline"]])
            for method in methods:
                request = {**validated_spec["adapter"]["trial_defaults"], "method": method}
                baseline_candidates.append(adapter.dispatch("run_baseline", request))
        primary_name = spec["metrics"]["primary"]["name"]
        for candidate in baseline_candidates:
            primary = (candidate.get("metrics") or {}).get(primary_name)
            if candidate.get("status") != "ok" or not isinstance(primary, (int, float)) or not math.isfinite(primary):
                raise TaskBootstrapError("generated baseline candidate did not produce a finite primary metric")
        direction = spec["metrics"]["primary"]["direction"]
        baseline = (
            max(baseline_candidates, key=lambda item: float(item["metrics"][primary_name]))
            if direction == "maximize"
            else min(baseline_candidates, key=lambda item: float(item["metrics"][primary_name]))
        )
        selected_method = str(baseline["method"])
        spec["adapter"]["trial_defaults"]["method"] = selected_method
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
        validated_spec = load_task_spec(spec_path)
        adapter = load_task_adapter(validated_spec, spec_path)
        if not baseline_reused:
            baseline_path.write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        (task_dir / "task_context.md").write_text(
            f"# {chosen_id}\n\nDataset: `{dataset_ref}`\n\n"
            f"Detected modality/task: `{profile['modality']}/{profile['task_type']}`.\n\n"
            f"Selected reviewed baseline: `{selected_method}`.\n",
            encoding="utf-8",
        )
    baseline_fingerprint = _canonical_sha({
        "baseline_suite_fingerprint": baseline_suite_fingerprint,
        "task_spec": validated_spec,
        "selected_method": baseline.get("method") if baseline else None,
    })
    record = {
        "schema_version": "omni-ar-task-bootstrap/v1", "status": "ok",
        "dataset_ref": dataset_ref, "dataset_profile": registry.manifest(dataset_ref)["profile"],
        "task_spec": str(spec_path.relative_to(repository)), "validation": validation,
        "baseline": baseline, "baseline_reused": baseline_reused,
        "baseline_candidates": baseline_candidates,
        "baseline_suite_fingerprint": baseline_suite_fingerprint,
        "baseline_fingerprint": baseline_fingerprint, "dataset_import": imported,
        "adapter_generation": {
            "route": "reviewed_template",
            "template": profile.get("adapter_template") or "generic_supervised",
            "selected_task_candidate": selected_candidate,
            "candidate_tasks": candidates,
            "checks": {
                "source_compiles": True,
                "task_schema_valid": True,
                "adapter_importable": True,
                "data_validation_passed": validation.get("status") == "ok",
                "split_integrity_passed": (
                    (validation.get("split_integrity") or {}).get("status") == "passed"
                ),
                "baseline_passed": baseline is None or baseline.get("status") == "ok",
            },
        },
    }
    bootstrap_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (frozen / "dataset_analysis.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (frozen / "adapter_generation.json").write_text(
        json.dumps(record["adapter_generation"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record
