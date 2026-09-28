from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import yaml

from datasets.structural_profiler import structural_profile
from omni_ar.universal_adapter import (
    UniversalAdapterError, _task_spec, register_supplied_adapter,
    validate_blueprint, validate_candidate_adapter,
)


ROOT = Path(__file__).resolve().parents[2]


def blueprint() -> dict:
    return {
        "schema_version": "omni-ar-adapter-blueprint/v1",
        "task": {"type": "classification", "modality": "custom_binary", "description": "Classify custom records."},
        "fields": {"inputs": ["payload"], "label": "label", "sample_id": "id", "groups": [], "time": None},
        "split": {"strategy": "stratified", "validation_fraction": 0.2, "seed": 42},
        "baseline": {"method": "generated_baseline", "parameters": {}},
        "metrics": {
            "primary": {"name": "accuracy", "direction": "maximize", "role": "primary"},
            "secondary": [{"name": "log_loss", "direction": "minimize", "role": "diagnostic"}],
        },
        "search": {"methods": ["generated_baseline"], "parameters": {}},
        "activation_diagnostics": ["custom_reader_active"],
        "resources": {"gpu_count": 0, "max_runtime_minutes": 10},
    }


class UniversalAdapterTests(unittest.TestCase):
    def test_unknown_binary_gets_bounded_structure_without_task_guess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "records.unknown"
            source.write_bytes(b"PRIVATE\x00FORMAT" + bytes(range(128)))
            profile = structural_profile(source)
        self.assertEqual(profile["source_kind"], "file")
        self.assertFalse(profile["task_semantics_inferred"])
        self.assertEqual(profile["size_bytes"], 142)
        self.assertNotIn("label", profile)

    def test_blueprint_rejects_baseline_outside_search_space(self) -> None:
        value = blueprint()
        value["search"]["methods"] = ["another_method"]
        with self.assertRaisesRegex(UniversalAdapterError, "baseline method"):
            validate_blueprint(ROOT, value)

    def _candidate(self, *, omit_diagnostic: bool = False) -> tuple[Path, Path]:
        task_id = f"universal-test-{uuid.uuid4().hex[:10]}"
        task_dir = ROOT / "tasks" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "implementation/frozen").mkdir(parents=True)
        (task_dir / "implementation/model.py").write_text("\n", encoding="utf-8")
        spec = _task_spec(task_id, f"{task_id}@1", blueprint())
        spec_path = task_dir / "task_spec.pending.yaml"
        spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        class_name = spec["adapter"]["class_name"]
        diagnostics = {
            "data_reader_active": True, "split_active": True,
            "method_active": True, "baseline_active": True,
            "method": "generated_baseline",
        }
        if not omit_diagnostic:
            diagnostics["custom_reader_active"] = True
        source = f'''from tasks.adapter_contract import TaskAdapter, adapter_main

class {class_name}(TaskAdapter):
    task_name = {task_id.replace('-', '_')!r}
    trial_parameters = frozenset({{"dataset_ref", "method", "seed", "validation_fraction", "max_train_samples", "max_eval_samples", "smoke_mode"}})
    method_capabilities = frozenset({{"generated_baseline"}})
    def prepare_data(self, request): return {{"status": "ok"}}
    def validate_data(self, request):
        return {{"status": "ok", "sample_count": 10, "split_integrity": {{"status": "passed", "identity_overlap": 0, "content_hash_overlap": 0, "group_overlap": 0}}}}
    def run_baseline(self, request):
        return {{
            "status": "ok", "metrics": {{"accuracy": 0.5, "log_loss": 0.7}},
            "baseline_metrics": {{"accuracy": 0.4, "log_loss": 0.8}},
            "resource_usage": {{"runtime_seconds": 0.01, "gpu_count": 0, "within_budget": True, "train_samples": 8, "eval_samples": 2}},
            "protocol": {{"seed": 42, "split": "fixed", "split_fingerprint": "abc123", "stability": {{"status": "single_seed", "num_seeds": 1, "mean": 0.5, "std": None, "min": 0.5, "max": 0.5}}}},
            "artifacts": {{}}, "activation_diagnostics": {diagnostics!r}
        }}
    def run_trial(self, request): return self.run_baseline(request)
    def evaluate(self, request): return self.run_baseline(request)
    def summarize_results(self, request): return {{"status": "ok", "rows": []}}

if __name__ == "__main__": raise SystemExit(adapter_main({class_name}()))
'''
        (task_dir / "adapter.py").write_text(source, encoding="utf-8")
        return task_dir, spec_path

    def test_candidate_must_pass_all_seven_acceptance_checks(self) -> None:
        task_dir, spec_path = self._candidate()
        try:
            with tempfile.TemporaryDirectory() as output:
                result = validate_candidate_adapter(ROOT, spec_path, blueprint(), Path(output))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["checks"].values()), {"passed"})
        finally:
            shutil.rmtree(task_dir)

    def test_missing_activation_diagnostic_fails_closed(self) -> None:
        task_dir, spec_path = self._candidate(omit_diagnostic=True)
        try:
            with tempfile.TemporaryDirectory() as output:
                with self.assertRaisesRegex(UniversalAdapterError, "activation diagnostics failed"):
                    validate_candidate_adapter(ROOT, spec_path, blueprint(), Path(output))
        finally:
            shutil.rmtree(task_dir)

    def test_supplied_adapter_is_registered_only_after_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            task_dir = repository / "tasks/user-format"
            frozen = task_dir / "implementation/frozen"
            frozen.mkdir(parents=True)
            pending_spec = task_dir / "task_spec.pending.yaml"
            pending_spec.write_text("pending: true\n", encoding="utf-8")
            supplied = Path(directory) / "adapter.py"
            supplied.write_text("# supplied\n", encoding="utf-8")
            pending = {
                "status": "pending_coding", "dataset_ref": "user-format@1",
                "task_dir": str(task_dir), "pending_task_spec": str(pending_spec),
                "request": {}, "dataset_import": {"status": "ok"},
            }
            with (
                mock.patch("omni_ar.universal_adapter.validate_blueprint", return_value=blueprint()),
                mock.patch("omni_ar.universal_adapter.scaffold_pending_task", return_value=pending),
                mock.patch(
                    "omni_ar.universal_adapter.validate_candidate_adapter",
                    return_value={"status": "passed", "checks": {}},
                ),
            ):
                result = register_supplied_adapter(
                    repository, Path(directory) / "data.any", dataset_id="user-format",
                    version="1", blueprint=blueprint(), adapter_path=supplied,
                )
            self.assertTrue(result["registered"])
            self.assertEqual(result["provider"], "user")
            self.assertTrue((task_dir / "task_spec.yaml").is_file())
            self.assertFalse(pending_spec.exists())
            self.assertEqual((task_dir / "adapter.py").read_text(encoding="utf-8"), "# supplied\n")

    def test_failed_supplied_adapter_remains_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            task_dir = repository / "tasks/user-format"
            (task_dir / "implementation/frozen").mkdir(parents=True)
            pending_spec = task_dir / "task_spec.pending.yaml"
            pending_spec.write_text("pending: true\n", encoding="utf-8")
            supplied = Path(directory) / "adapter.py"
            supplied.write_text("# invalid supplied adapter\n", encoding="utf-8")
            pending = {
                "status": "pending_coding", "dataset_ref": "user-format@1",
                "task_dir": str(task_dir), "pending_task_spec": str(pending_spec),
                "request": {}, "dataset_import": {"status": "ok"},
            }
            with (
                mock.patch("omni_ar.universal_adapter.validate_blueprint", return_value=blueprint()),
                mock.patch("omni_ar.universal_adapter.scaffold_pending_task", return_value=pending),
                mock.patch(
                    "omni_ar.universal_adapter.validate_candidate_adapter",
                    side_effect=UniversalAdapterError("smoke failed"),
                ),
            ):
                with self.assertRaisesRegex(UniversalAdapterError, "failed mandatory acceptance"):
                    register_supplied_adapter(
                        repository, Path(directory) / "data.any", dataset_id="user-format",
                        version="1", blueprint=blueprint(), adapter_path=supplied,
                    )
            self.assertTrue(pending_spec.is_file())
            self.assertFalse((task_dir / "task_spec.yaml").exists())


if __name__ == "__main__":
    unittest.main()
