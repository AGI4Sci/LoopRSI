from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml

from omni_ar.finalizer import (
    _archive_current_records,
    _archive_untracked_sources,
    _source_scope,
    refresh_records_artifact,
)
from omni_ar.project_records import ProjectRecords


class FinalizerScopeTests(unittest.TestCase):
    def test_source_archive_excludes_other_projects_tasks_and_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)

            task_root = repository / "tasks/current-task"
            dataset_root = repository / "datasets/current-data"
            project_dir = repository / "research_initializations/current/autoresearch"
            for directory in (
                task_root,
                dataset_root / "data",
                repository / "tasks/old-task",
                repository / "research_initializations/old-run",
                repository / "omni_ar",
                repository / "omni_ar/tests",
            ):
                directory.mkdir(parents=True, exist_ok=True)

            (task_root / "task_spec.yaml").write_text(
                yaml.safe_dump({"data": {"dataset_ref": "current-data@1"}}),
                encoding="utf-8",
            )
            (task_root / "adapter.py").write_text("CURRENT_TASK = True\n", encoding="utf-8")
            (task_root / "baseline_result.json").write_text("{}\n", encoding="utf-8")
            (dataset_root / "adapter.py").write_text("CURRENT_DATA = True\n", encoding="utf-8")
            (dataset_root / "dataset.yaml").write_text("dataset: {}\n", encoding="utf-8")
            (dataset_root / "data/raw.json").write_text('[{"secret": true}]\n', encoding="utf-8")
            (repository / "datasets/registry.yaml").write_text(
                yaml.safe_dump({
                    "datasets": [{
                        "ref": "current-data@1", "path": "current-data",
                        "manifest": "dataset.yaml",
                    }],
                }),
                encoding="utf-8",
            )
            (repository / "tasks/old-task/adapter.py").write_text("OLD = True\n", encoding="utf-8")
            (repository / "research_initializations/old-run/result.json").write_text(
                json.dumps({"old": True}), encoding="utf-8",
            )
            (repository / "omni_ar/runtime.py").write_text("FRAMEWORK = True\n", encoding="utf-8")
            (repository / "omni_ar/tests/test_runtime.py").write_text(
                "def test_runtime(): pass\n", encoding="utf-8",
            )

            source_paths, specific_roots = _source_scope(
                repository, task_root / "task_spec.yaml",
            )
            archive_path = project_dir / "final_package/artifacts/untracked_sources.tar.gz"
            result = _archive_untracked_sources(
                repository, project_dir, archive_path,
                source_paths=source_paths, specific_roots=specific_roots,
            )
            self.assertIsNotNone(result)
            files = set(result["files"])
            self.assertIn("tasks/current-task/adapter.py", files)
            self.assertIn("datasets/current-data/dataset.yaml", files)
            self.assertIn("omni_ar/runtime.py", files)
            self.assertNotIn("omni_ar/tests/test_runtime.py", files)
            self.assertNotIn("tasks/old-task/adapter.py", files)
            self.assertNotIn("research_initializations/old-run/result.json", files)
            self.assertNotIn("datasets/current-data/data/raw.json", files)
            self.assertNotIn("tasks/current-task/baseline_result.json", files)

    def test_records_archive_contains_only_current_record_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = ProjectRecords(root / "current")
            current.initialize({"task": "current"})
            other = ProjectRecords(root / "other")
            other.initialize({"task": "other"})

            destination = root / "current/final_package/artifacts/records.tar.gz"
            result = _archive_current_records(current, destination)
            with tarfile.open(destination, "r:gz") as archive:
                names = set(archive.getnames())

            self.assertIn("project.json", names)
            self.assertIn("archive.json", names)
            self.assertIn("events.jsonl", names)
            self.assertTrue(all("other" not in name for name in names))
            self.assertEqual(set(result["files"]), names)

            final_dir = root / "current/final_package"
            (final_dir / "manifest.json").write_text(
                json.dumps({"artifacts": {}, "records": {}}), encoding="utf-8",
            )
            current.stage("finalizer", "completed")
            refresh_records_artifact(final_dir, current)
            refreshed = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("records_archive", refreshed["artifacts"])
            self.assertIn(
                "records/stages.json",
                {item["path"] for item in refreshed["records"]["files"]},
            )


if __name__ == "__main__":
    unittest.main()
