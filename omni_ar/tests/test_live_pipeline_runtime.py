from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from omni_ar.initialization import InitializationError
from omni_ar.live_pipeline import _cache_artifacts, _cache_valid, _run_external_stage


class LivePipelineRuntimeTests(unittest.TestCase):
    def test_external_stage_writes_completed_progress_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _run_external_stage(
                [sys.executable, "-c", "print('finished')"], cwd=root, env={},
                output_dir=root, stage="boyue", timeout_s=5,
                stdout_name="stdout.log", stderr_name="stderr.log",
            )
            progress = json.loads((root / "planning_progress.json").read_text())
            self.assertEqual(progress["stages"]["boyue"]["status"], "completed")
            self.assertIn("finished", (root / "stdout.log").read_text())

    def test_external_stage_has_a_hard_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = time.monotonic()
            with self.assertRaisesRegex(InitializationError, "exceeded"):
                _run_external_stage(
                    [sys.executable, "-c", "import time; time.sleep(5)"],
                    cwd=root, env={}, output_dir=root, stage="researchstudio",
                    timeout_s=0.2, stdout_name="stdout.log", stderr_name="stderr.log",
                )
            self.assertLess(time.monotonic() - started, 2.0)
            progress = json.loads((root / "planning_progress.json").read_text())
            self.assertEqual(progress["stages"]["researchstudio"]["status"], "failed")

    def test_timeout_cleans_detached_process_in_run_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_code = "import time; time.sleep(30)"
            parent_code = (
                "import subprocess,sys,time; "
                "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],"
                "start_new_session=True); time.sleep(30)"
            )
            with self.assertRaises(InitializationError):
                _run_external_stage(
                    [sys.executable, "-c", parent_code, child_code, str(root)],
                    cwd=root, env={}, output_dir=root, stage="researchstudio",
                    timeout_s=0.2, stdout_name="stdout.log", stderr_name="stderr.log",
                )
            marker = str(root).encode()
            scoped = []
            for item in Path("/proc").iterdir():
                if not item.name.isdigit():
                    continue
                try:
                    if marker in (item / "cmdline").read_bytes():
                        scoped.append(item.name)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
            self.assertEqual(scoped, [])

    def test_cache_requires_matching_hashes_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "proposal.json"
            artifact.write_text('{"status":"ok"}\n', encoding="utf-8")
            cache = {"stages": {"heuresis": {
                "status": "completed", "context_sha256": "ctx",
                "artifacts": _cache_artifacts([artifact]),
            }}}
            self.assertTrue(_cache_valid(
                cache, "heuresis", [artifact], context_sha256="ctx",
            ))
            artifact.write_text('{"status":"changed"}\n', encoding="utf-8")
            self.assertFalse(_cache_valid(
                cache, "heuresis", [artifact], context_sha256="ctx",
            ))


if __name__ == "__main__":
    unittest.main()
