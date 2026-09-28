from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "controller_bash/scripts"
sys.path.insert(0, str(SCRIPTS))

import execution_backends as backends  # noqa: E402


class RjobExecutionBackendTest(unittest.TestCase):
    def environment(self, source: Path, shared: Path) -> dict[str, str]:
        return {
            "RJOB_NAMESPACE": "test-namespace",
            "RJOB_SYNC_SOURCE": str(source),
            "RJOB_SHARED_FOLDER": str(shared),
            "RJOB_GPU_LIMIT": "1",
            "RJOB_GPU_PER_TRIAL": "1",
            "RJOB_CPU": "8",
            "RJOB_MEMORY": "16000",
            "RJOB_POLL_INTERVAL_SEC": "10",
            "RJOB_TIMEOUT_MIN": "1",
            "RJOB_TAIL_LINES": "100",
            "RJOB_LOG_RETRIES": "1",
            "RJOB_LOG_RETRY_INTERVAL_SEC": "1",
            "RJOB_CANCEL_ON_TIMEOUT": "1",
            "TRIAL_WORKDIR_IN_PACKAGE": "implementation",
            "TRIAL_PYTHONPATH_IN_PACKAGE": ".",
        }

    def request(self, source: Path) -> backends.ExecutionRequest:
        trial_root = source / "artifacts/controller_trials/round_0"
        trial_root.mkdir(parents=True)
        return backends.ExecutionRequest(
            job_name="contract-job",
            command="python train.py --output artifacts/result.json",
            worker_script=trial_root / "trial_000_worker.sh",
            output=trial_root / "trial_000.json",
            stdout_path=trial_root / "trial_000.stdout.log",
            stderr_path=trial_root / "trial_000.stderr.log",
            trial_root=trial_root,
            resource_request={
                "gpu_count": 1, "cpu": 4, "memory_mb": 8000,
                "max_runtime_minutes": 1,
            },
            trial={"name": "metadata_only"},
        )

    def test_dry_run_plans_sync_and_submit_without_external_calls(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, shared = root / "source", root / "shared"
            source.mkdir()
            request = self.request(source)
            with patch.dict(os.environ, self.environment(source, shared), clear=True):
                backend = backends.RjobExecutionBackend(backends.RjobConfig.from_env(request.trial_root))
                with patch.object(backends, "run_capture") as runner:
                    record = backend.execute(request, dry_run=True)
            runner.assert_not_called()
            self.assertEqual(record["status"], "skipped")
            self.assertEqual(record["code_sync"]["command"][0], "rsync")
            worker_relative = str(request.worker_script.relative_to(source))
            self.assertIn(f"./{worker_relative}", record["code_sync"]["command"])
            self.assertIn(str(shared), record["command"])
            self.assertTrue(request.worker_script.exists())
            self.assertEqual(json.loads(request.output.read_text())["status"], "rjob_dry_run")

    def test_success_lifecycle_syncs_submits_polls_logs_and_recovers_result(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, shared = root / "source", root / "shared"
            source.mkdir()
            request = self.request(source)
            responses = [
                subprocess.CompletedProcess([], 0, "synced", ""),
                subprocess.CompletedProcess([], 0, "created rjob_name: actual-job", ""),
                subprocess.CompletedProcess([], 0, "rjob actual-job: Succeeded", ""),
                subprocess.CompletedProcess(
                    [], 0,
                    'worker >> {"status":"ok","metrics":{"quality":0.75},"seed":7}\n',
                    "",
                ),
            ]
            with patch.dict(os.environ, self.environment(source, shared), clear=True):
                backend = backends.RjobExecutionBackend(backends.RjobConfig.from_env(request.trial_root))
                with patch.object(backends, "run_capture", side_effect=responses) as runner:
                    record = backend.execute(request)
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["job_name"], "actual-job")
            self.assertTrue(record["result_recovered_from_rjob_log"])
            self.assertEqual(json.loads(request.output.read_text())["metrics"]["quality"], 0.75)
            calls = [call.args[0] for call in runner.call_args_list]
            self.assertEqual(calls[0][0], "rsync")
            self.assertEqual(calls[1][:2], ["rjob", "submit"])
            self.assertEqual(calls[2][:2], ["rjob", "get"])
            self.assertEqual(calls[3][:3], ["rjob", "logs", "job"])

    def test_timeout_is_logged_and_cancelled_by_backend(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, shared = root / "source", root / "shared"
            source.mkdir()
            request = self.request(source)
            responses = [
                subprocess.CompletedProcess([], 0, "synced", ""),
                subprocess.CompletedProcess([], 0, "created rjob_name: actual-job", ""),
            ]
            with patch.dict(os.environ, self.environment(source, shared), clear=True):
                backend = backends.RjobExecutionBackend(backends.RjobConfig.from_env(request.trial_root))
                with (
                    patch.object(backends, "run_capture", side_effect=responses),
                    patch.object(backend, "poll", return_value=("timeout", 60.0)),
                    patch.object(backend, "fetch_logs", return_value=0),
                    patch.object(backend, "cancel_timeout", return_value=0) as cancel,
                ):
                    record = backend.execute(request)
            cancel.assert_called_once_with("actual-job", Path(record["rjob_timeout_cancel_log"]))
            self.assertEqual(record["status"], "timeout")
            self.assertEqual(record["timeout_cancel_returncode"], 0)
            self.assertEqual(json.loads(request.output.read_text())["status"], "timeout")

    def test_submit_failure_becomes_machine_readable_failure(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, shared = root / "source", root / "shared"
            source.mkdir()
            request = self.request(source)
            responses = [
                subprocess.CompletedProcess([], 0, "synced", ""),
                subprocess.CompletedProcess([], 9, "", "quota denied"),
            ]
            with patch.dict(os.environ, self.environment(source, shared), clear=True):
                backend = backends.RjobExecutionBackend(backends.RjobConfig.from_env(request.trial_root))
                with patch.object(backends, "run_capture", side_effect=responses):
                    record = backend.execute(request)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["submit_returncode"], 9)
            raw = json.loads(request.output.read_text())
            self.assertEqual(raw["status"], "failed")
            self.assertIn("submit", raw["error"])

    def test_resource_request_cannot_exceed_controller_limits(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, shared = root / "source", root / "shared"
            source.mkdir()
            request = self.request(source)
            with patch.dict(os.environ, self.environment(source, shared), clear=True):
                backend = backends.RjobExecutionBackend(backends.RjobConfig.from_env(request.trial_root))
                with self.assertRaises(backends.ExecutionBackendError):
                    backend.submit_command(
                        "too-large", request.worker_script,
                        {"gpu_count": 2, "cpu": 4, "memory_mb": 8000},
                    )

    def test_task_code_contains_no_submission_flow(self) -> None:
        roots = [
            ROOT / "tasks",
            ROOT / "case03_vcc25/runs/e2e_vcc25_20260806/implementation",
            ROOT / "case04_six_method_benchmark",
        ]
        offenders = []
        for root in roots:
            for path in root.rglob("*"):
                relative_parts = set(path.relative_to(root).parts)
                if relative_parts.intersection({"frozen", "__pycache__", ".venv", ".venv_extra"}):
                    continue
                if path.is_file() and path.suffix == ".sh":
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if "rjob" + " submit" in text:
                        offenders.append(str(path))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
