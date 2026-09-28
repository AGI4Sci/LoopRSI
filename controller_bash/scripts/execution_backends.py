#!/usr/bin/env python3
"""Infrastructure-only execution backends used by the generic controller.

Task adapters provide a payload command. They never submit, poll, cancel, or
inspect rjob jobs. The backend owns the complete infrastructure lifecycle.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExecutionBackendError(RuntimeError):
    pass


class ExecutionBackend(ABC):
    """Infrastructure execution boundary consumed by the controller."""

    name: str

    @abstractmethod
    def execute(self, request: "ExecutionRequest", dry_run: bool = False) -> dict[str, Any]:
        raise NotImplementedError


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ExecutionBackendError(f"{name} must be an integer, got {raw!r}") from exc


def run_capture(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, text=True,
        capture_output=True, check=False,
    )


@dataclass(frozen=True)
class RjobConfig:
    namespace: str
    sync_source: Path
    shared_folder: Path
    charged_group: str
    image: str
    mount: str
    private_machine: str
    host_network: str
    gpu_limit: int
    default_gpu: int
    cpu_limit: int
    memory_limit_mb: int
    poll_interval_seconds: int
    timeout_minutes: int
    tail_lines: int
    log_retries: int
    log_retry_interval_seconds: int
    cancel_on_timeout: bool
    submit_retries: int
    submit_retry_interval_seconds: int
    sync_excludes: tuple[str, ...]
    sync_paths: tuple[str, ...]

    @classmethod
    def from_env(cls, trial_root: Path) -> "RjobConfig":
        namespace = os.environ.get("RJOB_NAMESPACE", "").strip()
        if not namespace:
            raise ExecutionBackendError("RJOB_NAMESPACE is required")
        legacy_folder = os.environ.get("RJOB_FOLDER", "").strip()
        source_raw = os.environ.get("RJOB_SYNC_SOURCE", "").strip() or legacy_folder
        if not source_raw:
            source_raw = str(trial_root)
        shared_raw = os.environ.get("RJOB_SHARED_FOLDER", "").strip() or legacy_folder or source_raw
        default_excludes = (".git/", ".venv/", "__pycache__/", "*.pyc")
        configured_excludes = tuple(
            item for item in os.environ.get("RJOB_SYNC_EXCLUDES", "").split(":") if item
        )
        sync_paths = tuple(
            item.strip("/") or "."
            for item in os.environ.get("RJOB_SYNC_PATHS", ".").split(":")
            if item
        )
        if not sync_paths:
            raise ExecutionBackendError("RJOB_SYNC_PATHS must contain at least one relative path")
        if any(path == ".." or path.startswith("../") or "/../" in path for path in sync_paths):
            raise ExecutionBackendError("RJOB_SYNC_PATHS entries must stay inside RJOB_SYNC_SOURCE")
        return cls(
            namespace=namespace,
            sync_source=Path(source_raw).resolve(),
            shared_folder=Path(shared_raw).resolve(),
            charged_group=os.environ.get("RJOB_CHARGED_GROUP", "").strip(),
            image=os.environ.get("RJOB_IMAGE", "").strip(),
            mount=os.environ.get("RJOB_MOUNT", "").strip(),
            private_machine=os.environ.get("RJOB_PRIVATE_MACHINE", "").strip(),
            host_network=os.environ.get("RJOB_HOST_NETWORK", "").strip(),
            gpu_limit=env_int("RJOB_GPU_LIMIT", 2),
            default_gpu=env_int("RJOB_GPU_PER_TRIAL", 1),
            cpu_limit=env_int("RJOB_CPU", 20),
            memory_limit_mb=env_int("RJOB_MEMORY", 20000),
            poll_interval_seconds=max(10, env_int("RJOB_POLL_INTERVAL_SEC", 60)),
            timeout_minutes=max(1, env_int("RJOB_TIMEOUT_MIN", 180)),
            tail_lines=max(1, env_int("RJOB_TAIL_LINES", 500)),
            log_retries=max(1, env_int("RJOB_LOG_RETRIES", 3)),
            log_retry_interval_seconds=max(1, env_int("RJOB_LOG_RETRY_INTERVAL_SEC", 10)),
            cancel_on_timeout=os.environ.get("RJOB_CANCEL_ON_TIMEOUT", "1") == "1",
            submit_retries=max(1, env_int("RJOB_SUBMIT_RETRIES", 3)),
            submit_retry_interval_seconds=max(1, env_int("RJOB_SUBMIT_RETRY_INTERVAL_SEC", 10)),
            sync_excludes=tuple(dict.fromkeys(default_excludes + configured_excludes)),
            sync_paths=sync_paths,
        )


@dataclass(frozen=True)
class ExecutionRequest:
    job_name: str
    command: str
    worker_script: Path
    output: Path
    stdout_path: Path
    stderr_path: Path
    trial_root: Path
    resource_request: dict[str, Any]
    trial: dict[str, Any]


class RjobExecutionBackend(ExecutionBackend):
    name = "rjob"

    def __init__(self, config: RjobConfig) -> None:
        self.config = config
        if (
            config.sync_source != config.shared_folder
            and config.sync_source in config.shared_folder.parents
        ):
            raise ExecutionBackendError("RJOB_SHARED_FOLDER cannot be nested inside RJOB_SYNC_SOURCE")

    def write_worker_script(self, path: Path, command: str) -> None:
        workdir = (
            os.environ.get("TRIAL_WORKDIR_IN_PACKAGE")
            or os.environ.get("TRIAL_WORKDIR")
            or os.environ.get("IMPLEMENTATION_DIR", "")
        )
        pythonpath = (
            os.environ.get("TRIAL_PYTHONPATH_IN_PACKAGE")
            or os.environ.get("TRIAL_PYTHONPATH")
            or os.environ.get("IMPLEMENTATION_DIR", "")
        )
        if workdir and pythonpath == workdir:
            pythonpath = "."
        lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
        if workdir:
            lines.append(f"cd {shlex.quote(workdir)}")
        if pythonpath:
            lines.append(f"export PYTHONPATH={shlex.quote(pythonpath)}:${{PYTHONPATH:-}}")
        lines += [
            'echo "[trial-worker] start $(date -Is)"',
            command,
            'echo "[trial-worker] done $(date -Is)"',
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o755)

    def sync_command(self, additional_paths: tuple[str, ...] = ()) -> list[str] | None:
        if self.config.sync_source == self.config.shared_folder:
            return None
        command = ["rsync", "-aR"]
        for pattern in self.config.sync_excludes:
            command.extend(["--exclude", pattern])
        paths = tuple(dict.fromkeys((*self.config.sync_paths, *additional_paths)))
        command += [f"./{path}" if path != "." else "." for path in paths]
        command.append(f"{self.config.shared_folder}/")
        return command

    def sync_code(
        self, log_prefix: Path, dry_run: bool = False,
        additional_paths: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        if not self.config.sync_source.is_dir():
            raise ExecutionBackendError(f"RJOB_SYNC_SOURCE is not a directory: {self.config.sync_source}")
        command = self.sync_command(additional_paths)
        if command is None:
            return {
                "status": "already_shared", "source": str(self.config.sync_source),
                "destination": str(self.config.shared_folder), "command": None,
            }
        record = {
            "status": "planned" if dry_run else "pending",
            "source": str(self.config.sync_source),
            "destination": str(self.config.shared_folder),
            "command": command,
            "cwd": str(self.config.sync_source),
        }
        if dry_run:
            return record
        self.config.shared_folder.parent.mkdir(parents=True, exist_ok=True)
        proc = run_capture(command, cwd=self.config.sync_source)
        log_prefix.with_suffix(".stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
        log_prefix.with_suffix(".stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")
        record["returncode"] = proc.returncode
        record["status"] = "ok" if proc.returncode == 0 else "failed"
        if proc.returncode:
            raise ExecutionBackendError(
                f"code sync failed with exit code {proc.returncode}; see {log_prefix.with_suffix('.stderr.log')}"
            )
        return record

    def worker_entry(self, worker_script: Path) -> Path:
        try:
            return worker_script.resolve().relative_to(self.config.sync_source)
        except ValueError as exc:
            raise ExecutionBackendError(
                f"worker script {worker_script} must be inside RJOB_SYNC_SOURCE={self.config.sync_source}"
            ) from exc

    def submit_command(
        self, job_name: str, worker_script: Path, resource_request: dict[str, Any]
    ) -> list[str]:
        gpu = int(resource_request.get("gpu_count", self.config.default_gpu))
        cpu = int(resource_request.get("cpu", self.config.cpu_limit))
        memory = int(resource_request.get("memory_mb", self.config.memory_limit_mb))
        if gpu < 0 or gpu > self.config.gpu_limit:
            raise ExecutionBackendError(f"requested gpu_count={gpu} exceeds controller limit {self.config.gpu_limit}")
        if cpu < 1 or cpu > self.config.cpu_limit:
            raise ExecutionBackendError(f"requested cpu={cpu} exceeds controller limit {self.config.cpu_limit}")
        if memory < 1 or memory > self.config.memory_limit_mb:
            raise ExecutionBackendError(
                f"requested memory_mb={memory} exceeds controller limit {self.config.memory_limit_mb}"
            )
        command = [
            "rjob", "submit", "--namespace", self.config.namespace,
            "--name", job_name, "--folder", str(self.config.shared_folder),
        ]
        optional = (
            (self.config.mount, f"--mount={self.config.mount}"),
            (self.config.charged_group, f"--charged-group={self.config.charged_group}"),
            (self.config.private_machine, f"--private-machine={self.config.private_machine}"),
            (self.config.image, f"--image={self.config.image}"),
            (self.config.host_network, f"--host-network={self.config.host_network}"),
        )
        command.extend(flag for value, flag in optional if value)
        command += [
            "--cpu", str(cpu), "--memory", str(memory), "--gpu", str(gpu),
            "--", "bash", str(self.worker_entry(worker_script)),
        ]
        return command

    def poll(self, job_name: str, log_path: Path, timeout_minutes: int) -> tuple[str, float]:
        timeout_seconds = max(1, timeout_minutes) * 60
        started = time.time()
        snapshots: list[str] = []
        while True:
            proc = run_capture(["rjob", "get", job_name, "--namespace", self.config.namespace])
            elapsed = time.time() - started
            snapshots.append(
                f"\n--- poll {len(snapshots)} elapsed={elapsed:.1f}s rc={proc.returncode} ---\n"
                f"{proc.stdout}\n{proc.stderr}"
            )
            log_path.write_text("".join(snapshots), encoding="utf-8", errors="replace")
            status = classify_rjob_status(proc.stdout + "\n" + proc.stderr)
            if status != "running":
                return status, elapsed
            if elapsed >= timeout_seconds:
                return "timeout", elapsed
            time.sleep(self.config.poll_interval_seconds)

    def fetch_logs(self, job_name: str, log_path: Path) -> int:
        chunks: list[str] = []
        last_returncode = 1
        for attempt in range(self.config.log_retries):
            proc = run_capture([
                "rjob", "logs", "job", job_name, "--namespace", self.config.namespace,
                "--tail-lines", str(self.config.tail_lines),
            ])
            chunks.append(f"\n--- logs attempt {attempt} rc={proc.returncode} ---\n{proc.stdout}{proc.stderr}")
            log_path.write_text("".join(chunks), encoding="utf-8", errors="replace")
            last_returncode = proc.returncode
            if proc.returncode == 0 and "NoneType" not in (proc.stdout + proc.stderr):
                break
            if attempt + 1 < self.config.log_retries:
                time.sleep(self.config.log_retry_interval_seconds)
        return last_returncode

    def cancel_timeout(self, job_name: str, log_path: Path) -> int | None:
        if not self.config.cancel_on_timeout:
            return None
        proc = run_capture(["rjob", "delete", job_name, "--namespace", self.config.namespace])
        log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")
        return proc.returncode

    def execute(self, request: ExecutionRequest, dry_run: bool = False) -> dict[str, Any]:
        self.write_worker_script(request.worker_script, request.command)
        worker_entry = str(self.worker_entry(request.worker_script))
        sync = self.sync_code(
            request.trial_root / "code_sync",
            dry_run=dry_run,
            additional_paths=(worker_entry,),
        )
        submit_command = self.submit_command(
            request.job_name, request.worker_script, request.resource_request
        )
        paths = {
            "rjob_submit_stdout": request.trial_root / f"{request.worker_script.stem}.rjob_submit.stdout.log",
            "rjob_submit_stderr": request.trial_root / f"{request.worker_script.stem}.rjob_submit.stderr.log",
            "rjob_get_log": request.trial_root / f"{request.worker_script.stem}.rjob_get.log",
            "rjob_log": request.trial_root / f"{request.worker_script.stem}.rjob.log",
            "rjob_timeout_cancel_log": request.trial_root / f"{request.worker_script.stem}.rjob_timeout_cancel.log",
        }
        request.stdout_path.write_text("", encoding="utf-8")
        request.stderr_path.write_text("", encoding="utf-8")
        record: dict[str, Any] = {
            "mode": "rjob_dry_run" if dry_run else "rjob",
            "trial": request.trial,
            "command": submit_command,
            "worker_script": str(request.worker_script),
            "job_name": request.job_name,
            "requested_job_name": request.job_name,
            "output": str(request.output),
            "stdout": str(request.stdout_path),
            "stderr": str(request.stderr_path),
            "code_sync": sync,
            **{key: str(value) for key, value in paths.items()},
        }
        if dry_run:
            record.update(status="skipped", reason="TRIAL_EXECUTION_MODE=rjob_dry_run")
            request.output.write_text(json.dumps({
                "status": "rjob_dry_run", "trial": request.trial,
                "job_name": request.job_name, "worker_script": str(request.worker_script),
                "sync_command": sync.get("command"),
                "submit_command": " ".join(shlex.quote(part) for part in submit_command),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            return record

        request.output.write_text(json.dumps({
            "status": "pending", "trial": request.trial, "mode": "rjob",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        started = time.time()
        submit_attempts = []
        for attempt in range(self.config.submit_retries):
            proc = run_capture(submit_command, cwd=request.trial_root)
            submit_attempts.append({
                "attempt": attempt + 1, "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
            })
            if proc.returncode == 0 or not is_transient_submit_failure(proc.stdout + "\n" + proc.stderr):
                break
            if attempt + 1 < self.config.submit_retries:
                time.sleep(self.config.submit_retry_interval_seconds)
        paths["rjob_submit_stdout"].write_text(
            "".join(f"--- attempt {row['attempt']} ---\n{row['stdout']}\n" for row in submit_attempts),
            encoding="utf-8", errors="replace",
        )
        paths["rjob_submit_stderr"].write_text(
            "".join(f"--- attempt {row['attempt']} ---\n{row['stderr']}\n" for row in submit_attempts),
            encoding="utf-8", errors="replace",
        )
        record["submit_attempts"] = len(submit_attempts)
        record["submit_returncode"] = proc.returncode
        if proc.returncode:
            record.update(status="failed", error="rjob submit failed", elapsed_sec=time.time() - started)
            request.output.write_text(json.dumps({
                "status": "failed", "error": record["error"], "trial": request.trial,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            return record

        actual_name = parse_actual_rjob_name(proc.stdout + "\n" + proc.stderr, request.job_name)
        record["job_name"] = actual_name
        requested_timeout = int(request.resource_request.get(
            "max_runtime_minutes", self.config.timeout_minutes
        ))
        status, elapsed = self.poll(actual_name, paths["rjob_get_log"], requested_timeout)
        record["rjob_logs_returncode"] = self.fetch_logs(actual_name, paths["rjob_log"])
        if status == "timeout":
            record["timeout_cancel_returncode"] = self.cancel_timeout(
                actual_name, paths["rjob_timeout_cancel_log"]
            )
        record.update(status=status, elapsed_sec=elapsed, returncode=0 if status == "ok" else 1)
        if status == "ok":
            recovered = recover_json_result_from_rjob_log(paths["rjob_log"])
            # Kubebrain can report Succeeded before the final worker log line
            # reaches the log service. Re-fetch briefly before declaring that
            # a successful job emitted no machine-readable result.
            for _ in range(3):
                if recovered is not None:
                    break
                time.sleep(2)
                record["rjob_logs_returncode"] = self.fetch_logs(actual_name, paths["rjob_log"])
                recovered = recover_json_result_from_rjob_log(paths["rjob_log"])
            if recovered is not None:
                request.output.write_text(
                    json.dumps(recovered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                record["result_recovered_from_rjob_log"] = True
            elif json.loads(request.output.read_text(encoding="utf-8")).get("status") == "pending":
                record.update(status="failed", returncode=1, error="rjob succeeded but emitted no machine-readable result")
                request.output.write_text(json.dumps({
                    "status": "failed", "error": record["error"], "trial": request.trial,
                    "job_name": actual_name,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            request.output.write_text(json.dumps({
                "status": status, "trial": request.trial, "job_name": actual_name,
                "elapsed_sec": elapsed, "rjob_log": str(paths["rjob_log"]),
                "rjob_get_log": str(paths["rjob_get_log"]),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
        return record


def parse_actual_rjob_name(submit_text: str, requested_name: str) -> str:
    for pattern in (
        r"created rjob_name:\s*([A-Za-z0-9_.-]+)",
        r"use\s+([A-Za-z0-9_.-]+)\s+as metadata name",
    ):
        match = re.search(pattern, submit_text)
        if match:
            return match.group(1)
    return requested_name


def is_transient_submit_failure(text: str) -> bool:
    """Retry only failures known to happen before a job is created."""
    lowered = text.lower()
    return any(token in lowered for token in (
        "temporarily unavailable", "connection reset", "connection refused",
        "i/o timeout", "timed out", "service unavailable", "too many requests",
    ))


def classify_rjob_status(text: str) -> str:
    ok_patterns = (
        r"\brjob\s+\S+.*:\s*Succeeded\b",
        r"\breplica\s+\S+:\s*(SUCCEED|Succeeded)\b",
    )
    fail_patterns = (
        r"\brjob\s+\S+.*:\s*(Failed|Error|Cancelled|Canceled|Deleted|Killed)\b",
        r"\breplica\s+\S+:\s*(FAIL|FAILED|ERROR|CANCELLED|CANCELED|DELETED|KILLED|OOM)\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in ok_patterns):
        return "ok"
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in fail_patterns):
        return "failed"
    return "running"


def recover_json_result_from_rjob_log(log_path: Path) -> dict[str, Any] | None:
    if not log_path.exists():
        return None
    candidates: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            candidates.append(value)
    for value in reversed(candidates):
        if str(value.get("status", "")).lower() == "ok" and isinstance(value.get("metrics"), dict):
            return value
    return candidates[-1] if candidates else None
