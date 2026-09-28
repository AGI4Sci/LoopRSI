"""Task-agnostic candidate execution boundary."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .plugin_protocols import CandidateExecutionRequest, CandidateExecutionResult


def execute_candidate(request: CandidateExecutionRequest) -> CandidateExecutionResult:
    started = time.monotonic()
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in request.environment.items()})
    proc = subprocess.run(
        list(request.command), cwd=request.working_directory, env=env,
        text=True, capture_output=True, check=False,
    )
    produced_paths = []
    for raw in request.expected_artifacts:
        path = Path(raw)
        if not path.is_absolute():
            path = Path(request.working_directory) / path
        if path.exists():
            produced_paths.append(str(path))
    produced = tuple(produced_paths)
    return CandidateExecutionResult(
        candidate_id=request.candidate_id,
        return_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        execution_metadata={
            "working_directory": request.working_directory,
            "command": list(request.command),
            "runtime_seconds": time.monotonic() - started,
        },
        produced_artifacts=produced,
        artifact_manifest={
            "output_location": request.output_location,
            "expected": list(request.expected_artifacts),
            "produced": list(produced),
        },
    )
