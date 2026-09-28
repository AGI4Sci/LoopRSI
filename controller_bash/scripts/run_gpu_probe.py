#!/usr/bin/env python3
"""Submit the stability CUDA probe through the shared rjob backend."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from execution_backends import ExecutionRequest, RjobConfig, RjobExecutionBackend
from run_research import load_env_defaults


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/execution.env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    if output_dir != repository and repository not in output_dir.parents:
        raise SystemExit("output-dir must stay inside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = load_env_defaults(args.execution_config.resolve())
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    shared_root = os.environ.get("OMNI_AR_SHARED_ROOT", "").strip()
    if not shared_root:
        raise SystemExit("OMNI_AR_SHARED_ROOT is required")
    os.environ.update({
        "RJOB_SYNC_SOURCE": str(repository),
        "RJOB_SYNC_PATHS": "controller_bash/scripts/gpu_probe.py",
        "RJOB_SHARED_FOLDER": str(Path(shared_root) / "rjob_packages/stability_gpu_probe"),
        "TRIAL_WORKDIR_IN_PACKAGE": ".", "TRIAL_PYTHONPATH_IN_PACKAGE": ".",
    })
    output = output_dir / "gpu_probe.raw.json"
    backend = RjobExecutionBackend(RjobConfig.from_env(output_dir))
    record = backend.execute(ExecutionRequest(
        job_name="omni-ar-stability-gpu-probe",
        command="python controller_bash/scripts/gpu_probe.py",
        worker_script=output_dir / "gpu_probe_worker.sh", output=output,
        stdout_path=output_dir / "gpu_probe.stdout.log",
        stderr_path=output_dir / "gpu_probe.stderr.log",
        trial_root=output_dir,
        resource_request={"gpu_count": 1, "cpu": 4, "memory_mb": 8000, "max_runtime_minutes": 10},
        trial={"name": "gpu_probe", "profile": "stability/profiles.yaml#gpu_probe"},
    ), dry_run=args.dry_run)
    (output_dir / "gpu_probe.execution.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_dir / "gpu_probe.execution.json")
    return 0 if record.get("status") in ({"skipped"} if args.dry_run else {"ok"}) else 1


if __name__ == "__main__":
    raise SystemExit(main())
