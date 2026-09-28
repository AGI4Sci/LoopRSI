#!/usr/bin/env python3
"""Execute one validated live Heuresis proposal through the generic backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from run_research import execution_environment, load_env_defaults
from task_contract import (
    find_repository_root, load_json, load_task_spec, validate_proposal, write_json,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-mode", choices=("rjob", "rjob_dry_run", "local"), default="rjob")
    parser.add_argument(
        "--execution-config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/execution.env",
    )
    args = parser.parse_args()
    spec_path, proposal_path = args.task.resolve(), args.proposal.resolve()
    spec = load_task_spec(spec_path)
    proposal = load_json(proposal_path)
    validate_proposal(proposal, spec, spec_path)
    repository = find_repository_root(spec_path)
    output_dir = args.output_dir.resolve()
    if args.execution_mode.startswith("rjob") and output_dir != repository and repository not in output_dir.parents:
        raise SystemExit("rjob output-dir must stay inside the repository")
    round_dir, artifact_dir = output_dir / "state/round_0", output_dir / "artifacts"
    results_path = round_dir / "trial_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = load_env_defaults(args.execution_config.resolve())
    env = execution_environment(
        {**defaults, **os.environ}, repository, spec, spec_path,
        spec["task"]["name"], args.execution_mode,
    )
    command = [
        sys.executable, str(repository / "controller_bash/scripts/run_trials.py"),
        "--suggestion", str(proposal_path), "--round-dir", str(round_dir),
        "--artifact-dir", str(artifact_dir), "--out", str(results_path),
    ]
    proc = subprocess.run(command, cwd=repository, env=env, text=True, capture_output=True, check=False)
    (output_dir / "controller.stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (output_dir / "controller.stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")
    payload = load_json(results_path) if results_path.exists() else {"results": []}
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    dry_run = args.execution_mode == "rjob_dry_run"
    expected_statuses = {"skipped"} if dry_run else {"ok"}
    technically_complete = bool(rows) and all(
        row.get("status") in expected_statuses
        and row.get("result_contract_status") == "ok"
        and isinstance(row.get("archive_gate"), dict)
        and row["archive_gate"].get("decision") in {"accept", "reject"}
        for row in rows
    )
    summary = {
        "schema_version": "omni-ar-live-execution-trace/v1",
        "status": "ok" if proc.returncode == 0 and technically_complete else "failed",
        "task": spec["task"]["name"], "proposal_id": proposal["proposal_id"],
        "proposal": {"path": str(proposal_path), "sha256": sha256(proposal_path)},
        "execution_mode": args.execution_mode, "controller_returncode": proc.returncode,
        "scientific_evidence": not dry_run,
        "technically_complete": technically_complete,
        "num_trials": len(rows),
        "accepted": sum(row.get("archive_gate", {}).get("eligible") is True for row in rows),
        "rejected": sum(row.get("archive_gate", {}).get("eligible") is False for row in rows),
        "jobs": [{
            "job_name": row.get("job_name"), "status": row.get("status"),
            "result_contract_status": row.get("result_contract_status"),
            "archive_gate": row.get("archive_gate"), "output": row.get("output"),
            "standardized_result": row.get("standardized_result"), "rjob_log": row.get("rjob_log"),
        } for row in rows],
        "artifacts": {
            "trial_results": str(results_path),
            "stdout": str(output_dir / "controller.stdout.log"),
            "stderr": str(output_dir / "controller.stderr.log"),
        },
    }
    write_json(output_dir / "execution_trace.json", summary)
    print(output_dir / "execution_trace.json")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
