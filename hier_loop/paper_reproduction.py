"""Bounded, CPU-only source attempts for the eight VCC25 paper cards."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


METHODS = (
    "lingshu-cell", "primeflow", "state", "gears", "presage",
    "scgenept", "sclambda", "linear-baseline",
)
SOURCE_CHECKOUTS = (
    Path("/mnt/shared-storage-user/gaozhangyang/RSI"),
    Path("/mnt/shared-storage-gpfs2/beam-gpfs02/huangwenxuan/lingshu-cell-agent"),
)
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _read_card(root: Path, kind: str, method: str) -> dict[str, Any]:
    path = root / "knowledge" / "vcc25" / kind / f"{method}.json"
    if kind == "models":
        repository = json.loads((root / "knowledge" / "vcc25" / "repositories" / f"{method}.json").read_text())
        ids = [value for value in repository["relations"] if value.startswith("kb:model:")]
        if len(ids) != 1:
            raise ValueError(f"{method}: expected one ModelCard relation")
        path = root / "knowledge" / "vcc25" / kind / f"{ids[0].split(':')[-1]}.json"
    card = json.loads(path.read_text(encoding="utf-8"))
    expected_type = {"papers": "paper", "repositories": "repository", "models": "model"}[kind]
    if card.get("asset_type") != expected_type:
        raise ValueError(f"{method}: invalid {kind} card")
    return card


def _attempt(command: list[str], cwd: Path, runner: RunCommand) -> dict[str, Any]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    record: dict[str, Any] = {"command": command, "cwd": str(cwd), "timeout_seconds": 15}
    try:
        result = runner(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=15, check=False)
        record.update(exit_code=result.returncode, stdout=(result.stdout or "")[-2000:], stderr=(result.stderr or "")[-2000:])
    except subprocess.TimeoutExpired as exc:
        record.update(exit_code=None, error=f"timed out after 15 seconds: {exc.cmd}")
    except OSError as exc:
        record.update(exit_code=None, error=f"{type(exc).__name__}: {exc}")
    return record


def run_paper_reproduction(
    experiment_dir: Path, output_dir: Path, *, runner: RunCommand = subprocess.run,
    code_root: Path | None = None,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    output_dir = output_dir.resolve()
    root = (code_root or Path(__file__).resolve().parents[1]).resolve()
    if not _inside(output_dir, experiment_dir) or output_dir == experiment_dir:
        raise ValueError("output must be inside the independent experiment directory")
    if not _inside(root, experiment_dir):
        raise ValueError("LoopRSI code must run from the independent experiment directory")
    if any(_inside(experiment_dir, source) for source in SOURCE_CHECKOUTS):
        raise ValueError("experiment directory overlaps a source checkout")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    (output_dir / "cards").mkdir(parents=True)
    upstream_root = experiment_dir / "upstream"
    upstream_root.mkdir(exist_ok=True)
    if not _inside(upstream_root.resolve(), experiment_dir):
        raise ValueError("upstream directory must remain inside the experiment")

    statuses: dict[str, str] = {}
    for method in METHODS:
        paper = _read_card(root, "papers", method)
        repository = _read_card(root, "repositories", method)
        model = _read_card(root, "models", method)
        url = repository["official_url"]
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "github.com":
            raise ValueError(f"{method}: unsupported source URL")
        checkout = upstream_root / method
        if checkout.exists() and not _inside(checkout.resolve(), upstream_root.resolve()):
            raise ValueError(f"{method}: source checkout escapes experiment directory")
        attempts = [_attempt(["git", "ls-remote", "--heads", url], experiment_dir, runner)]
        revision = None
        source_files: list[str] = []
        if attempts[0].get("exit_code") == 0:
            if not checkout.exists():
                attempts.append(_attempt(
                    ["git", "clone", "--depth", "1", "--filter=blob:none", url, str(checkout)],
                    experiment_dir, runner,
                ))
        if checkout.is_dir() and (checkout / ".git").exists():
            revision_result = _attempt(["git", "rev-parse", "HEAD"], checkout, runner)
            attempts.append(revision_result)
            if revision_result.get("exit_code") == 0:
                revision = revision_result["stdout"].strip()
            source_files = sorted(path.name for path in checkout.iterdir() if path.name != ".git")[:50]

        prior_evidence = None
        status = "blocked"
        blocker = "official source unavailable; see the bounded Git command result"
        if method == "lingshu-cell":
            prior_path = experiment_dir / "artifacts" / "phase-d-validation" / "stream-cuda-full-20260930-r2" / "lineage_summary.json"
            if prior_path.is_file():
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                contract = prior.get("data_contract") or {}
                if (prior.get("status") == "pass" and prior.get("results")
                        and contract.get("test_expression_read") is False
                        and prior.get("official_score_claim") is False):
                    status = "partial"
                    blocker = "prior validation succeeded; method-specific paper reproduction is incomplete"
                    prior_evidence = {
                        "path": str(prior_path), "status": prior["status"],
                        "data_contract": contract,
                        "aggregate_metrics": {
                            key: value.get("aggregate")
                            for key, value in prior["results"].items()
                        },
                    }
        elif revision:
            blocker = "source checkout obtained; method-specific H1 execution adapter and assets are not ready"

        record = {
            "schema_version": "vcc25.paper-evidence/v1",
            "card_id": f"kb:evidence:{method}:20260930",
            "authority": "remote_looprsi_attempt",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "paper_card_id": paper["id"],
            "code_card_id": repository["id"],
            "model_card_id": model["id"],
            "source_url": url,
            "card_source_revision": repository.get("revision"),
            "observed_source_revision": revision,
            "model_artifacts": model.get("artifacts", []),
            "commands_attempted": attempts,
            "source_files": source_files,
            "data_scope": "source-only CPU attempt; no VCC25 expression data read in this run",
            "status": status,
            "blocker": blocker,
            "limitations": ["No method-specific training or inference was run in this attempt"],
            "validation_metrics": None,
            "prior_validation_evidence": prior_evidence,
            "next_condition": "Provide reachable pinned source, required assets, and a reviewed method-specific H1 adapter",
            "test_expression_read": False,
            "gpu_used": False,
            "final_evaluation_run": False,
        }
        path = output_dir / "cards" / f"{method}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        statuses[method] = status

    summary = {
        "schema_version": "vcc25.paper-reproduction-summary/v1",
        "authority": "remote_looprsi_attempt",
        "decision_backend": "none; source execution phase before hierarchical selection",
        "methods": statuses,
        "evidence_cards": len(statuses),
        "candidate_request_generated": False,
        "gpu_used": False,
        "test_expression_read": False,
        "final_evaluation_run": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded VCC25 paper source attempts")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_paper_reproduction(args.experiment_dir, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
