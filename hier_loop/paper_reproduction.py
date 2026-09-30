"""Bounded, CPU-only source attempts for the eight VCC25 paper cards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
STATIC_ENTRYPOINTS = {
    "lingshu-cell": "workflows/infer/main.py",
    "primeflow": "src/primeflow/modelcore/train.py",
    "state": "src/state/_cli/_tx/_train.py",
    "gears": "gears/gears.py",
    "presage": "src/train_presage.py",
    "scgenept": "train.py",
    "sclambda": "sclambda/model.py",
    "linear-baseline": "benchmark/src/run_linear_pretrained_model.R",
}
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
    code_root: Path | None = None, source_manifest: Path | None = None,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    output_dir = output_dir.resolve()
    root = (code_root or Path(__file__).resolve().parents[1]).resolve()
    if not _inside(output_dir, experiment_dir) or output_dir == experiment_dir:
        raise ValueError("output must be inside the independent experiment directory")
    if source_manifest is None and not _inside(root, experiment_dir):
        raise ValueError("LoopRSI code must run from the independent experiment directory")
    if any(_inside(experiment_dir, source) for source in SOURCE_CHECKOUTS):
        raise ValueError("experiment directory overlaps a source checkout")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty")
    sources: dict[str, dict[str, Any]] = {}
    if source_manifest is not None:
        manifest_path = source_manifest.resolve()
        if not _inside(manifest_path, experiment_dir):
            raise ValueError("source manifest must be inside the independent experiment directory")
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        sources = {row["method"]: row for row in rows}
        if set(sources) != set(METHODS) or len(rows) != len(METHODS):
            raise ValueError("source manifest must cover the eight formal methods exactly once")
    (output_dir / "cards").mkdir(parents=True)
    if source_manifest is None:
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
        attempts: list[dict[str, Any]] = []
        revision = None
        source_files: list[str] = []
        archive_sha256 = None
        if source_manifest is not None:
            source = sources[method]
            checkout = Path(source["source_dir"]).resolve()
            archive = Path(source["archive"]).resolve()
            if (source.get("status") != "fetched" or source.get("official_url") != url
                    or not _inside(checkout, experiment_dir) or not _inside(archive, experiment_dir)
                    or not checkout.is_dir() or not archive.is_file()):
                raise ValueError(f"{method}: invalid local source record")
            archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
            if archive_sha256 != source.get("archive_sha256"):
                raise ValueError(f"{method}: source archive hash mismatch")
            revision = source["revision"]
            entrypoint = checkout / STATIC_ENTRYPOINTS[method]
            if not entrypoint.is_file() or not _inside(entrypoint.resolve(), checkout):
                raise ValueError(f"{method}: documented entrypoint is missing or escapes source")
            if method == "linear-baseline":
                command = ["Rscript", "--vanilla", "-e", "parse(file=commandArgs(TRUE)[1])", str(entrypoint)]
            else:
                command = [sys.executable, "-c", "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), filename=sys.argv[1]); print('syntax-ok')", str(entrypoint)]
            attempts.append(_attempt(command, checkout, runner))
            source_files = sorted(path.name for path in checkout.iterdir())[:50]
        else:
            checkout = upstream_root / method
            if checkout.exists() and not _inside(checkout.resolve(), upstream_root.resolve()):
                raise ValueError(f"{method}: source checkout escapes experiment directory")
            attempts.append(_attempt(["git", "ls-remote", "--heads", url], experiment_dir, runner))
            if attempts[0].get("exit_code") == 0 and not checkout.exists():
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
            elif source_manifest is not None:
                prior_path = root / "knowledge" / "vcc25" / "evidence" / "stream-cuda-full-validation-20260930.json"
                if prior_path.is_file():
                    prior = json.loads(prior_path.read_text(encoding="utf-8"))
                    if (prior.get("job_status") == "Succeeded"
                            and prior.get("scope", {}).get("test_expression_read") is False):
                        status = "partial"
                        blocker = "prior remote validation succeeded; method-specific paper reproduction is incomplete"
                        prior_evidence = {"path": str(prior_path), "job_name": prior.get("job_name"),
                                          "scope": prior.get("scope"), "results": prior.get("results")}
        elif revision:
            blocker = "source obtained; method-specific H1 execution adapter and assets are not ready"
        if source_manifest is not None and attempts[0].get("exit_code") != 0:
            blocker = "local source entrypoint check failed; see the recorded command result"

        record = {
            "schema_version": "vcc25.paper-evidence/v1",
            "card_id": (
                f"kb:evidence:{method}:{revision[:12]}:local"
                if source_manifest is not None else f"kb:evidence:{method}:20260930"
            ),
            "authority": "local_looprsi_attempt" if source_manifest is not None else "remote_looprsi_attempt",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "paper_card_id": paper["id"],
            "code_card_id": repository["id"],
            "model_card_id": model["id"],
            "source_url": url,
            "card_source_revision": repository.get("revision"),
            "observed_source_revision": revision,
            "source_archive_sha256": archive_sha256,
            "model_artifacts": model.get("artifacts", []),
            "commands_attempted": attempts,
            "source_files": source_files,
            "data_scope": "source-only CPU attempt; no VCC25 expression data read in this run",
            "status": status,
            "blocker": blocker,
            "limitations": ["No method-specific training or inference was run in this attempt"],
            "validation_metrics": None,
            "prior_validation_evidence": prior_evidence,
            "next_condition": (
                ("Provide an isolated R runtime, required assets, and a reviewed H1 adapter"
                 if method == "linear-baseline" and attempts[0].get("exit_code") != 0
                 else "Provide required assets and a reviewed method-specific H1 adapter")
                if source_manifest is not None else
                "Provide reachable pinned source, required assets, and a reviewed method-specific H1 adapter"
            ),
            "test_expression_read": False,
            "gpu_used": False,
            "final_evaluation_run": False,
        }
        path = output_dir / "cards" / f"{method}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        statuses[method] = status

    summary = {
        "schema_version": "vcc25.paper-reproduction-summary/v1",
        "authority": "local_looprsi_attempt" if source_manifest is not None else "remote_looprsi_attempt",
        "decision_backend": "none; source execution phase before hierarchical selection",
        "methods": statuses,
        "evidence_cards": len(statuses),
        "method_reproduction_complete": False,
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
    parser.add_argument("--source-manifest", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_paper_reproduction(args.experiment_dir, args.output, source_manifest=args.source_manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
