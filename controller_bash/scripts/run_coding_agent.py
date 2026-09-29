#!/usr/bin/env python3
"""Apply one implementation request through the sandboxed Heuresis Harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from task_contract import (
    ContractError, editable_workspace_path, find_repository_root, load_task_spec,
    path_is_editable, validate_proposal, write_json,
)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-") or "candidate"


def file_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if (
            path.is_file()
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
            and not path.name.endswith(".pyc")
        ):
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def create_worktree(repository: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "worktree", "add", "--detach", str(target), "HEAD"],
        cwd=repository, text=True, capture_output=True, check=False,
    )
    if proc.returncode:
        raise ContractError(f"git worktree add failed: {proc.stderr.strip()}")


def ensure_bubblewrap(agent: str) -> str | None:
    """Expose Codex's bundled Bubblewrap without weakening Harness isolation."""
    existing = shutil.which("bwrap")
    if existing:
        return existing
    configured = os.environ.get("OMNI_AR_BWRAP")
    candidates = [Path(configured).expanduser()] if configured else []
    agent_binary = shutil.which(agent)
    if agent_binary:
        resolved = Path(agent_binary).resolve()
        for parent in resolved.parents:
            candidates.extend(parent.glob("node_modules/@openai/codex-*/vendor/*/codex-resources/bwrap"))
            candidates.extend(parent.glob("vendor/*/codex-resources/bwrap"))
    selected = next((path.resolve() for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)
    if selected is None:
        return None
    os.environ["PATH"] = str(selected.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(selected)


def _safe_snapshot_path(relative: Path) -> bool:
    blocked_parts = {
        ".git", ".venv", "__pycache__", "release_verification",
        "research_initializations", "Agent报告", "artifacts", "outputs",
    }
    lowered = str(relative).lower()
    if any(part in blocked_parts for part in relative.parts):
        return False
    if any(token in lowered for token in (".env", "credential", "secret", "api_key", "token")):
        return False
    return relative.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".toml", ".j2"}


def materialize_source_snapshot(repository: Path, worktree: Path, output_dir: Path) -> dict[str, Any]:
    """Apply reviewed local source state to a candidate worktree without committing it."""
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--no-ext-diff"], cwd=repository,
        capture_output=True, check=False,
    )
    if diff.returncode:
        raise ContractError("cannot capture tracked source changes")
    patch_path = output_dir / "source_snapshot.patch"
    patch_path.write_bytes(diff.stdout)
    if diff.stdout:
        applied = subprocess.run(
            ["git", "apply", "--binary", "-"], cwd=worktree,
            input=diff.stdout, capture_output=True, check=False,
        )
        if applied.returncode:
            raise ContractError(f"cannot apply tracked source snapshot: {applied.stderr.decode(errors='replace')}")

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repository,
        capture_output=True, check=False,
    )
    if untracked.returncode:
        raise ContractError("cannot list untracked source files")
    copied: list[str] = []
    for raw in untracked.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not raw:
            continue
        relative = Path(raw)
        source = repository / relative
        if not _safe_snapshot_path(relative) or not source.is_file() or source.stat().st_size > 10_000_000:
            continue
        destination = worktree / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(relative))
    staged = subprocess.run(
        ["git", "add", "-A"], cwd=worktree, text=True, capture_output=True, check=False,
    )
    if staged.returncode:
        raise ContractError(f"cannot stage source snapshot: {staged.stderr.strip()}")
    manifest = {
        "schema_version": "omni-ar-source-snapshot/v1",
        "base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "tracked_patch": str(patch_path),
        "tracked_patch_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "untracked_sources": copied,
    }
    write_json(output_dir / "source_snapshot.json", manifest)
    return manifest


def materialize_dataset_binding(
    repository: Path, worktree: Path, spec: dict[str, Any], output_dir: Path,
) -> dict[str, Any] | None:
    """Hard-link ignored Dataset Pack payloads into an isolated worktree.

    Generated candidates need the same immutable data as their parent run.
    Hard links keep one physical payload and remain regular files when rsync
    packages the candidate for rjob. Cross-filesystem copies are rejected so a
    multi-gigabyte dataset is never duplicated silently.
    """
    dataset_ref = str((spec.get("data") or {}).get("dataset_ref") or "")
    if not dataset_ref:
        return None
    from datasets.registry import DatasetRegistry

    registry = DatasetRegistry(repository / "datasets")
    manifest = registry.manifest(dataset_ref)
    source_pack = registry.pack_root(dataset_ref).resolve()
    target_pack = worktree / source_pack.relative_to(repository)
    linked: list[dict[str, Any]] = []
    for artifact_name, artifact in (manifest.get("artifacts") or {}).items():
        relative = Path(str(artifact["path"]))
        source = (source_pack / relative).resolve()
        if source != source_pack and source_pack not in source.parents:
            raise ContractError(f"dataset artifact escapes its pack: {artifact_name}")
        if not source.exists():
            raise ContractError(f"dataset artifact is missing: {source}")
        files = [source] if source.is_file() else [path for path in source.rglob("*") if path.is_file()]
        for item in files:
            suffix = item.relative_to(source_pack)
            destination = target_pack / suffix
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(item, destination)
            except OSError as exc:
                raise ContractError(
                    "dataset payload and Coding Agent worktree must share a filesystem; "
                    "set OMNI_AR_WORKTREE_ROOT beside the repository"
                ) from exc
            linked.append({"path": str(suffix), "size_bytes": item.stat().st_size})
    record = {
        "schema_version": "omni-ar-candidate-dataset-binding/v1",
        "dataset_ref": dataset_ref, "source_pack": str(source_pack),
        "candidate_pack": str(target_pack), "mode": "hardlink",
        "linked_files": linked, "total_bytes": sum(item["size_bytes"] for item in linked),
    }
    write_json(output_dir / "dataset_binding.json", record)
    return record


def prompt_for(request: dict[str, Any]) -> str:
    return f"""Implement this reviewed AutoResearch request in `/workspace/solution`.

Rules:
- Edit only paths listed in allowed_paths, relative to `/workspace/solution`.
- `task_spec.yaml` is read-only context.
- Never edit datasets, protocols, metrics, controller code, environment files, or credentials.
- Preserve entrypoints and result schemas.
- Register new trial parameters in the task adapter only when that adapter is explicitly allowed.
- Emit each activation diagnostic with a nonzero or expected-state value when active.
- Run static checks only. Do not train, submit jobs, or use network services.

Request:
{json.dumps(request, ensure_ascii=False, indent=2)}
"""


@dataclass(frozen=True)
class DirectAgentRun:
    exit_code: int


def run_direct_codex(
    agent: str,
    model: str | None,
    source: Path,
    agent_dir: Path,
    prompt: str,
    *,
    timeout: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> DirectAgentRun:
    """Run Codex in an isolated editable copy when the Heuresis package is unavailable."""

    edited = agent_dir / "solution"
    agent_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source,
        edited,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    command = [
        agent,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--cd",
        str(edited),
    ]
    base_url = os.environ.get("OMNI_AR_CODEX_BASE_URL")
    if base_url:
        command.extend(["--config", f"openai_base_url={json.dumps(base_url)}"])
    if model:
        command.extend(["--model", model])
    command.append("-")
    secret_tokens = ("API_KEY", "ACCESS_TOKEN", "SECRET_KEY", "PASSWORD")
    environment = {
        name: value
        for name, value in os.environ.items()
        if not any(token in name.upper() for token in secret_tokens)
    }
    completed = runner(
        command,
        cwd=edited,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=environment,
    )
    (agent_dir / "agent.log").write_text(
        str(completed.stdout or "") + "\n--- stderr ---\n" + str(completed.stderr or ""),
        encoding="utf-8",
    )
    return DirectAgentRun(exit_code=completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("plan", "apply"), default="plan")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--backend", choices=("harness", "direct-codex"), default="harness")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "coding_route.json"
    try:
        spec_path = args.task.resolve()
        spec = load_task_spec(spec_path)
        repository = find_repository_root(spec_path)
        request = json.loads(args.request.read_text(encoding="utf-8"))
        request = request.get("implementation_request", request)
        required = {
            "request_id", "hypothesis", "change_scope", "allowed_paths",
            "required_capabilities", "activation_diagnostics", "trial_proposal",
        }
        if not isinstance(request, dict) or set(request) != required:
            raise ContractError("implementation request fields do not match the v1 contract")
        invalid = [path for path in request["allowed_paths"] if not path_is_editable(path, spec, spec_path)]
        if invalid:
            raise ContractError(f"implementation request escapes editable paths: {invalid}")
        workspace_request = dict(request)
        workspace_request["allowed_paths"] = [
            editable_workspace_path(path, spec, spec_path)
            for path in request["allowed_paths"]
        ]
        write_json(output_dir / "implementation_request.json", request)
        write_json(output_dir / "workspace_request.json", workspace_request)
        result: dict[str, Any] = {
            "schema_version": "omni-ar-coding-route/v1",
            "status": "planned", "mode": args.mode,
            "request_id": request["request_id"], "route": "coding_agent",
            "allowed_paths": request["allowed_paths"],
            "workspace_allowed_paths": workspace_request["allowed_paths"],
            "activation_diagnostics": request["activation_diagnostics"],
            "candidate_worktree": None, "code_patch": None,
            "checks": {"path_policy": "ok", "static": "not_run", "post_compile": "not_run"},
        }
        if args.mode == "plan":
            write_json(result_path, result)
            print(result_path)
            return 0

        digest = hashlib.sha256(str(output_dir).encode()).hexdigest()[:12]
        default_worktree_root = repository.parent / ".omni-ar-worktrees"
        worktree_root = Path(os.environ.get("OMNI_AR_WORKTREE_ROOT", str(default_worktree_root)))
        worktree = worktree_root / f"{slug(request['request_id'])}-{digest}"
        resume_index = 0
        while worktree.exists():
            resume_index += 1
            worktree = worktree_root / f"{slug(request['request_id'])}-{digest}-resume-{resume_index:02d}"
        create_worktree(repository, worktree)
        source_snapshot = materialize_source_snapshot(repository, worktree, output_dir)
        candidate_task = worktree / spec_path.relative_to(repository)
        candidate_spec = load_task_spec(candidate_task)
        dataset_binding = materialize_dataset_binding(
            repository, worktree, candidate_spec, output_dir,
        )
        candidate_solution = worktree
        agent_dir = output_dir / (
            "agent_workspace" if resume_index == 0 else f"agent_workspace_resume_{resume_index:02d}"
        )
        timeout = int(os.environ.get("OMNI_AR_CODING_TIMEOUT_SEC", "1800"))
        if args.backend == "direct-codex":
            bubblewrap = None
            run = run_direct_codex(
                args.agent,
                args.model,
                candidate_solution,
                agent_dir,
                prompt_for(workspace_request).replace(
                    "`/workspace/solution`", "the current working directory"
                ),
                timeout=timeout,
            )
        else:
            heuresis_src = repository / "Heuresis_PJLAB-boyue/src"
            if str(heuresis_src) not in sys.path:
                sys.path.insert(0, str(heuresis_src))
            from heuresis.harness import Harness
            from heuresis.workspace import Workspace

            workspace = Workspace(
                files={"solution": candidate_solution, "task_spec.yaml": candidate_task},
                prompt="", venv=Path(sys.prefix), editable="solution", lock_down_edits=True,
            )
            bubblewrap = ensure_bubblewrap(args.agent)
            # Codex authenticates through its mounted profile. Research and data
            # service credentials are unnecessary for code generation and must not
            # be visible inside the candidate sandbox or its persistent agent.log.
            secret_env = [
                name for name in os.environ
                if any(token in name.upper() for token in (
                    "API_KEY", "ACCESS_TOKEN", "SECRET_KEY", "PASSWORD",
                ))
            ]
            harness = Harness(
                args.agent, model=args.model, gpus=[], max_workers=1,
                strip_env=secret_env,
            )
            errors = harness.preflight()
            if errors:
                raise ContractError("coding agent preflight failed: " + "; ".join(errors))
            run = harness.run(
                workspace, prompt_for(workspace_request), path=agent_dir,
                timeout=timeout,
            ).result(timeout=timeout + 30)
        edited = agent_dir / "solution"
        before, after = file_hashes(candidate_solution), file_hashes(edited)
        changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        deleted = sorted(path for path in before if path not in after)
        allowed = [str(Path(path)) for path in workspace_request["allowed_paths"]]
        outside = [
            path for path in changed
            if not any(path == root or path.startswith(root.rstrip("/") + "/") for root in allowed)
        ]
        python_errors = []
        for relative in changed:
            path = edited / relative
            if path.suffix == ".py" and path.is_file() and relative not in outside:
                try:
                    py_compile.compile(str(path), doraise=True)
                except py_compile.PyCompileError as exc:
                    python_errors.append(str(exc))
        changed_text = "\n".join(
            (edited / relative).read_text(encoding="utf-8", errors="replace")
            for relative in changed if (edited / relative).is_file()
        )
        missing_diagnostics = [name for name in request["activation_diagnostics"] if name not in changed_text]
        safe_to_copy = run.exit_code == 0 and bool(changed) and not outside and not deleted and not python_errors and not missing_diagnostics
        if safe_to_copy:
            for relative in changed:
                source, destination = edited / relative, candidate_solution / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            subprocess.run(
                ["git", "add", "-N", "--", *changed], cwd=worktree,
                text=True, capture_output=True, check=False,
            )
        patch_path = output_dir / "candidate.patch"
        patch_proc = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff"], cwd=worktree,
            capture_output=True, check=False,
        )
        patch_path.write_bytes(patch_proc.stdout)
        wrapper = {
            "schema_version": "omni-ar-proposal/v2",
            "proposal_id": f"coding-{slug(request['request_id'])}",
            "task_name": spec["task"]["name"], "round": 0,
            "verdict": "Post-code candidate trial.", "evidence_gaps": [],
            "experiment_proposals": [request["trial_proposal"]], "risks": [],
        }
        post_compile_error = None
        if safe_to_copy:
            try:
                candidate_spec = load_task_spec(candidate_task)
                validate_proposal(wrapper, candidate_spec, candidate_task)
            except Exception as exc:  # noqa: BLE001
                post_compile_error = str(exc)
        else:
            post_compile_error = "candidate failed pre-copy safety checks"
        valid = safe_to_copy and post_compile_error is None
        result.update({
            "status": "ready" if valid else "invalid",
            "candidate_worktree": str(worktree),
            "candidate_task_spec": str(candidate_task),
            "code_patch": str(patch_path), "changed_paths": changed,
            "agent_returncode": run.exit_code,
            "source_snapshot": source_snapshot,
            "dataset_binding": dataset_binding,
            "bubblewrap": bubblewrap,
            "checks": {
                "path_policy": "ok" if not outside else "failed", "outside_paths": outside,
                "deletions_forbidden": not deleted, "deleted_paths": deleted,
                "static": "ok" if not python_errors else "failed", "static_errors": python_errors,
                "activation_diagnostics_declared": not missing_diagnostics,
                "missing_activation_diagnostics": missing_diagnostics,
                "post_compile": "ok" if post_compile_error is None else "failed",
                "post_compile_error": post_compile_error,
            },
        })
        write_json(result_path, result)
        print(result_path)
        return 0 if valid else 2
    except Exception as exc:  # noqa: BLE001
        write_json(result_path, {
            "schema_version": "omni-ar-coding-route/v1",
            "status": "failed", "mode": args.mode, "error": str(exc),
        })
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
