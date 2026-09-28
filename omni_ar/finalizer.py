from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import yaml

from .project_records import ProjectRecords, sha256, write_json


_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".toml", ".j2"}
_FRAMEWORK_ROOTS = (
    "omni_ar",
    "controller_bash",
    "Heuresis_PJLAB-boyue",
)
_FRAMEWORK_FILES = (
    "run_research",
    "pyproject.toml",
    "requirements.txt",
    "tasks/adapter_contract.py",
    "tasks/leakage.py",
    "datasets/__init__.py",
    "datasets/adapter_contract.py",
    "datasets/registry.py",
    "datasets/generic_adapter.py",
)
_IGNORED_SOURCE_PARTS = {
    ".git", ".venv", "__pycache__", "artifacts", "logs", "outputs",
    "research_initializations", "acceptance_inputs", "fresh_inputs",
}
_IGNORED_SOURCE_FILES = {"baseline_result.json"}


def choose_winner(
    entries: list[dict[str, Any]], *, required_seeds: int = 1,
    baseline_value: float | None = None, min_improvement: float = 0.0,
) -> dict[str, Any] | None:
    eligible = [
        item for item in entries
        if item.get("bucket") == "accepted"
        and item.get("scientific_evidence") is True
        and isinstance(item.get("score"), (int, float))
        and (item.get("resource_usage") or {}).get("within_budget") is not False
        and (item.get("activation_validation") or {}).get("passed") is not False
        and not (item.get("protocol") or {}).get("evaluation_protocol_changed", False)
        and (
            (item.get("route") != "coding_agent" and not item.get("candidate_worktree"))
            or (item.get("control_validation") or {}).get("passed") is True
        )
        and int(item.get("seed_count", 1)) >= required_seeds
        and (
            baseline_value is None
            or (
                float(item["score"]) - baseline_value >= min_improvement
                if item.get("direction") == "maximize"
                else baseline_value - float(item["score"]) >= min_improvement
            )
        )
    ]
    if not eligible:
        return None
    directions = {item.get("direction") for item in eligible}
    if len(directions) != 1 or directions == {None}:
        raise ValueError("accepted candidates have inconsistent metric directions")
    maximize = directions.pop() == "maximize"

    def patch_size(item: dict[str, Any]) -> int:
        value = item.get("code_patch")
        path = Path(str(value)) if value else None
        return path.stat().st_size if path and path.is_file() else 0

    def rank(item: dict[str, Any]) -> tuple[float, float, float, int, str]:
        stability = item.get("stability") or {}
        mean = stability.get("mean")
        std = stability.get("std")
        score = float(item["score"])
        quality = score if maximize else -score
        stable_mean = float(mean) if isinstance(mean, (int, float)) else score
        stable_quality = stable_mean if maximize else -stable_mean
        stable_std = float(std) if isinstance(std, (int, float)) else float("inf")
        return (
            quality, stable_quality, -stable_std, -patch_size(item),
            str(item.get("candidate_id") or ""),
        )

    return max(eligible, key=rank)


def _copy_if_file(source: str | Path | None, destination: Path) -> dict[str, Any] | None:
    if not source:
        return None
    path = Path(source)
    if not path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {"path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size}


def _git_metadata(repository: Path, source_paths: list[Path]) -> dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=repository, text=True,
            capture_output=True, check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    scoped = [str(path) for path in source_paths]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain", "--", *scoped),
        "diff_sha256": None,
        "source_scope": scoped,
    }


def _relative_to_repository(repository: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository.resolve())
    except ValueError as exc:
        raise ValueError(f"source path escapes repository: {resolved}") from exc


def _dataset_pack_path(repository: Path, task_spec: Path) -> Path | None:
    spec = yaml.safe_load(task_spec.read_text(encoding="utf-8")) or {}
    dataset_ref = str((spec.get("data") or {}).get("dataset_ref") or "")
    if not dataset_ref:
        return None
    registry_path = repository / "datasets/registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if "@" not in dataset_ref or dataset_ref.endswith("@current"):
        dataset_id = dataset_ref.removesuffix("@current")
        dataset_ref = str((registry.get("channels") or {}).get(dataset_id, dataset_ref))
    entry = next(
        (item for item in registry.get("datasets", []) if item.get("ref") == dataset_ref),
        None,
    )
    if not entry:
        raise ValueError(f"task references an unregistered dataset: {dataset_ref}")
    pack = (repository / "datasets" / str(entry["path"])).resolve()
    datasets_root = (repository / "datasets").resolve()
    if pack != datasets_root and datasets_root not in pack.parents:
        raise ValueError(f"dataset pack escapes datasets root: {pack}")
    return pack


def _source_scope(repository: Path, task_spec: Path) -> tuple[list[Path], list[Path]]:
    """Return scoped Git paths and task-specific roots eligible for source archiving."""
    task_root = task_spec.resolve().parent
    dataset_pack = _dataset_pack_path(repository, task_spec)
    specific_roots = [task_root]
    if dataset_pack:
        specific_roots.append(dataset_pack)
    git_paths = [repository / item for item in (*_FRAMEWORK_ROOTS, *_FRAMEWORK_FILES)]
    git_paths.extend(specific_roots)
    git_paths.append(repository / "datasets/registry.yaml")
    relative_git_paths = sorted({_relative_to_repository(repository, path) for path in git_paths})
    relative_specific_roots = sorted({_relative_to_repository(repository, path) for path in specific_roots})
    return relative_git_paths, relative_specific_roots


def _within(relative: Path, root: Path) -> bool:
    return relative == root or root in relative.parents


def _eligible_untracked_source(relative: Path, specific_roots: list[Path]) -> bool:
    if any(part in _IGNORED_SOURCE_PARTS for part in relative.parts):
        return False
    if relative.name in _IGNORED_SOURCE_FILES or relative.suffix.lower() not in _SOURCE_SUFFIXES:
        return False
    for root in specific_roots:
        if _within(relative, root):
            tail = relative.relative_to(root)
            # Raw dataset payloads are bound by version and hash, never copied into a source archive.
            return not tail.parts or tail.parts[0] != "data"
    if relative.parts[0] in _FRAMEWORK_ROOTS:
        return "tests" not in relative.parts
    return str(relative) in _FRAMEWORK_FILES or str(relative) == "datasets/registry.yaml"


def _archive_untracked_sources(
    repository: Path,
    project_dir: Path,
    destination: Path,
    *,
    source_paths: list[Path],
    specific_roots: list[Path],
) -> dict[str, Any] | None:
    proc = subprocess.run(
        [
            "git", "ls-files", "--others", "--exclude-standard", "-z", "--",
            *(str(path) for path in source_paths),
        ],
        cwd=repository, capture_output=True, check=False,
    )
    if proc.returncode:
        return None
    project_relative = None
    try:
        project_relative = project_dir.resolve().relative_to(repository.resolve())
    except ValueError:
        pass
    selected: list[Path] = []
    for raw in proc.stdout.decode("utf-8", errors="surrogateescape").split("\0"):
        if not raw:
            continue
        relative = Path(raw)
        if project_relative and (relative == project_relative or project_relative in relative.parents):
            continue
        source = (repository / relative).resolve()
        if (
            _eligible_untracked_source(relative, specific_roots)
            and source.is_file()
            and source.stat().st_size <= 10_000_000
        ):
            selected.append(relative)
    if not selected:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for relative in sorted(selected):
            archive.add(repository / relative, arcname=str(relative), recursive=False)
    return {
        "path": str(destination), "sha256": sha256(destination),
        "bytes": destination.stat().st_size, "files": [str(path) for path in sorted(selected)],
    }


def _archive_current_records(records: ProjectRecords, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in records.root.rglob("*") if path.is_file())
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(records.root)), recursive=False)
    return {
        "path": str(destination), "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "files": [str(path.relative_to(records.root)) for path in files],
    }


def refresh_records_artifact(final_dir: Path, records: ProjectRecords) -> None:
    """Refresh the records bundle after the project and finalizer stages are finalized."""
    manifest_path = final_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = records.manifest()
    manifest.setdefault("artifacts", {})["records_archive"] = _archive_current_records(
        records, final_dir / "artifacts/records.tar.gz"
    )
    write_json(manifest_path, manifest)


def build_final_package(
    repository: Path,
    project_dir: Path,
    records: ProjectRecords,
    *,
    rough_idea: Path,
    task_spec: Path,
    idea_card: Path | None,
    termination: dict[str, Any],
) -> Path:
    final_dir = project_dir / "final_package"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    winner = choose_winner(
        records.archive(), required_seeds=int(termination.get("required_seeds", 1)),
        baseline_value=termination.get("baseline_value"),
        min_improvement=float(termination.get("min_improvement", 0.0)),
    )
    copied: dict[str, Any] = {}
    qa_transcript = rough_idea.parent / "qa_transcript.jsonl"
    question_generation = rough_idea.parent / "question_generation.json"
    for name, source in (
        ("rough_idea", rough_idea), ("task_spec", task_spec),
        ("qa_transcript", qa_transcript if qa_transcript.is_file() else None),
        ("question_generation", question_generation if question_generation.is_file() else None),
        ("idea_card", idea_card),
        ("proposal", winner.get("proposal_path") if winner else None),
        ("standardized_result", winner.get("standardized_result") if winner else None),
        ("execution_trace", winner.get("execution_trace") if winner else None),
        ("code_patch", winner.get("code_patch") if winner else None),
    ):
        suffix = Path(source).suffix if source else ".json"
        item = _copy_if_file(source, final_dir / "artifacts" / f"{name}{suffix}")
        if item:
            copied[name] = item

    source_paths, specific_roots = _source_scope(repository, task_spec)
    git = _git_metadata(repository, source_paths)
    diff_path = final_dir / "artifacts/source_changes.patch"
    diff = subprocess.run(
        [
            "git", "diff", "HEAD", "--binary", "--",
            *(str(path) for path in source_paths),
        ], cwd=repository,
        text=False, capture_output=True, check=False,
    ).stdout
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_bytes(diff)
    git["diff_sha256"] = sha256(diff_path)
    copied["source_changes"] = {
        "path": str(diff_path), "sha256": git["diff_sha256"],
        "bytes": diff_path.stat().st_size,
    }
    untracked = _archive_untracked_sources(
        repository, project_dir, final_dir / "artifacts/untracked_sources.tar.gz",
        source_paths=source_paths, specific_roots=specific_roots,
    )
    if untracked:
        copied["untracked_sources"] = untracked
    copied["records_archive"] = _archive_current_records(
        records, final_dir / "artifacts/records.tar.gz"
    )

    manifest = {
        "schema_version": "omni-ar-final-package/v1",
        "status": "ok" if winner else "no_accepted_candidate",
        "reproducibility_scope": (
            "best accepted candidate, current task and dataset interfaces, scoped framework state, "
            "current run records, and immutable input hashes"
        ),
        "winner": winner,
        "termination": termination,
        "qa_constraint_sources": (
            yaml.safe_load(rough_idea.read_text(encoding="utf-8")).get("provenance", {})
            .get("constraint_sources", {})
        ),
        "git": git,
        "records": records.manifest(),
        "artifacts": copied,
    }
    write_json(final_dir / "manifest.json", manifest)
    reproduce = final_dir / "run_final.sh"
    if winner and copied.get("proposal"):
        task_relative = str(task_spec.resolve().relative_to(repository.resolve()))
        candidate = str(winner.get("candidate_id"))
        use_candidate_patch = bool(winner.get("candidate_worktree") and copied.get("code_patch"))
        reproduce.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "ROOT=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\n"
            f"REPOSITORY={json.dumps(str(repository))}\n"
            f"BASE_COMMIT={json.dumps(str(git['commit']))}\n"
            f"TASK_REL={json.dumps(task_relative)}\n"
            f"CANDIDATE={json.dumps(candidate)}\n"
            "WORKTREE=${OMNI_AR_REPRO_WORKTREE:-/tmp/omni-ar-reproduce-$CANDIDATE}\n"
            "if [[ -e \"$WORKTREE\" ]]; then echo \"reproduction worktree exists: $WORKTREE\" >&2; exit 2; fi\n"
            "git -C \"$REPOSITORY\" worktree add --detach \"$WORKTREE\" \"$BASE_COMMIT\"\n"
            + "if [[ -s \"$ROOT/final_package/artifacts/source_changes.patch\" ]]; then "
            "git -C \"$WORKTREE\" apply \"$ROOT/final_package/artifacts/source_changes.patch\"; fi\n"
            + (
                "tar -xzf \"$ROOT/final_package/artifacts/untracked_sources.tar.gz\" -C \"$WORKTREE\"\n"
                if copied.get("untracked_sources") else ""
            )
            + (
                "git -C \"$WORKTREE\" apply \"$ROOT/final_package/artifacts/code_patch.patch\"\n"
                if use_candidate_patch else ""
            )
            + "\"$WORKTREE/run_research\" execute-proposal "
            "--task \"$WORKTREE/$TASK_REL\" --proposal \"$ROOT/final_package/artifacts/proposal.json\" "
            "--output-dir \"$WORKTREE/.omni_reproduction\" --execution-mode rjob\n",
            encoding="utf-8",
        )
        reproduce.chmod(0o755)
    else:
        reproduce.write_text(
            "#!/usr/bin/env bash\necho 'No accepted candidate is available for reproduction.' >&2\nexit 2\n",
            encoding="utf-8",
        )
        reproduce.chmod(0o755)
    readme = final_dir / "README.md"
    readme.write_text(
        "# AutoResearch Final Package\n\n"
        f"- Status: `{manifest['status']}`\n"
        f"- Termination: `{termination.get('reason')}` after {termination.get('rounds_completed')} round(s)\n"
        f"- Winner: `{winner.get('candidate_id') if winner else 'none'}`\n\n"
        "`manifest.json` contains input, result, code-state and record hashes. "
        "Run `run_final.sh` only when real rjob execution is intended.\n",
        encoding="utf-8",
    )
    return final_dir
