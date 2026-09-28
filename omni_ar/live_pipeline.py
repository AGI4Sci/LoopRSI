from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from datasets.registry import DatasetRegistry
from tasks.adapter_contract import merge_research_capabilities
from .feature_planner import plan_features
from .initialization import InitializationError, RoughIdeaEngine


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InitializationError(f"expected mapping in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _progress(
    output_dir: Path, stage: str, status: str, *, message: str, **details: Any,
) -> None:
    """Persist a planning heartbeat and show it immediately in the one-command CLI."""
    path = output_dir / "planning_progress.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        payload = {}
    value = {
        **((payload.get("stages") or {}).get(stage) or {}),
        **details, "status": status, "message": message,
        "updated_at_epoch": time.time(),
    }
    if status == "running":
        value.setdefault("started_at_epoch", time.time())
    if status in {"completed", "failed", "cached"}:
        value["finished_at_epoch"] = time.time()
    payload.update({"schema_version": "omni-ar-planning-progress/v1"})
    payload.setdefault("stages", {})[stage] = value
    _write_json(path, payload)
    print(f"[planning] {stage}: {message}", flush=True)


def _descendant_pids(root_pid: int) -> list[int]:
    """Snapshot descendants before a shell can orphan or re-session them."""
    found: list[int] = []
    pending = [root_pid]
    seen = {root_pid}
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in children_path.read_text().split()]
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            children = []
        for child in children:
            if child not in seen:
                seen.add(child)
                found.append(child)
                pending.append(child)
    return found


def _scoped_process_pids(scope: Path) -> list[int]:
    """Find same-user processes whose command line belongs to one run scope."""
    marker = str(scope.resolve()).encode()
    found: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == os.getpid():
            continue
        try:
            if proc_dir.stat().st_uid != os.getuid():
                continue
            command = (proc_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if marker in command:
            found.append(pid)
    return found


def _terminate_process_tree(
    process: subprocess.Popen[Any], *, scope: Path | None = None,
) -> None:
    """Terminate the stage, including descendants that detached from it."""
    descendants = _descendant_pids(process.pid)
    scoped = _scoped_process_pids(scope) if scope is not None else []
    targets = list(reversed(list(dict.fromkeys([*descendants, *scoped]))))
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    for pid in targets:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_external_stage(
    command: list[str], *, cwd: Path, env: dict[str, str], output_dir: Path,
    stage: str, timeout_s: float, stdout_name: str, stderr_name: str,
    checkpoints: list[tuple[str, Path]] | None = None,
) -> None:
    """Run an external planner with heartbeat output, logs, and a hard deadline."""
    if timeout_s <= 0:
        raise InitializationError(f"{stage} timeout must be positive")
    interval = max(5.0, float(os.environ.get("OMNI_AR_PLANNING_HEARTBEAT_SEC", "30")))
    started = time.monotonic()
    announced: set[str] = set()
    _progress(output_dir, stage, "running", message=f"started; timeout={timeout_s:g}s")
    with (
        (output_dir / stdout_name).open("w", encoding="utf-8") as stdout,
        (output_dir / stderr_name).open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        next_heartbeat = interval
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                for label, path in checkpoints or []:
                    if label not in announced and path.exists():
                        announced.add(label)
                        _progress(
                            output_dir, stage, "running",
                            message=f"checkpoint {label} completed; elapsed={elapsed:.0f}s",
                            elapsed_seconds=round(elapsed, 3), checkpoints=sorted(announced),
                        )
                if elapsed >= next_heartbeat:
                    _progress(
                        output_dir, stage, "running",
                        message=f"still running; elapsed={elapsed:.0f}s",
                        elapsed_seconds=round(elapsed, 3), checkpoints=sorted(announced),
                    )
                    next_heartbeat += interval
                if elapsed >= timeout_s:
                    _terminate_process_tree(process, scope=output_dir)
                    _progress(
                        output_dir, stage, "failed",
                        message=f"timed out after {timeout_s:g}s; completed checkpoints can be resumed",
                        elapsed_seconds=round(elapsed, 3), checkpoints=sorted(announced),
                    )
                    raise InitializationError(
                        f"{stage} exceeded {timeout_s:g}s; completed checkpoints are reusable; "
                        f"see {output_dir / stderr_name}"
                    )
                time.sleep(min(1.0, max(0.05, timeout_s - elapsed)))
        except BaseException:
            if process.poll() is None:
                _terminate_process_tree(process, scope=output_dir)
            raise
        returncode = process.returncode
    elapsed = time.monotonic() - started
    if returncode:
        _progress(
            output_dir, stage, "failed",
            message=f"failed with exit code {returncode}; elapsed={elapsed:.0f}s",
            elapsed_seconds=round(elapsed, 3), checkpoints=sorted(announced),
        )
        raise InitializationError(
            f"{stage} failed with exit code {returncode}; see {output_dir / stderr_name}"
        )
    _progress(
        output_dir, stage, "completed", message=f"completed; elapsed={elapsed:.0f}s",
        elapsed_seconds=round(elapsed, 3), checkpoints=sorted(announced),
    )


def _cache_artifacts(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha(path) for path in paths if path.is_file()}


def _cache_valid(cache: dict[str, Any], stage: str, paths: list[Path], **expected: Any) -> bool:
    record = (cache.get("stages") or {}).get(stage) or {}
    if record.get("status") != "completed":
        return False
    if any(record.get(key) != value for key, value in expected.items()):
        return False
    artifacts = record.get("artifacts") or {}
    return bool(paths) and all(
        path.is_file() and artifacts.get(str(path)) == _sha(path) for path in paths
    )


def _controller_api(repository: Path):
    scripts = repository / "controller_bash/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from task_contract import load_task_adapter, load_task_spec, validate_proposal

    return load_task_adapter, load_task_spec, validate_proposal


def run_live_pipeline(repository: Path, rough_path: Path, output_dir: Path) -> Path:
    repository, rough_path, output_dir = repository.resolve(), rough_path.resolve(), output_dir.resolve()
    rough = _load(rough_path)
    engine = RoughIdeaEngine(repository)
    engine.validate(rough)
    if rough["confirmation"]["status"] != "confirmed" or rough["constraints"]["conflicts"]:
        raise InitializationError("live pipeline requires a confirmed, conflict-free rough idea")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_spec_value = str((rough.get("task") or {}).get("task_spec") or "")
    task_spec_input = repository / task_spec_value if task_spec_value else None
    cache_key = hashlib.sha256(json.dumps({
        "rough_sha256": _sha(rough_path),
        "task_spec_sha256": (
            _sha(task_spec_input) if task_spec_input and task_spec_input.is_file() else None
        ),
        "model": os.environ.get("OMNI_AR_BOYUE_MODEL") or None,
        "pipeline_version": 2,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = output_dir / "planning_cache.json"
    try:
        previous_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_cache = {}
    same_input = previous_cache.get("cache_key") == cache_key
    cache: dict[str, Any] = previous_cache if same_input else {
        "schema_version": "omni-ar-planning-cache/v1", "cache_key": cache_key,
        "stages": {},
    }
    _write_json(cache_path, cache)
    env_file = repository / "Heuresis_PJLAB-boyue/.env"
    researchstudio_dir = output_dir / "researchstudio"
    entrypoint = repository / "ResearchStudio-main/scripts/rough_idea_entry.py"
    rs_command = [
        sys.executable, str(entrypoint), "--rough-idea", str(rough_path),
        "--output-dir", str(researchstudio_dir), "--env-file", str(env_file),
        "--run-name", "ideaspark_live",
    ]
    phase0 = researchstudio_dir / "ideaspark_live/phase0"
    idea_card = researchstudio_dir / "idea.detail.en.md"
    researchstudio_artifacts = [
        researchstudio_dir / "researchstudio_result.json",
        phase0 / "lit_results.json", phase0 / "lit_table.md",
        phase0 / ".lit_grounding_mode", idea_card,
        researchstudio_dir / "method_view.json",
    ]
    if same_input and (phase0 / "lit_results.json").exists():
        rs_command.append("--resume")
    external_env = dict(os.environ)
    external_env.setdefault("OMNI_AR_BOYUE_DISABLE_PROXY", "1")
    external_env.setdefault("IDEASPARK_BOYUE_TIMEOUT_SEC", "180")
    # One retry absorbs transient 429/5xx/network failures while the outer
    # stage timeout still provides a hard upper bound.  Invalid planning
    # output remains fail-closed in the ResearchStudio validators.
    external_env.setdefault("IDEASPARK_BOYUE_MAX_RETRIES", "2")
    rs_cached = same_input and _cache_valid(
        cache, "researchstudio", researchstudio_artifacts,
    )
    if rs_cached:
        _progress(
            output_dir, "researchstudio", "cached",
            message="reused literature retrieval and Idea Card from matching input",
        )
    else:
        _run_external_stage(
            rs_command, cwd=repository, env=external_env, output_dir=output_dir,
            stage="researchstudio", timeout_s=float(os.environ.get(
                "OMNI_AR_RESEARCHSTUDIO_TIMEOUT_SEC", "600"
            )), stdout_name="researchstudio.entry.stdout.log",
            stderr_name="researchstudio.entry.stderr.log",
            checkpoints=[
                ("literature_retrieval", phase0 / "lit_results.json"),
                ("literature_table", phase0 / "lit_table.md"),
                ("idea_card", idea_card),
            ],
        )
        missing_rs = [str(path) for path in researchstudio_artifacts if not path.is_file()]
        if missing_rs:
            raise InitializationError(
                f"ResearchStudio completed without required artifacts: {missing_rs}"
            )
        cache.setdefault("stages", {})["researchstudio"] = {
            "status": "completed", "artifacts": _cache_artifacts(researchstudio_artifacts),
        }
        _write_json(cache_path, cache)
    if not rough["task"].get("task_spec"):
        trace = {
            "schema_version": "omni-ar-live-trace/v1",
            "status": "ok",
            "outcome": "researchstudio_complete_adapter_required",
            "initialization_id": rough["initialization_id"],
            "execution_authorized": False,
            "stages": [
                {"name": "initialization", "inputs": [], "outputs": ["rough_idea"]},
                {"name": "researchstudio_literature_and_idea", "inputs": ["rough_idea"], "outputs": ["idea_card", "method_view"]},
            ],
            "checks": {
                "researchstudio_complete": idea_card.exists(),
                "task_adapter_bound": False,
                "heuresis_started": False,
                "training_started": False,
                "rjob_submitted": False,
            },
            "blockers": list((rough.get("execution_readiness") or {}).get("blockers") or []),
            "artifacts": {
                "rough_idea": {"path": str(rough_path), "sha256": _sha(rough_path)},
                "idea_card": {"path": str(idea_card), "sha256": _sha(idea_card)},
            },
        }
        trace_path = output_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return trace_path
    task_spec = (repository / rough["task"]["task_spec"]).resolve()
    load_adapter, load_spec, validate_proposal = _controller_api(repository)
    spec = load_spec(task_spec)
    adapter = load_adapter(spec, task_spec)
    task_capabilities = adapter.initialization_capabilities(spec)
    dataset_adapter = DatasetRegistry(repository / "datasets").load_adapter(
        rough["dataset"]["ref"]
    )
    feature_plan_path = output_dir / "feature_plan.json"
    feature_plan = plan_features(
        repository, rough["dataset"]["ref"],
        rough_idea=str(rough.get("rough_idea") or rough.get("research_goal") or ""),
        output=feature_plan_path,
    )
    capabilities = merge_research_capabilities(
        task_capabilities, dataset_adapter.dispatch("capabilities")
    )
    context_capabilities = dict(capabilities)
    context_capabilities["operations"] = capabilities.get("operation_capabilities")
    context = {
        "schema_version": "omni-ar-live-research-context/v1", "round": 0,
        "rough_idea": rough,
        "idea_card": idea_card.read_text(encoding="utf-8")[:30000],
        "task_spec": spec,
        "dataset": {
            "ref": rough["dataset"]["ref"], "mode": rough["dataset"]["usage_mode"],
            "capabilities": context_capabilities, "feature_plan": feature_plan,
        },
        "paths": {"task_spec": str(task_spec), "idea_card": str(idea_card)},
        "execution_authorized": False,
    }
    context_path = output_dir / "heuresis_context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    proposal_path = output_dir / "heuresis_proposal.json"
    heuresis = repository / "Heuresis_PJLAB-boyue"
    env = dict(os.environ)
    env.update({
        "HEURESIS_DIR": str(heuresis), "HEURESIS_ENV_FILE": str(env_file),
        "BOYUE_DISABLE_PROXY": os.environ.get("OMNI_AR_BOYUE_DISABLE_PROXY", "1"),
        "HEURESIS_SUGGESTION_TIMEOUT_SEC": os.environ.get(
            "HEURESIS_SUGGESTION_TIMEOUT_SEC", "180"
        ),
        "HEURESIS_SUGGESTION_MAX_RETRIES": os.environ.get(
            "HEURESIS_SUGGESTION_MAX_RETRIES", "2"
        ),
        "HEURESIS_SUGGESTION_JSON_RETRIES": os.environ.get(
            "HEURESIS_SUGGESTION_JSON_RETRIES", "2"
        ),
        "HEURESIS_DATASET_TOOL_MAX_CALLS": os.environ.get(
            "HEURESIS_DATASET_TOOL_MAX_CALLS", "4"
        ),
        "HEURESIS_SUGGESTION_TEMPERATURE": os.environ.get(
            "HEURESIS_SUGGESTION_TEMPERATURE", "0.2"
        ),
    })
    if os.environ.get("OMNI_AR_BOYUE_MODEL"):
        env["BOYUE_MODEL_NAME"] = os.environ["OMNI_AR_BOYUE_MODEL"]
    context_sha = _sha(context_path)
    hs_cached = same_input and _cache_valid(
        cache, "heuresis", [proposal_path], context_sha256=context_sha,
    )
    if hs_cached:
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            validate_proposal(proposal, spec, task_spec)
        except (OSError, json.JSONDecodeError, ValueError):
            hs_cached = False
    if hs_cached:
        _progress(
            output_dir, "heuresis_boyue", "cached",
            message="reused validated proposal from matching research context",
        )
    else:
        _run_external_stage(
            [sys.executable, str(repository / "controller_bash/scripts/heuresis_suggest.py"),
             "--context", str(context_path), "--out", str(proposal_path)],
            cwd=repository, env=env, output_dir=output_dir,
            stage="heuresis_boyue", timeout_s=float(os.environ.get(
                "OMNI_AR_HEURESIS_STAGE_TIMEOUT_SEC", "420"
            )), stdout_name="heuresis.stdout.log", stderr_name="heuresis.stderr.log",
            checkpoints=[("proposal", proposal_path)],
        )
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        validate_proposal(proposal, spec, task_spec)
        cache.setdefault("stages", {})["heuresis"] = {
            "status": "completed", "context_sha256": context_sha,
            "artifacts": _cache_artifacts([proposal_path]),
        }
        _write_json(cache_path, cache)
    _progress(
        output_dir, "task_adapter_compile", "running",
        message=f"compiling {len(proposal['experiment_proposals'])} proposal(s)",
    )
    defaults = spec["adapter"]["trial_defaults"]
    compiled = [
        adapter.proposal_to_trial(item, defaults, f"experiment_{index:03d}")
        for index, item in enumerate(proposal["experiment_proposals"])
    ]
    compiled_path = output_dir / "compiled_trials.json"
    compiled_path.write_text(json.dumps({"status": "ok", "execution_authorized": False, "trials": compiled}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _progress(
        output_dir, "task_adapter_compile", "completed",
        message=f"compiled {len(compiled)} trial(s)",
    )
    artifacts = {
        "rough_idea": rough_path,
        "researchstudio_result": researchstudio_dir / "researchstudio_result.json",
        "literature_results": researchstudio_dir / "ideaspark_live/phase0/lit_results.json",
        "literature_table": researchstudio_dir / "ideaspark_live/phase0/lit_table.md",
        "idea_card": idea_card,
        "method_view": researchstudio_dir / "method_view.json",
        "heuresis_context": context_path,
        "heuresis_proposal": proposal_path,
        "compiled_trials": compiled_path,
        "feature_plan": feature_plan_path,
    }
    missing = [name for name, path in artifacts.items() if not path.exists()]
    if missing:
        raise InitializationError(f"live pipeline missing trace artifacts: {missing}")
    trace = {
        "schema_version": "omni-ar-live-research-trace/v1", "status": "ok",
        "initialization_id": rough["initialization_id"], "execution_authorized": False,
        "stages": [
            {"name": "initialization", "inputs": [], "outputs": ["rough_idea"]},
            {"name": "researchstudio_literature_and_idea", "inputs": ["rough_idea"], "outputs": ["literature_results", "literature_table", "idea_card", "method_view"]},
            {"name": "heuresis_boyue_proposal", "inputs": ["rough_idea", "idea_card", "literature_results"], "outputs": ["heuresis_proposal"]},
            {"name": "task_adapter_compile", "inputs": ["heuresis_proposal"], "outputs": ["compiled_trials"]},
        ],
        "artifacts": {name: {"path": str(path), "sha256": _sha(path)} for name, path in artifacts.items()},
        "checks": {
            "literature_grounding": (researchstudio_dir / "ideaspark_live/phase0/.lit_grounding_mode").read_text().strip(),
            "connectors_degraded": (researchstudio_dir / "ideaspark_live/phase0/.connectors_degraded").exists(),
            "proposal_schema": "ok", "task_adapter_compile": "ok",
            "num_proposals": len(compiled), "training_started": False, "rjob_submitted": False,
        },
    }
    trace_path = output_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return trace_path
