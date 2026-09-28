"""Unified, auditable multi-round AutoResearch orchestration.

The orchestrator deliberately separates planning from evidence.  Dry-run and
simulation modes exercise lineage and termination, but can never produce a
scientific winner.  Only a standard result accepted by the controller archive
gate is eligible for the final reproducibility package.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .finalizer import build_final_package, choose_winner, refresh_records_artifact
from .initialization import InitializationError, RoughIdeaEngine, dry_run_pipeline
from .live_pipeline import _progress, _run_external_stage, run_live_pipeline
from .project_records import ProjectRecords, write_json
from .preflight import PreflightError, run_preflight


class AutoResearchError(RuntimeError):
    pass


def _call_with_retry(
    records: ProjectRecords, stage: str, attempts: int, function,
):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        records.stage(stage, "running", max_attempts=attempts)
        try:
            value = function()
        except Exception as exc:  # noqa: BLE001 - the caller owns final handling
            last_error = exc
            records.stage(stage, "failed", error=f"{type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(float(os.environ.get("OMNI_AR_STAGE_RETRY_BACKOFF_SEC", "1")))
                continue
            raise
        records.stage(stage, "completed")
        return value
    raise AutoResearchError(str(last_error))


def _reuse_planning_artifacts(project_dir: Path, mode: str) -> tuple[Path, Path, Path | None] | None:
    planning = project_dir / "planning"
    trace_path = planning / "trace.json"
    if not trace_path.is_file():
        return None
    if mode == "live":
        trace = _load(trace_path)
        artifacts = trace.get("artifacts") or {}
        try:
            return (
                Path(artifacts["heuresis_proposal"]["path"]),
                Path(artifacts["idea_card"]["path"]),
                Path(artifacts["heuresis_context"]["path"]),
            )
        except (KeyError, TypeError):
            return None
    proposal = planning / "heuresis_proposal.json"
    idea = planning / "idea_card/idea.detail.en.md"
    return (proposal, idea, None) if proposal.is_file() and idea.is_file() else None


def _restore_strategy(strategy: "StrategyBridge", entries: list[dict[str, Any]]) -> None:
    for item in sorted(entries, key=lambda value: (int(value.get("round", 0)), str(value.get("candidate_id")))):
        if item.get("bucket") != "accepted" or item.get("scientific_evidence") is not True:
            continue
        if not isinstance(item.get("score"), (int, float)):
            continue
        strategy.on_result(
            str(item["candidate_id"]), float(item["score"]),
            idea=str(item.get("hypothesis") or ""),
            parent_ids=list(item.get("parent_ids") or []),
            round_number=int(item.get("round", 0)),
        )


def _load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AutoResearchError(f"expected mapping in {path}")
    return value


def _controller_api(repository: Path):
    scripts = repository / "controller_bash/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from task_contract import load_task_spec, validate_proposal

    return load_task_spec, validate_proposal


def _adapter_activation_requirements(
    repository: Path, spec: dict[str, Any], task_spec: Path,
    experiments: list[dict[str, Any]],
) -> list[list[str]]:
    """Resolve runtime diagnostics required by an already implemented method."""
    scripts = repository / "controller_bash/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from task_contract import load_task_adapter

    adapter = load_task_adapter(spec, task_spec)
    capabilities = adapter.initialization_capabilities(spec)
    contract = capabilities.get("activation_contract") or {}
    method_specific = contract.get("method_specific") or {}
    common = list(contract.get("diagnostics") or [])
    defaults = dict(spec["adapter"]["trial_defaults"])
    requirements: list[list[str]] = []
    for experiment in experiments:
        parameters = {**defaults, **dict(experiment.get("parameters") or {})}
        method = str(parameters.get("method") or "")
        required = method_specific.get(method)
        requirements.append(list(required if required is not None else common))
    return requirements


def _direct_proposal(
    repository: Path, spec: dict[str, Any], task_spec: Path,
    proposal: dict[str, Any], output: Path, *, max_experiments: int | None = None,
) -> Path | None:
    """Write the subset that the current Adapter can run without code changes."""
    scripts = repository / "controller_bash/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from task_contract import load_task_adapter

    adapter = load_task_adapter(spec, task_spec)
    defaults = spec["adapter"]["trial_defaults"]
    experiments = []
    for index, experiment in enumerate(proposal.get("experiment_proposals") or []):
        try:
            adapter.proposal_to_trial(experiment, defaults, f"experiment_{index:03d}")
        except Exception:  # implemented only inside a Coding Agent candidate
            continue
        experiments.append(experiment)
    if max_experiments is not None:
        experiments = experiments[:max(0, max_experiments)]
    if not experiments:
        return None
    direct = {**proposal, "experiment_proposals": experiments, "implementation_requests": []}
    write_json(output, direct)
    return output


def _proposal_execution_partition(
    repository: Path, spec: dict[str, Any], task_spec: Path,
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate runnable trials from requests that still require code changes."""
    scripts = repository / "controller_bash/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from task_contract import load_task_adapter

    adapter = load_task_adapter(spec, task_spec)
    defaults = spec["adapter"]["trial_defaults"]
    direct: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for index, experiment in enumerate(proposal.get("experiment_proposals") or []):
        try:
            adapter.proposal_to_trial(experiment, defaults, f"experiment_{index:03d}")
        except Exception:
            continue
        direct.append(experiment)
    for index, request in enumerate(proposal.get("implementation_requests") or []):
        trial = request["trial_proposal"]
        try:
            adapter.proposal_to_trial(trial, defaults, f"implemented_{index:03d}")
        except Exception:
            unresolved.append(request)
        else:
            if trial not in direct:
                direct.append(trial)
    return direct, unresolved


def _request_fingerprint(request: dict[str, Any]) -> str:
    """Stable key used to avoid paying for the same failed code request twice."""
    payload = {
        "hypothesis": request.get("hypothesis"),
        "change_scope": request.get("change_scope") or [],
        "required_capabilities": request.get("required_capabilities") or [],
        "trial": (request.get("trial_proposal") or {}).get("parameters") or {},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _execution_context_for_parents(
    repository: Path, task_spec: Path, archive: list[dict[str, Any]],
    parent_ids: list[str],
) -> tuple[Path, Path, str | None]:
    """Return the nearest selected ancestor whose generated source still exists."""
    by_id = {str(item.get("candidate_id")): item for item in archive}
    pending = list(parent_ids)
    visited: set[str] = set()
    while pending:
        candidate_id = pending.pop(0)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        item = by_id.get(candidate_id) or {}
        worktree_value = item.get("candidate_worktree")
        if worktree_value:
            worktree = Path(str(worktree_value)).resolve()
            if worktree.is_dir():
                try:
                    relative = task_spec.resolve().relative_to(repository.resolve())
                except ValueError:
                    relative = Path("tasks") / task_spec.parent.name / task_spec.name
                inherited_task = worktree / relative
                if inherited_task.is_file():
                    return worktree, inherited_task, candidate_id
        pending.extend(str(value) for value in item.get("parent_ids") or [])
    return repository, task_spec, None


def _dataset_lineage(spec: dict[str, Any]) -> dict[str, Any]:
    dataset = spec.get("dataset") or spec.get("data") or {}
    return {
        key: dataset.get(key) for key in (
            "id", "version", "dataset_ref", "manifest", "manifest_sha256",
            "train_path", "validation_path", "test_path",
        ) if dataset.get(key) is not None
    }


def _proposal_text(proposal: dict[str, Any]) -> str:
    parts = [str(proposal.get("verdict") or "")]
    for item in proposal.get("experiment_proposals") or []:
        parts.append(str(item.get("hypothesis") or ""))
        parts.extend(str(scope) for scope in item.get("change_scope") or [])
        parts.append(json.dumps(item.get("parameters") or {}, sort_keys=True))
    for item in proposal.get("implementation_requests") or []:
        parts.append(str(item.get("hypothesis") or ""))
    return "\n".join(parts)


def _validate_execution_policy(proposal: dict[str, Any], policy: dict[str, Any]) -> None:
    if not policy:
        return
    experiments = list(proposal.get("experiment_proposals") or [])
    implementations = list(proposal.get("implementation_requests") or [])
    if int(policy["max_trials_per_round"]) < 1:
        raise AutoResearchError("QA-confirmed trial limit must allow at least one trial")
    requests = [item.get("resource_request") or {} for item in experiments]
    requests.extend(
        ((item.get("trial_proposal") or {}).get("resource_request") or {})
        for item in implementations
    )
    for request in requests:
        if int(request.get("gpu_count", 0)) > int(policy["max_gpus"]):
            raise AutoResearchError("proposal exceeds the QA-confirmed GPU limit")
        if int(request.get("max_runtime_minutes", 0)) > int(policy["max_runtime_minutes"]):
            raise AutoResearchError("proposal exceeds the QA-confirmed runtime limit")


class StrategyBridge:
    """Use Heuresis' native parent selection and generation bookkeeping."""

    def __init__(self, repository: Path, name: str, direction: str) -> None:
        heuresis_src = repository / "Heuresis_PJLAB-boyue/src"
        if str(heuresis_src) not in sys.path:
            sys.path.insert(0, str(heuresis_src))
        try:
            from heuresis.qd import ArchiveIndex, IslandSearch, LexicalHashEmbedder, OmniEpicSearch
        except ModuleNotFoundError as exc:
            raise AutoResearchError(
                "Heuresis runtime dependencies are missing from this Python interpreter; "
                "invoke the workflow through ./run_research or set OMNI_AR_PYTHON to the project venv"
            ) from exc

        self.name = name
        self.index = None
        self._moi_assessment = None
        if name == "islands":
            self.strategy = IslandSearch(
                num_islands=3, topology="ring", max_population=30,
                maximize=direction == "maximize", crossover_rate=0.4,
                parent_k=2, migration_interval=6, seed=42,
            )
        elif name == "omni_epic":
            self.index = ArchiveIndex(LexicalHashEmbedder(dim=512))
            # Parent selection, archive retrieval and generation tracking do
            # not call reviewer.review(). Proposal validity is gated by the
            # structured Heuresis planner and controller fail-closed checks.
            self.strategy = OmniEpicSearch(
                self.index, reviewer=None, lower_is_better=direction == "minimize"
            )
        else:
            raise AutoResearchError(f"unsupported search strategy: {name}")

    def configure_moi(
        self, repository: Path, project_dir: Path, spec: dict[str, Any],
        rough: dict[str, Any], idea_card: Path,
    ) -> None:
        if self.name != "omni_epic" or self.index is None:
            return
        from heuresis.qd import MoIContext, MoIReviewer_Boyue

        env_path = repository / "Heuresis_PJLAB-boyue/.env"
        env_lines = env_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines() if env_path.is_file() else []
        for raw in env_lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        task_dir = project_dir / "records/moi_context"
        task_dir.mkdir(parents=True, exist_ok=True)
        metric = spec["metrics"]["primary"]
        baseline = (rough.get("evaluation") or {}).get("baseline") or {}
        (task_dir / "task_config.yaml").write_text(yaml.safe_dump({
            "name": spec["task"]["name"],
            "description": spec["task"]["description"],
        }, sort_keys=False), encoding="utf-8")
        (task_dir / "baseline_scores.yaml").write_text(yaml.safe_dump({
            "metric": metric["name"],
            "baseline": baseline.get("value"),
            "objective": "min" if metric["direction"] == "minimize" else "max",
        }, sort_keys=False), encoding="utf-8")
        problem = idea_card.read_text(encoding="utf-8", errors="replace")[:8000]
        (task_dir / "description.md").write_text(
            spec["task"]["description"] + "\n", encoding="utf-8"
        )
        self.strategy.reviewer = MoIReviewer_Boyue(
            self.index, task_dir,
            model=os.environ.get("BOYUE_MODEL_NAME", "deepseek-v4-flash"),
            min_archive_size=int(os.environ.get("OMNI_AR_MOI_MIN_ARCHIVE_SIZE", "10")),
            request_timeout_s=float(os.environ.get("OMNI_AR_MOI_TIMEOUT_SEC", "120")),
            max_attempts=int(os.environ.get("OMNI_AR_MOI_MAX_ATTEMPTS", "3")),
            context=MoIContext(
                task_name=spec["task"]["name"],
                task_description=spec["task"]["description"],
                domain_description=str((rough.get("research_intent") or {}).get("rough_idea") or spec["task"]["description"]),
                problem_text=problem, metric=metric["name"],
                baseline=baseline.get("value"),
                lower_is_better=metric["direction"] == "minimize",
            ),
        )

    def review(self, idea: str):
        if self.name != "omni_epic" or self.strategy.reviewer is None:
            return None
        self._moi_assessment = self.strategy.review_idea(idea)
        return self._moi_assessment

    def reject(self, candidate_id: str, idea: str, parent_ids: list[str]) -> dict[str, Any]:
        metadata = self.strategy.on_moi_rejected(
            candidate_id, idea, self._moi_assessment, parent_ids=parent_ids,
        )
        metadata["bucket"] = "failed_moi"
        return metadata

    def select(self, round_number: int) -> tuple[list[str], str]:
        ideator_id = round_number % 3
        parents = self.strategy.select_parents(ideator_id=ideator_id)
        return parents, self.strategy.context(ideator_id=ideator_id)

    def on_result(
        self, candidate_id: str, score: float | None, *, idea: str,
        parent_ids: list[str], round_number: int,
    ) -> dict[str, Any]:
        kwargs = {}
        if self.name == "omni_epic":
            kwargs["moi_assessment"] = self._moi_assessment
        metadata = self.strategy.on_result(
            candidate_id, score, idea=idea, parent_ids=parent_ids,
            ideator_id=round_number % 3, **kwargs,
        )
        if self.name == "omni_epic":
            metadata["bucket"] = metadata.get("omniepic_bucket")
        else:
            metadata["bucket"] = "accepted" if score is not None else "failed_train"
        return metadata


def _planning_artifacts(
    repository: Path, rough_path: Path, project_dir: Path, mode: str,
) -> tuple[Path, Path, Path | None]:
    planning = project_dir / "planning"
    if mode == "live":
        trace_path = run_live_pipeline(repository, rough_path, planning)
        trace = _load(trace_path)
        artifacts = trace.get("artifacts") or {}
        proposal = Path(artifacts["heuresis_proposal"]["path"])
        idea = Path(artifacts["idea_card"]["path"])
        context = Path(artifacts["heuresis_context"]["path"])
        return proposal, idea, context
    result = dry_run_pipeline(repository, rough_path, planning)
    proposal = Path(result["artifacts"]["heuresis_proposal"])
    idea = Path(result["artifacts"]["idea_card"])
    return proposal, idea, None


def _next_live_proposal(
    repository: Path, base_context: Path, round_dir: Path, round_number: int,
    records: ProjectRecords, parent_ids: list[str], search_context: str,
    active_task_spec: Path | None = None,
) -> Path:
    context = _load(base_context)
    if active_task_spec is not None:
        active_repository = active_task_spec.parents[2]
        load_spec, _ = _controller_api(active_repository)
        active_spec = load_spec(active_task_spec)
        scripts = active_repository / "controller_bash/scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        from task_contract import load_task_adapter
        from tasks.adapter_contract import merge_research_capabilities
        from datasets.registry import DatasetRegistry
        from omni_ar.feature_planner import plan_features

        dataset_ref = str((active_spec.get("data") or {}).get("dataset_ref") or "")
        if dataset_ref:
            registry = DatasetRegistry(active_repository / "datasets")
            dataset_adapter = registry.load_adapter(dataset_ref)
            task_adapter = load_task_adapter(active_spec, active_task_spec)
            capabilities = merge_research_capabilities(
                task_adapter.initialization_capabilities(active_spec),
                dataset_adapter.dispatch("capabilities"),
            )
            capabilities["operations"] = capabilities.get("operation_capabilities")
            context.setdefault("dataset", {})["feature_plan"] = plan_features(
                active_repository, dataset_ref,
                rough_idea=str((context.get("rough_idea") or {}).get("rough_idea") or ""),
            )
            context["task_spec"] = active_spec
            context.setdefault("dataset", {})["capabilities"] = capabilities
            context["active_code"] = {
                "task_spec": str(active_task_spec),
                "method_capabilities": capabilities.get("method_capabilities") or [],
                "trial_parameters": capabilities.get("trial_parameters") or [],
            }
    context.update({
        "round": round_number,
        "archive_context": search_context,
        "selected_parent_ids": parent_ids,
        "prior_candidates": [
            {
                key: item.get(key) for key in (
                    "candidate_id", "generation", "bucket", "score", "direction",
                    "hypothesis", "route", "parent_ids",
                )
            }
            for item in records.archive()
        ],
        "execution_authorized": False,
    })
    context_path = round_dir / "heuresis_context.json"
    write_json(context_path, context)
    proposal_path = round_dir / "heuresis_proposal.json"
    cache_path = round_dir / "proposal_cache.json"
    context_sha = hashlib.sha256(context_path.read_bytes()).hexdigest()
    try:
        proposal_cache = _load(cache_path)
    except (OSError, ValueError, json.JSONDecodeError):
        proposal_cache = {}
    if (
        proposal_cache.get("context_sha256") == context_sha
        and proposal_path.is_file()
        and proposal_cache.get("proposal_sha256")
        == hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    ):
        _progress(
            round_dir, f"heuresis_round_{round_number:03d}", "cached",
            message="reused proposal from matching round context",
        )
        return proposal_path
    env = dict(os.environ)
    env.update({
        "HEURESIS_DIR": str(repository / "Heuresis_PJLAB-boyue"),
        "HEURESIS_ENV_FILE": str(repository / "Heuresis_PJLAB-boyue/.env"),
        "BOYUE_DISABLE_PROXY": os.environ.get("OMNI_AR_BOYUE_DISABLE_PROXY", "1"),
    })
    # Keep one model choice across ResearchStudio and every later Heuresis
    # generation.  Previously only the initial planning stage honored
    # OMNI_AR_BOYUE_MODEL, so later rounds silently fell back to the model in
    # Heuresis' .env and could exhibit very different latency/availability.
    if os.environ.get("OMNI_AR_BOYUE_MODEL"):
        env["BOYUE_MODEL_NAME"] = os.environ["OMNI_AR_BOYUE_MODEL"]
    env.setdefault("HEURESIS_SUGGESTION_TIMEOUT_SEC", "180")
    # Match the initial live-planning stage: allow one retry for transient
    # service failures, bounded by OMNI_AR_HEURESIS_STAGE_TIMEOUT_SEC.
    env.setdefault("HEURESIS_SUGGESTION_MAX_RETRIES", "2")
    env.setdefault("HEURESIS_SUGGESTION_JSON_RETRIES", "2")
    env.setdefault("HEURESIS_DATASET_TOOL_MAX_CALLS", "4")
    _run_external_stage(
        [sys.executable, str(repository / "controller_bash/scripts/heuresis_suggest.py"),
         "--context", str(context_path), "--out", str(proposal_path)],
        cwd=repository, env=env, output_dir=round_dir,
        stage=f"heuresis_round_{round_number:03d}", timeout_s=float(os.environ.get(
            "OMNI_AR_HEURESIS_STAGE_TIMEOUT_SEC", "420"
        )), stdout_name="heuresis.stdout.log", stderr_name="heuresis.stderr.log",
        checkpoints=[("proposal", proposal_path)],
    )
    write_json(cache_path, {
        "schema_version": "omni-ar-proposal-cache/v1",
        "context_sha256": context_sha,
        "proposal_sha256": hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
    })
    return proposal_path


def _next_offline_proposal(initial: Path, round_dir: Path, round_number: int) -> Path:
    proposal = copy.deepcopy(_load(initial))
    proposal["proposal_id"] = f"{proposal['proposal_id']}-round-{round_number:03d}"
    proposal["round"] = round_number
    proposal["verdict"] = (
        f"Offline orchestration validation round {round_number}; "
        "this repeated proposal is not new scientific evidence."
    )
    path = round_dir / "heuresis_proposal.json"
    write_json(path, proposal)
    return path


def _primary_from_standardized(path: str | Path | None) -> tuple[float | None, str | None]:
    if not path or not Path(path).is_file():
        return None, None
    result = _load(Path(path))
    metrics = result.get("metrics") or {}
    for value in metrics.values():
        if isinstance(value, dict) and value.get("role") == "primary":
            score = value.get("value")
            return (float(score), value.get("direction")) if isinstance(score, (int, float)) else (None, value.get("direction"))
    return None, None


def _selection_metadata(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {}
    value = _load(Path(path))
    protocol = value.get("protocol") or {}
    stability = (
        protocol.get("stability") or protocol.get("multi_seed_stability")
        or value.get("multi_seed_stability") or {}
    )
    seed_values = stability.get("values") or stability.get("scores") or []
    return {
        "baseline_metrics": value.get("baseline_metrics") or {},
        "resource_usage": value.get("resource_usage") or {},
        "protocol": protocol,
        "stability": stability,
        "seed_count": (
            len(seed_values) if isinstance(seed_values, list) and seed_values
            else int(stability.get("num_seeds", stability.get("seed_count", 1)))
        ),
    }


def _activation_report(
    standardized_path: str | Path | None, required: list[str] | None,
) -> dict[str, Any]:
    """Verify generated-code diagnostics from the raw worker result.

    Static name checks prevent accidental omissions in source. This runtime
    check proves that the candidate path executed and reported an active state.
    """
    names = list(required or [])
    report: dict[str, Any] = {"required": names, "values": {}, "missing": [], "inactive": []}
    if not names:
        report["passed"] = True
        return report
    raw: dict[str, Any] = {}
    try:
        standardized = _load(Path(str(standardized_path)))
        raw_path = (standardized.get("artifacts") or {}).get("raw_result")
        raw = _load(Path(str(raw_path))) if raw_path else {}
    except (OSError, ValueError, json.JSONDecodeError):
        raw = {}
    diagnostics = raw.get("activation_diagnostics") if isinstance(raw, dict) else None
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}

    def active(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return math.isfinite(float(value)) and float(value) != 0.0
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "off", "disabled", "none", "null"}
        return bool(value)

    for name in names:
        if name not in diagnostics:
            report["missing"].append(name)
            continue
        report["values"][name] = diagnostics[name]
        if not active(diagnostics[name]):
            report["inactive"].append(name)
    report["passed"] = not report["missing"] and not report["inactive"]
    return report


def _executed_parameter_report(
    standardized_path: str | Path | None, expected: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check that proposal parameters reached the Task Adapter unchanged."""
    expected = dict(expected or {})
    report: dict[str, Any] = {
        "expected": expected, "observed": {}, "missing": [], "mismatched": [],
    }
    try:
        standardized = _load(Path(str(standardized_path)))
        observed = (standardized.get("protocol") or {}).get("executed_parameters")
        observed = dict(observed) if isinstance(observed, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        observed = {}
    for name, value in expected.items():
        if name not in observed:
            report["missing"].append(name)
            continue
        report["observed"][name] = observed[name]
        if observed[name] != value:
            report["mismatched"].append(name)
    report["passed"] = not report["missing"] and not report["mismatched"]
    return report


def _simulated_trace(
    round_dir: Path, proposal: dict[str, Any], spec: dict[str, Any], round_number: int,
) -> Path:
    direction = spec["metrics"]["primary"]["direction"]
    baseline = next(iter((proposal["experiment_proposals"][0].get("acceptance_criteria") or {}).values()), {}).get("value", 0.0)
    delta = 0.001 * (round_number + 1)
    score = float(baseline) + delta if direction == "maximize" else float(baseline) - delta
    result_path = round_dir / "simulation_standardized_result.json"
    write_json(result_path, {
        "schema_version": "omni-ar-result/v1", "status": "simulated",
        "task": spec["task"]["name"], "trial_id": f"simulation-{round_number}",
        "metrics": {spec["metrics"]["primary"]["name"]: {
            "value": score, "direction": direction, "role": "primary",
            "beats_baseline": None, "constraint_satisfied": None,
        }},
        "baseline_metrics": {}, "resource_usage": {"within_budget": None},
        "protocol": {"simulation_only": True}, "artifacts": {},
    })
    trace_path = round_dir / "execution_trace.json"
    write_json(trace_path, {
        "schema_version": "omni-ar-live-execution-trace/v1",
        "status": "simulated", "scientific_evidence": False,
        "task": spec["task"]["name"], "proposal_id": proposal["proposal_id"],
        "execution_mode": "simulated", "technically_complete": True,
        "accepted": 1, "rejected": 0,
        "jobs": [{
            "job_name": f"simulation-{round_number}", "status": "simulated",
            "archive_gate": {"decision": "accept", "eligible": True, "simulation_only": True},
            "standardized_result": str(result_path),
        }],
    })
    return trace_path


def _execute(
    repository: Path, task_spec: Path, proposal_path: Path, output_dir: Path,
    execution_mode: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    proposal = _load(proposal_path)
    load_spec, _ = _controller_api(repository)
    spec = load_spec(task_spec)
    if execution_mode == "simulated":
        return _simulated_trace(output_dir, proposal, spec, int(proposal["round"]))
    proc = subprocess.run(
        [sys.executable, str(repository / "controller_bash/scripts/execute_proposal.py"),
         "--task", str(task_spec), "--proposal", str(proposal_path),
         "--output-dir", str(output_dir), "--execution-mode", execution_mode],
        cwd=repository, text=True, capture_output=True, check=False,
    )
    (output_dir / "execute.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "execute.stderr.log").write_text(proc.stderr, encoding="utf-8")
    trace = output_dir / "execution_trace.json"
    if not trace.is_file():
        raise AutoResearchError(f"execution produced no trace; see {output_dir}")
    return trace


def _execute_with_transient_check(
    repository: Path, task_spec: Path, proposal_path: Path, output_dir: Path,
    execution_mode: str,
) -> Path:
    trace_path = _execute(repository, task_spec, proposal_path, output_dir, execution_mode)
    trace = _load(trace_path)
    jobs = trace.get("jobs") or []
    if jobs and all(
        (job.get("archive_gate") or {}).get("retryable") is True
        and (job.get("archive_gate") or {}).get("failure_class") == "rjob_submit_transient"
        for job in jobs
    ):
        raise AutoResearchError("all rjob submissions failed transiently")
    return trace_path


def _result_evidence_policy(standardized_result: str | None) -> dict[str, Any]:
    """Classify a result's evidence role from the standard result protocol.

    A real training backend is necessary but not sufficient for scientific
    evidence.  Activation/smoke/control runs may contain real metrics while
    being intentionally ineligible for archive selection.
    """
    if not standardized_result:
        return {"scientific": True, "bucket": None, "reason": None}
    try:
        result = _load(Path(standardized_result))
    except (OSError, ValueError, json.JSONDecodeError, AutoResearchError):
        # Result validity is handled by the controller gate.  Do not silently
        # change legacy behaviour here when the result cannot be inspected.
        return {"scientific": True, "bucket": None, "reason": None}
    protocol = result.get("protocol") or {}
    explicit = protocol.get("scientific_evidence")
    name = str(protocol.get("name") or "").strip().lower()
    non_scientific_names = {
        "activation_verification", "activation_smoke", "smoke",
        "dry_run", "control", "negative_control",
    }
    if explicit is False or name in non_scientific_names:
        return {
            "scientific": False,
            "bucket": "verification" if "activation" in name else "control",
            "reason": f"non_scientific_protocol:{name or 'explicit_flag'}",
        }
    return {"scientific": True, "bucket": None, "reason": None}


def _candidate_records(
    proposal: dict[str, Any], proposal_path: Path, trace_path: Path,
    round_number: int, parent_ids: list[str], strategy: StrategyBridge,
    *, route: str = "adapter", code_patch: str | None = None,
    code_snapshot: dict[str, Any] | None = None,
    changed_paths: list[str] | None = None,
    candidate_prefix: str | None = None, candidate_worktree: str | None = None,
    required_activation_diagnostics: list[str] | None = None,
    activation_diagnostics_by_experiment: list[list[str]] | None = None,
    dataset_lineage: dict[str, Any] | None = None,
    code_parent_id: str | None = None,
    control_roles: list[list[str]] | None = None,
    inherited_control_validation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trace = _load(trace_path)
    trace_scientific_evidence = trace.get("scientific_evidence") is not False
    execution_mode = trace.get("execution_mode")
    non_scientific_bucket = "simulated" if execution_mode == "simulated" else "dry_run"
    records: list[dict[str, Any]] = []
    experiments = proposal.get("experiment_proposals") or []
    for index, job in enumerate(trace.get("jobs") or []):
        gate = dict(job.get("archive_gate") or {})
        evidence_policy = _result_evidence_policy(job.get("standardized_result"))
        scientific_evidence = trace_scientific_evidence and evidence_policy["scientific"]
        if trace_scientific_evidence and not evidence_policy["scientific"]:
            gate = {
                "eligible": False, "decision": "reject",
                "reasons": [evidence_policy["reason"]],
                "failure_class": "non_scientific_evidence", "retryable": False,
                "archive_bucket": evidence_policy["bucket"],
                "outcome": "activation_verified" if evidence_policy["bucket"] == "verification" else "non_scientific_control",
            }
        score, direction = _primary_from_standardized(job.get("standardized_result"))
        candidate_id = candidate_prefix or f"round-{round_number:03d}-trial-{index:03d}"
        if candidate_prefix and len((trace.get("jobs") or [])) > 1:
            candidate_id = f"{candidate_prefix}-trial-{index:03d}"
        experiment = experiments[index] if index < len(experiments) else {}
        roles = control_roles[index] if control_roles and index < len(control_roles) else []
        adapter_required = (
            activation_diagnostics_by_experiment[index]
            if activation_diagnostics_by_experiment
            and index < len(activation_diagnostics_by_experiment)
            else []
        )
        inherited_required = list(required_activation_diagnostics or [])
        required = list(dict.fromkeys([*inherited_required, *adapter_required]))
        if set(roles).intersection({"baseline", "disabled"}):
            required = []
        activation = _activation_report(job.get("standardized_result"), required)
        validates_generated_code = (
            route == "coding_agent" and set(roles).intersection({"enabled", "combination"})
        ) or (bool(candidate_worktree) and bool(required))
        if validates_generated_code:
            parameter_report = _executed_parameter_report(
                job.get("standardized_result"), experiment.get("parameters") or {},
            )
            activation["parameter_validation"] = parameter_report
            activation["passed"] = activation["passed"] and parameter_report["passed"]
        if not activation["passed"]:
            gate = {
                "eligible": False, "decision": "reject",
                "reasons": ["activation_diagnostics_missing_or_inactive"],
                "failure_class": "method_inactive", "retryable": False,
                "archive_bucket": "invalid",
            }
            eligible = False
            strategy_score = None
        else:
            eligible = gate.get("eligible") is True and scientific_evidence
            # Synthetic scores may drive an explicitly simulated strategy smoke,
            # but dry-run placeholders never influence parent selection.
            strategy_score = score if (
                gate.get("eligible") is True
                and (scientific_evidence or execution_mode == "simulated")
            ) else None
        is_control_only = bool(roles) and not set(roles).intersection({"enabled", "combination"})
        if is_control_only:
            strategy_score = None
        idea = str(experiment.get("hypothesis") or proposal.get("verdict") or "")
        # Only a formally accepted scientific candidate is inserted into the
        # strategy's selectable archive.  Verification/control results and
        # typed rejections remain in the unified project records with their
        # real outcome; sending score=None to OMNI-EPIC would incorrectly
        # relabel all of them as failed_train.
        strategy_smoke = (
            execution_mode == "simulated" and gate.get("eligible") is True
        )
        if eligible or strategy_smoke:
            metadata = strategy.on_result(
                candidate_id, strategy_score, idea=idea,
                parent_ids=parent_ids, round_number=round_number,
            )
        else:
            recorded_outcome = (
                "activation_verified" if evidence_policy["bucket"] == "verification"
                else "non_scientific_control" if not scientific_evidence
                else str(gate.get("outcome") or gate.get("failure_class") or "invalid")
            )
            metadata = {
                "parent_ids": parent_ids, "generation": round_number,
                "bucket": recorded_outcome, "idea": idea,
                "excluded_from_search_archive": True,
            }
        records.append({
            "candidate_id": candidate_id, "round": round_number,
            "generation": metadata.get("generation", 0),
            "bucket": (
                str(evidence_policy["bucket"] or non_scientific_bucket)
                if not scientific_evidence else
                "control" if is_control_only and gate.get("eligible") is True
                else ("accepted" if eligible else str(gate.get("archive_bucket") or "invalid"))
            ),
            "strategy_bucket": metadata.get("bucket"),
            "score": score if eligible else None, "observed_score": score,
            "direction": direction, "scientific_evidence": scientific_evidence,
            "proposal_id": proposal.get("proposal_id"),
            "proposal_path": str(proposal_path), "trial_id": job.get("job_name"),
            "hypothesis": experiment.get("hypothesis"), "route": route,
            "parent_ids": parent_ids, "strategy_metadata": metadata,
            "archive_gate": gate, "standardized_result": job.get("standardized_result"),
            "outcome": gate.get("outcome") or (
                "accepted" if eligible else gate.get("failure_class") or "invalid"
            ),
            "execution_trace": str(trace_path), "code_patch": code_patch,
            "code_snapshot": code_snapshot, "changed_paths": changed_paths or [],
            "candidate_worktree": candidate_worktree,
            "activation_validation": activation,
            "parameters": experiment.get("parameters") or {},
            "resource_request": experiment.get("resource_request") or {},
            "control_roles": roles,
            "control_validation": inherited_control_validation,
            "code_parent_id": code_parent_id,
            "dataset_lineage": dataset_lineage or {},
            **_selection_metadata(job.get("standardized_result")),
        })
    return records


def _run_coding_agent_with_retry(
    repository: Path, task_spec: Path, request_path: Path,
    coding_dir: Path, mode: str,
) -> Path:
    """Retry independent Coding Agent worktrees and keep a complete attempt log."""
    attempts = 1 if mode != "apply" else max(1, int(os.environ.get("OMNI_AR_CODING_MAX_ATTEMPTS", "3")))
    history: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        attempt_dir = coding_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, str(repository / "controller_bash/scripts/run_coding_agent.py"),
             "--task", str(task_spec), "--request", str(request_path),
             "--output-dir", str(attempt_dir), "--mode", mode],
            cwd=repository, text=True, capture_output=True, check=False,
        )
        (attempt_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
        (attempt_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")
        route_path = attempt_dir / "coding_route.json"
        route = _load(route_path) if route_path.is_file() else {
            "status": "failed", "error": "coding agent produced no route record",
        }
        history.append({
            "attempt": attempt, "returncode": proc.returncode,
            "status": route.get("status"), "route": str(route_path),
            "error": route.get("error"),
        })
        if route.get("status") == "ready":
            selected = route
            break
    aggregate = dict(selected or {
        "schema_version": "omni-ar-coding-route/v1", "status": "invalid",
        "mode": mode, "error": "all Coding Agent attempts failed closed",
    })
    aggregate["attempt_history"] = history
    aggregate["attempts_used"] = len(history)
    aggregate["max_attempts"] = attempts
    route_path = coding_dir / "coding_route.json"
    write_json(route_path, aggregate)
    return route_path


def _coding_control_suite(
    repository: Path, task_spec: Path, request: dict[str, Any],
    parent_parameters: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Build executable controls around a newly implemented method.

    The default-method run serves both as the task baseline and as the
    new-mechanism-disabled control. A distinct combination is included when
    inherited parent parameters compile in the generated worktree.
    """
    load_spec, _ = _controller_api(repository)
    spec = load_spec(task_spec)
    scripts = repository / "controller_bash/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from task_contract import load_task_adapter

    adapter = load_task_adapter(spec, task_spec)
    defaults = spec["adapter"]["trial_defaults"]
    enabled = copy.deepcopy(request["trial_proposal"])
    template = {
        "hypothesis": "Task baseline and new mechanism disabled control.",
        "change_scope": ["model"],
        "parameters": {"method": defaults.get("method")},
        "expected_effect": {}, "acceptance_criteria": {},
        "resource_request": copy.deepcopy(enabled.get("resource_request") or {}),
    }
    experiments = [template, enabled]
    roles: list[list[str]] = [["baseline", "disabled"], ["enabled"]]
    combined = copy.deepcopy(enabled)
    inherited = dict(parent_parameters or {})
    inherited.pop("method", None)
    combined["parameters"] = {**inherited, **(combined.get("parameters") or {})}
    if inherited and combined["parameters"] != enabled.get("parameters"):
        combined["hypothesis"] = (
            str(enabled.get("hypothesis") or "New mechanism")
            + " Combined with the selected parent's compatible settings."
        )
        try:
            adapter.proposal_to_trial(combined, defaults, "combination_control")
        except Exception:
            pass
        else:
            experiments.append(combined)
            roles.append(["combination"])
    wrapper = {
        "schema_version": "omni-ar-proposal/v2",
        "proposal_id": f"coding-{request['request_id']}",
        "task_name": spec["task"]["name"], "round": 0,
        "verdict": "Coding candidate with automatic controls.",
        "evidence_gaps": [], "experiment_proposals": experiments, "risks": [],
    }
    return wrapper, roles


def _small_smoke_proposal(
    repository: Path, task_spec: Path, wrapper: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a small disabled/enabled comparison using Adapter capabilities."""
    load_spec, _ = _controller_api(repository)
    spec = load_spec(task_spec)
    scripts = repository / "controller_bash/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from task_contract import load_task_adapter

    adapter = load_task_adapter(spec, task_spec)
    supported = set(adapter.trial_parameters or ())
    limits = {
        "max_train_samples": 1024, "max_eval_samples": 256,
        "max_steps": 8, "steps": 8, "epochs": 1,
    }
    experiments = copy.deepcopy(wrapper["experiment_proposals"][:2])
    changed = False
    for experiment in experiments:
        parameters = experiment.setdefault("parameters", {})
        for key, limit in limits.items():
            if key not in supported:
                continue
            current = parameters.get(key)
            parameters[key] = (
                min(current, limit)
                if isinstance(current, (int, float))
                and not isinstance(current, bool) and current > 0
                else limit
            )
            changed = True
        experiment["acceptance_criteria"] = {}
    if not changed:
        return None
    return {
        **wrapper, "proposal_id": f"{wrapper['proposal_id']}-smoke",
        "experiment_proposals": experiments,
    }


def _evaluate_coding_smoke(
    trace_path: Path, smoke: dict[str, Any], required_diagnostics: list[str],
) -> dict[str, Any]:
    """Reject broken, inactive, or clearly underperforming code early."""
    trace = _load(trace_path)
    jobs = list(trace.get("jobs") or [])
    technical = len(jobs) == 2 and all(
        (job.get("archive_gate") or {}).get("failure_class") is None
        for job in jobs
    )
    baseline_score = enabled_score = None
    direction = None
    if len(jobs) == 2:
        baseline_score, direction = _primary_from_standardized(jobs[0].get("standardized_result"))
        enabled_score, enabled_direction = _primary_from_standardized(jobs[1].get("standardized_result"))
        if enabled_direction != direction:
            technical = False
    max_drop = max(0.0, float(os.environ.get("OMNI_AR_CODING_SMOKE_MAX_PRIMARY_DROP", "0.08")))
    effect_passed = False
    delta = None
    if baseline_score is not None and enabled_score is not None and direction in {"maximize", "minimize"}:
        delta = enabled_score - baseline_score
        effect_passed = (
            enabled_score >= baseline_score - max_drop
            if direction == "maximize"
            else enabled_score <= baseline_score + max_drop
        )
    enabled_path = jobs[1].get("standardized_result") if len(jobs) == 2 else None
    activation = _activation_report(enabled_path, required_diagnostics)
    parameters = _executed_parameter_report(
        enabled_path, (smoke["experiment_proposals"][1].get("parameters") or {}),
    )
    passed = technical and effect_passed and activation["passed"] and parameters["passed"]
    return {
        "status": "passed" if passed else "failed",
        "trace": str(trace_path), "technical_passed": technical,
        "effect_passed": effect_passed, "baseline_score": baseline_score,
        "enabled_score": enabled_score, "direction": direction, "delta": delta,
        "max_primary_drop": max_drop, "activation_validation": activation,
        "parameter_validation": parameters,
    }


def _route_implementation_requests(
    repository: Path, task_spec: Path, proposal: dict[str, Any], round_dir: Path,
    execution_mode: str, coding_mode: str, parent_ids: list[str],
    strategy: StrategyBridge, *, max_requests: int | None = None,
    dataset_lineage: dict[str, Any] | None = None,
    code_parent_id: str | None = None,
    parent_parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    requests = list(proposal.get("implementation_requests") or [])
    if max_requests is not None:
        requests = requests[:max(0, max_requests)]
    for index, request in enumerate(requests):
        coding_dir = round_dir / f"coding_{index:03d}"
        request_path = coding_dir / "request.json"
        coding_dir.mkdir(parents=True, exist_ok=True)
        write_json(request_path, {"implementation_request": request})
        mode = "plan" if coding_mode == "off" else coding_mode
        route_path = _run_coding_agent_with_retry(
            repository, task_spec, request_path, coding_dir, mode,
        )
        route = _load(route_path) if route_path.is_file() else {"status": "failed"}
        candidate_id = f"round-{proposal['round']:03d}-coding-{index:03d}"
        if route.get("status") != "ready":
            metadata = strategy.on_result(
                candidate_id, None, idea=request["hypothesis"],
                parent_ids=parent_ids, round_number=int(proposal["round"]),
            )
            records.append({
                "candidate_id": candidate_id, "round": proposal["round"],
                "generation": metadata.get("generation", 0), "bucket": "invalid",
                "strategy_bucket": metadata.get("bucket"), "score": None,
                "direction": None, "scientific_evidence": False,
                "proposal_id": proposal["proposal_id"], "proposal_path": None,
                "trial_id": None, "hypothesis": request["hypothesis"],
                "route": "coding_agent", "parent_ids": parent_ids,
                "strategy_metadata": metadata, "coding_route": str(route_path),
                "implementation_fingerprint": _request_fingerprint(request),
                "failure_reason": route.get("error") or f"coding route status={route.get('status')}",
            })
            continue
        candidate_task = Path(route["candidate_task_spec"])
        candidate_repository = Path(route["candidate_worktree"])
        wrapper, control_roles = _coding_control_suite(
            candidate_repository, candidate_task, request, parent_parameters,
        )
        wrapper["round"] = proposal["round"]
        smoke = _small_smoke_proposal(candidate_repository, candidate_task, wrapper)
        smoke_record: dict[str, Any] = {"status": "not_applicable"}
        if smoke is not None and execution_mode not in {"simulated", "rjob_dry_run"}:
            smoke_path = coding_dir / "smoke_proposal.json"
            write_json(smoke_path, smoke)
            smoke_dir = candidate_repository / ".omni_runs" / f"{candidate_id}-smoke"
            smoke_trace = _execute(
                candidate_repository, candidate_task, smoke_path, smoke_dir,
                "local" if int((smoke["experiment_proposals"][0].get("resource_request") or {}).get("gpu_count", 0)) == 0 else execution_mode,
            )
            smoke_record = _evaluate_coding_smoke(
                smoke_trace, smoke, list(request["activation_diagnostics"]),
            )
            smoke_ok = smoke_record["status"] == "passed"
            write_json(coding_dir / "smoke_validation.json", smoke_record)
            if not smoke_ok:
                records.append({
                    "candidate_id": candidate_id, "round": proposal["round"],
                    "generation": 0, "bucket": "failed_train", "score": None,
                    "scientific_evidence": False, "proposal_id": wrapper["proposal_id"],
                    "hypothesis": request["hypothesis"], "route": "coding_agent",
                    "parent_ids": parent_ids, "coding_route": str(route_path),
                    "candidate_worktree": str(candidate_repository),
                    "implementation_fingerprint": _request_fingerprint(request),
                    "failure_reason": "small smoke failed; full training was not started",
                    "smoke_validation": smoke_record,
                })
                continue
        candidate_proposal = coding_dir / "candidate_proposal.json"
        write_json(candidate_proposal, wrapper)
        execution_dir = candidate_repository / ".omni_runs" / candidate_id
        trace = _execute(
            Path(route["candidate_worktree"]), candidate_task, candidate_proposal,
            execution_dir, execution_mode,
        )
        local_trace = coding_dir / "execution_trace.json"
        local_trace.write_bytes(trace.read_bytes())
        request_records = _candidate_records(
            wrapper, candidate_proposal, local_trace, int(proposal["round"]),
            parent_ids, strategy, route="coding_agent", code_patch=route.get("code_patch"),
            code_snapshot=route.get("source_snapshot"),
            changed_paths=list(route.get("changed_paths") or []),
            candidate_prefix=candidate_id,
            candidate_worktree=route.get("candidate_worktree"),
            required_activation_diagnostics=list(request["activation_diagnostics"]),
            dataset_lineage=dataset_lineage, code_parent_id=code_parent_id,
            control_roles=control_roles,
        )
        records.extend(request_records)
        executed_roles = {role for values in control_roles for role in values}
        controls_ok = all(
            item.get("bucket") == "control"
            for item in request_records
            if set(item.get("control_roles") or []).intersection({"baseline", "disabled"})
        )
        controls_ok = controls_ok and {"baseline", "disabled", "enabled"}.issubset(executed_roles)
        for item in request_records:
            item["implementation_fingerprint"] = _request_fingerprint(request)
            item["smoke_validation"] = smoke_record
            item["control_validation"] = {
                "required_roles": ["baseline", "disabled", "enabled"],
                "roles_executed": sorted(executed_roles),
                "combination_status": "executed" if "combination" in executed_roles else "no_compatible_parent_settings",
                "passed": controls_ok,
            }
            if item["bucket"] == "accepted" and not item["control_validation"]["passed"]:
                item["bucket"] = "invalid"
                item["score"] = None
                item["failure_reason"] = "automatic control suite incomplete"
                item["archive_gate"] = {
                    "eligible": False, "decision": "reject",
                    "reasons": ["automatic_control_suite_incomplete"],
                    "failure_class": "result_invalid", "retryable": False,
                    "archive_bucket": "invalid",
                }
    return records


def _run_stability_validation(
    repository: Path, task_spec: Path, records: ProjectRecords,
    strategy: StrategyBridge, winner: dict[str, Any], required_seeds: int,
    execution_mode: str, round_number: int,
) -> dict[str, Any]:
    """Re-run the selected candidate on distinct seeds and archive the mean."""
    archive = records.archive()
    run_repository, run_task, code_parent_id = _execution_context_for_parents(
        repository, task_spec, archive, [str(winner["candidate_id"])],
    )
    load_spec, _ = _controller_api(run_repository)
    spec = load_spec(run_task)
    base_seed = int((winner.get("parameters") or {}).get("seed", 42))
    experiments = []
    for offset in range(required_seeds):
        parameters = dict(winner.get("parameters") or {})
        parameters["seed"] = base_seed + offset
        experiments.append({
            "hypothesis": f"Stability rerun of {winner['candidate_id']} at seed {base_seed + offset}.",
            "change_scope": ["evaluation"], "parameters": parameters,
            "expected_effect": {}, "acceptance_criteria": {},
            "resource_request": dict(winner.get("resource_request") or {}),
        })
    proposal = {
        "schema_version": "omni-ar-proposal/v2",
        "proposal_id": f"{winner['candidate_id']}-stability",
        "task_name": spec["task"]["name"], "round": round_number,
        "verdict": "Automatic final multi-seed validation.",
        "evidence_gaps": [], "experiment_proposals": experiments, "risks": [],
    }
    stability_dir = records.project_dir / "stability_validation"
    proposal_path = stability_dir / "proposal.json"
    write_json(proposal_path, proposal)
    trace_path = _execute(
        run_repository, run_task, proposal_path, stability_dir / "execution", execution_mode,
    )
    trace = _load(trace_path)
    jobs = trace.get("jobs") or []
    scores: list[float] = []
    standardized_paths: list[str] = []
    required_diagnostics = list((winner.get("activation_validation") or {}).get("required") or [])
    valid = len(jobs) == required_seeds
    for job in jobs:
        path = job.get("standardized_result")
        score, _ = _primary_from_standardized(path)
        activation = _activation_report(path, required_diagnostics)
        valid = valid and (job.get("archive_gate") or {}).get("eligible") is True and score is not None and activation["passed"]
        if score is not None:
            scores.append(score)
        if path:
            standardized_paths.append(str(path))
    mean = sum(scores) / len(scores) if scores else None
    variance = sum((value - mean) ** 2 for value in scores) / len(scores) if scores and mean is not None else None
    std = math.sqrt(variance) if variance is not None else None
    candidate_id = f"{winner['candidate_id']}-stability-{required_seeds}s"
    aggregate_path = stability_dir / "standardized_result.json"
    if standardized_paths:
        aggregate = copy.deepcopy(_load(Path(standardized_paths[0])))
        for metric in (aggregate.get("metrics") or {}).values():
            if isinstance(metric, dict) and metric.get("role") == "primary":
                metric["value"] = mean
        aggregate.setdefault("protocol", {})["stability"] = {
            "status": "multi_seed", "num_seeds": len(scores),
            "mean": mean, "std": std, "min": min(scores) if scores else None,
            "max": max(scores) if scores else None, "values": scores,
        }
        usages = [
            (_load(Path(path)).get("resource_usage") or {}) for path in standardized_paths
        ]
        aggregate["resource_usage"] = {
            **(usages[0] if usages else {}),
            "runtime_seconds": sum(float(item.get("runtime_seconds", 0.0)) for item in usages),
            "within_budget": all(item.get("within_budget") is True for item in usages),
            "stability_runs": len(usages),
        }
        aggregate.setdefault("artifacts", {})["seed_results"] = standardized_paths
        aggregate["trial_id"] = candidate_id
        write_json(aggregate_path, aggregate)
    metadata = strategy.on_result(
        candidate_id, mean if valid else None,
        idea="Automatic multi-seed validation", parent_ids=[str(winner["candidate_id"])],
        round_number=round_number,
    )
    return {
        **winner,
        "candidate_id": candidate_id, "round": round_number,
        "generation": metadata.get("generation", int(winner.get("generation", 0)) + 1),
        "bucket": "accepted" if valid else "failed_train",
        "strategy_bucket": metadata.get("bucket"),
        "score": mean if valid else None, "observed_score": mean,
        "proposal_id": proposal["proposal_id"], "proposal_path": str(proposal_path),
        "trial_id": candidate_id, "parent_ids": [str(winner["candidate_id"])],
        "route": "stability_validation", "execution_trace": str(trace_path),
        "standardized_result": str(aggregate_path) if aggregate_path.is_file() else None,
        "stability": {
            "status": "multi_seed", "num_seeds": len(scores),
            "mean": mean, "std": std, "min": min(scores) if scores else None,
            "max": max(scores) if scores else None, "values": scores,
        },
        "seed_count": len(scores), "scientific_evidence": trace.get("scientific_evidence") is not False,
        "code_parent_id": code_parent_id,
        "failure_reason": None if valid else "one or more multi-seed runs failed acceptance or activation checks",
    }


def run_autoresearch(
    repository: Path, rough_path: Path, project_dir: Path, *, rounds: int = 3,
    strategy_name: str = "omni_epic", planning_mode: str = "live",
    execution_mode: str = "rjob_dry_run", coding_mode: str = "plan",
    target_accepted: int = 0, patience: int = 0, resume: bool = False,
    min_rounds: int = 1, min_improvement: float = 0.0,
    required_seeds: int = 1, max_coding_requests_per_round: int = 1,
) -> Path:
    repository, rough_path, project_dir = repository.resolve(), rough_path.resolve(), project_dir.resolve()
    rough = _load(rough_path)
    RoughIdeaEngine(repository).validate(rough)
    if rough.get("confirmation", {}).get("status") != "confirmed":
        raise AutoResearchError("auto-run requires a confirmed rough idea")
    relative_task = rough.get("task", {}).get("task_spec")
    if not relative_task:
        raise AutoResearchError(
            "ResearchStudio can plan this open task, but multi-round execution requires a registered Task Adapter"
        )
    task_spec = (repository / relative_task).resolve()
    load_spec, validate_proposal = _controller_api(repository)
    spec = load_spec(task_spec)
    direction = spec["metrics"]["primary"]["direction"]
    records = ProjectRecords(project_dir)
    if records.project_path.exists() and not resume:
        raise AutoResearchError(
            f"research project already exists at {project_dir}; use run_research resume"
        )
    policy = rough.get("execution_policy") or {}
    if policy and rounds > int(policy.get("max_rounds", rounds)):
        raise AutoResearchError("requested rounds exceed the QA-confirmed execution policy")
    attempts = int(policy.get("max_attempts", 1)) if policy.get("auto_retry") else 1
    records.initialize({
        "repository": str(repository), "rough_idea": str(rough_path), "task_spec": str(task_spec),
        "task": spec["task"]["name"], "strategy": strategy_name,
        "planning_mode": planning_mode, "execution_mode": execution_mode,
        "coding_mode": coding_mode, "round_budget": rounds,
        "target_accepted": target_accepted, "patience": patience,
        "min_rounds": min_rounds, "min_improvement": min_improvement,
        "required_seeds": required_seeds,
        "max_coding_requests_per_round": max_coding_requests_per_round,
        "planning_model": os.environ.get("OMNI_AR_BOYUE_MODEL") or None,
        "qa_constraint_sources": (rough.get("provenance") or {}).get("constraint_sources") or {},
        "qa_transcript": str(rough_path.parent / "qa_transcript.jsonl")
        if (rough_path.parent / "qa_transcript.jsonl").is_file() else None,
    })
    records.stage("preflight", "running")
    try:
        run_preflight(
            repository, rough, task_spec, records.root / "preflight.json",
            planning_mode=planning_mode, execution_mode=execution_mode,
            coding_mode=coding_mode, project_dir=project_dir,
        )
    except Exception as exc:
        records.stage("preflight", "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    records.stage("preflight", "completed")
    reused = _reuse_planning_artifacts(project_dir, planning_mode) if resume else None
    if reused:
        initial_proposal, idea_card, base_context = reused
        records.stage("planning", "completed", reused=True)
    else:
        initial_proposal, idea_card, base_context = _call_with_retry(
            records, "planning", attempts,
            lambda: _planning_artifacts(repository, rough_path, project_dir, planning_mode),
        )
    strategy = StrategyBridge(repository, strategy_name, direction)
    moi_enabled = planning_mode == "live" and strategy_name == "omni_epic"
    if moi_enabled:
        strategy.configure_moi(repository, project_dir, spec, rough, idea_card)
    existing = records.archive() if resume else []
    _restore_strategy(strategy, existing)
    accepted_entries = [
        item for item in existing
        if item.get("bucket") == "accepted" and item.get("scientific_evidence") is True
    ]
    accepted_count = len(accepted_entries)
    accepted_scores = [float(item["score"]) for item in accepted_entries if isinstance(item.get("score"), (int, float))]
    best_score = (
        (max(accepted_scores) if direction == "maximize" else min(accepted_scores))
        if accepted_scores else None
    )
    stagnant = 0
    reason = "round_budget_exhausted"
    completed = records.completed_rounds() if resume else 0
    baseline_value = ((rough.get("evaluation") or {}).get("baseline") or {}).get("value")
    fatal_error = None
    try:
        for round_number in range(completed, rounds):
            records.stage(f"round_{round_number:03d}", "running")
            round_dir = project_dir / "rounds" / f"round_{round_number:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            parent_ids, search_context = strategy.select(round_number)
            archive_before_round = records.archive()
            round_repository, round_task_spec, code_parent_id = _execution_context_for_parents(
                repository, task_spec, archive_before_round, parent_ids,
            )
            round_spec = load_spec(round_task_spec)
            parent_entry = next(
                (item for item in archive_before_round if item.get("candidate_id") in parent_ids),
                {},
            )
            if round_number == 0:
                proposal_path = initial_proposal
            elif planning_mode == "live":
                if base_context is None:
                    raise AutoResearchError("live planning context is missing")
                proposal_path = _call_with_retry(
                    records, f"round_{round_number:03d}_proposal", attempts,
                    lambda: _next_live_proposal(
                        repository, base_context, round_dir, round_number,
                        records, parent_ids, search_context,
                        active_task_spec=round_task_spec,
                    ),
                )
            else:
                proposal_path = _next_offline_proposal(initial_proposal, round_dir, round_number)
            proposal = _load(proposal_path)
            validate_proposal(proposal, round_spec, round_task_spec)
            _validate_execution_policy(proposal, policy)
            if moi_enabled:
                idea_text = _proposal_text(proposal)
                try:
                    assessment = strategy.review(idea_text)
                except Exception as exc:  # fail closed: no review, no execution
                    assessment = None
                    moi_error = f"{type(exc).__name__}: {exc}"
                else:
                    moi_error = None
                if moi_error or (assessment is not None and not assessment.interesting):
                    candidate_id = f"round-{round_number:03d}-moi"
                    if moi_error:
                        generation = 0
                        if parent_ids:
                            generations = {
                                item["candidate_id"]: int(item.get("generation", 0))
                                for item in records.archive()
                            }
                            generation = max((generations.get(pid, 0) for pid in parent_ids), default=0) + 1
                        metadata = {"generation": generation, "bucket": "invalid"}
                        bucket, reason_text = "invalid", moi_error
                    else:
                        metadata = strategy.reject(candidate_id, idea_text, parent_ids)
                        bucket, reason_text = "failed_moi", assessment.reasoning
                    candidate = {
                        "candidate_id": candidate_id, "round": round_number,
                        "generation": metadata.get("generation", 0), "bucket": bucket,
                        "strategy_bucket": metadata.get("bucket"), "score": None,
                        "direction": direction, "scientific_evidence": False,
                        "proposal_id": proposal.get("proposal_id"),
                        "proposal_path": str(proposal_path), "trial_id": None,
                        "hypothesis": proposal.get("verdict"), "route": "moi_gate",
                        "parent_ids": parent_ids, "strategy_metadata": metadata,
                        "failure_reason": reason_text,
                    }
                    records.add_candidate(candidate)
                    records.record_round(round_number, {
                        "schema_version": "omni-ar-round-record/v1", "status": "completed",
                        "round": round_number, "proposal": str(proposal_path),
                        "parent_ids": parent_ids, "candidate_ids": [candidate_id],
                        "accepted_this_round": 0, "best_score": best_score,
                        "execution_skipped": True, "skip_reason": bucket,
                    })
                    completed += 1
                    records.stage(f"round_{round_number:03d}", "completed")
                    stagnant += 1
                    if patience and stagnant >= patience:
                        reason = "no_improvement_patience"
                        break
                    continue
            execution_dir = round_dir / "adapter_execution"
            direct_experiments, unresolved_requests = _proposal_execution_partition(
                round_repository, round_spec, round_task_spec, proposal,
            )
            failed_fingerprints = {
                str(item.get("implementation_fingerprint"))
                for item in archive_before_round
                if item.get("bucket") in {"failed_train", "invalid", "rejected"}
                and item.get("implementation_fingerprint")
            }
            unresolved_requests = [
                item for item in unresolved_requests
                if _request_fingerprint(item) not in failed_fingerprints
            ]
            trial_limit = int(policy.get("max_trials_per_round", 1_000_000))
            implementation_count = min(
                len(unresolved_requests), trial_limit, max_coding_requests_per_round,
            )
            direct_limit = max(0, trial_limit - implementation_count)
            write_json(round_dir / "execution_budget_plan.json", {
                "trial_limit": trial_limit,
                "priority": "implementation_requests_first",
                "selected_implementation_requests": implementation_count,
                "direct_experiment_limit": direct_limit,
                "deferred_implementation_requests": max(
                    0, len(unresolved_requests) - implementation_count,
                ),
            })
            selected_direct = direct_experiments[:direct_limit]
            direct_path = None
            if selected_direct:
                direct_path = round_dir / "direct_adapter_proposal.json"
                write_json(direct_path, {
                    **proposal, "experiment_proposals": selected_direct,
                    "implementation_requests": [],
                })
            candidates = []
            if direct_path is not None:
                trace = _call_with_retry(
                    records, f"round_{round_number:03d}_execution", attempts,
                    lambda: _execute_with_transient_check(
                        round_repository, round_task_spec, direct_path, execution_dir, execution_mode,
                    ),
                )
                candidates.extend(_candidate_records(
                    _load(direct_path), direct_path, trace, round_number, parent_ids, strategy,
                    code_patch=parent_entry.get("code_patch"),
                    code_snapshot=parent_entry.get("code_snapshot"),
                    changed_paths=list(parent_entry.get("changed_paths") or []),
                    candidate_worktree=(
                        str(round_repository) if round_repository != repository else None
                    ),
                    dataset_lineage=_dataset_lineage(round_spec),
                    code_parent_id=code_parent_id,
                    required_activation_diagnostics=list(
                        (parent_entry.get("activation_validation") or {}).get("required") or []
                    ),
                    activation_diagnostics_by_experiment=_adapter_activation_requirements(
                        round_repository, round_spec, round_task_spec, selected_direct,
                    ),
                    inherited_control_validation=parent_entry.get("control_validation"),
                ))
            route_proposal = {**proposal, "implementation_requests": unresolved_requests}
            candidates.extend(_route_implementation_requests(
                round_repository, round_task_spec, route_proposal, round_dir, execution_mode,
                coding_mode, parent_ids, strategy, max_requests=implementation_count,
                dataset_lineage=_dataset_lineage(round_spec),
                code_parent_id=code_parent_id,
                parent_parameters=parent_entry.get("parameters") or {},
            ))
            if not candidates:
                candidate_id = f"round-{round_number:03d}-failed"
                metadata = strategy.on_result(
                    candidate_id, None, idea=_proposal_text(proposal),
                    parent_ids=parent_ids, round_number=round_number,
                )
                candidates.append({
                    "candidate_id": candidate_id, "round": round_number,
                    "generation": metadata.get("generation", 0),
                    "bucket": "failed_train", "strategy_bucket": metadata.get("bucket"),
                    "score": None, "direction": direction, "scientific_evidence": False,
                    "proposal_id": proposal.get("proposal_id"),
                    "proposal_path": str(proposal_path), "trial_id": None,
                    "hypothesis": proposal.get("verdict"), "route": "adapter",
                    "parent_ids": parent_ids, "strategy_metadata": metadata,
                    "execution_trace": str(locals().get("trace")) if locals().get("trace") else None,
                    "failure_reason": "execution trace contained no trial jobs",
                })
            for candidate in candidates:
                records.add_candidate(candidate)
            real_accepted = [item for item in candidates if item["bucket"] == "accepted"]
            accepted_count += len(real_accepted)
            round_scores = [float(item["score"]) for item in real_accepted]
            improved = False
            if round_scores:
                round_best = max(round_scores) if direction == "maximize" else min(round_scores)
                improved = best_score is None or (
                    round_best > best_score if direction == "maximize" else round_best < best_score
                )
                if improved:
                    best_score = round_best
            stagnant = 0 if improved else stagnant + 1
            records.record_round(round_number, {
                "schema_version": "omni-ar-round-record/v1", "status": "completed",
                "round": round_number, "proposal": str(proposal_path),
                "parent_ids": parent_ids, "search_context": search_context,
                "candidate_ids": [item["candidate_id"] for item in candidates],
                "accepted_this_round": len(real_accepted), "best_score": best_score,
                "execution_repository": str(round_repository),
                "code_parent_id": code_parent_id,
            })
            completed += 1
            records.stage(f"round_{round_number:03d}", "completed")
            winner_now = choose_winner(records.archive())
            improvement_ok = True
            if baseline_value is not None and winner_now is not None:
                improvement = (
                    float(winner_now["score"]) - float(baseline_value)
                    if direction == "maximize"
                    else float(baseline_value) - float(winner_now["score"])
                )
                improvement_ok = improvement >= min_improvement
            seeds_ok = bool(winner_now) and int(winner_now.get("seed_count", 1)) >= required_seeds
            if (
                target_accepted and accepted_count >= target_accepted
                and completed >= min_rounds and improvement_ok and seeds_ok
            ):
                reason = "target_accepted_reached"
                break
            if patience and stagnant >= patience and completed >= min_rounds:
                reason = "no_improvement_patience"
                break
    except Exception as exc:  # noqa: BLE001 - persist a fail-closed terminal record
        fatal_error = f"{type(exc).__name__}: {exc}"
        reason = "fatal_round_failure"
        failed_round = locals().get("round_number", completed)
        records.record_round(int(failed_round), {
            "schema_version": "omni-ar-round-record/v1", "status": "failed",
            "round": int(failed_round), "error": fatal_error,
        })
        records.stage(f"round_{int(failed_round):03d}", "failed", error=fatal_error)
    if fatal_error is None and required_seeds > 1 and execution_mode in {"local", "rjob"}:
        single_seed_winner = choose_winner(records.archive())
        if single_seed_winner and int(single_seed_winner.get("seed_count", 1)) < required_seeds:
            records.stage("stability_validation", "running")
            try:
                stable_candidate = _run_stability_validation(
                    repository, task_spec, records, strategy, single_seed_winner,
                    required_seeds, execution_mode, completed,
                )
                records.add_candidate(stable_candidate)
                if stable_candidate.get("bucket") == "accepted":
                    accepted_count += 1
                    best_score = float(stable_candidate["score"])
                    reason = "multi_seed_validation_completed"
                records.stage("stability_validation", "completed")
            except Exception as exc:  # fail closed at the final evidence gate
                fatal_error = f"{type(exc).__name__}: {exc}"
                reason = "stability_validation_failed"
                records.stage("stability_validation", "failed", error=fatal_error)
    termination = {
        "reason": reason, "rounds_completed": completed,
        "accepted_candidates": accepted_count, "best_score": best_score,
        "round_budget": rounds, "target_accepted": target_accepted,
        "patience": patience,
        "min_rounds": min_rounds, "min_improvement": min_improvement,
        "required_seeds": required_seeds,
        "baseline_value": baseline_value,
        "error": fatal_error,
    }
    records.stage("finalizer", "running")
    final_package = build_final_package(
        repository, project_dir, records, rough_idea=rough_path,
        task_spec=task_spec, idea_card=idea_card, termination=termination,
    )
    records.stage("finalizer", "completed")
    records.finalize(termination, final_package)
    refresh_records_artifact(final_package, records)
    summary = {
        "status": "failed" if fatal_error else "ok", "project_dir": str(project_dir),
        "records": str(records.root), "termination": termination,
        "winner": choose_winner(
            records.archive(), required_seeds=required_seeds,
            baseline_value=float(baseline_value) if baseline_value is not None else None,
            min_improvement=min_improvement,
        ),
        "final_package": str(final_package),
        "qa_constraint_sources": (rough.get("provenance") or {}).get("constraint_sources") or {},
        "qa_transcript": str(rough_path.parent / "qa_transcript.jsonl")
        if (rough_path.parent / "qa_transcript.jsonl").is_file() else None,
    }
    write_json(project_dir / "autoresearch_result.json", summary)
    return project_dir / "autoresearch_result.json"


def resume_autoresearch(project_dir: Path) -> Path:
    project_dir = project_dir.resolve()
    project_path = project_dir / "records/project.json"
    if not project_path.is_file():
        raise AutoResearchError(f"project record not found: {project_path}")
    metadata = _load(project_path).get("metadata") or {}
    required = {"rough_idea", "strategy", "planning_mode", "execution_mode", "coding_mode", "round_budget"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise AutoResearchError(f"project record cannot resume; missing metadata: {missing}")
    planning_model = metadata.get("planning_model")
    previous_model = os.environ.get("OMNI_AR_BOYUE_MODEL")
    if planning_model:
        os.environ["OMNI_AR_BOYUE_MODEL"] = str(planning_model)
    try:
        return run_autoresearch(
            Path(metadata.get("repository") or Path(__file__).resolve().parents[1]),
            Path(metadata["rough_idea"]), project_dir,
            rounds=int(metadata["round_budget"]), strategy_name=str(metadata["strategy"]),
            planning_mode=str(metadata["planning_mode"]), execution_mode=str(metadata["execution_mode"]),
            coding_mode=str(metadata["coding_mode"]), target_accepted=int(metadata.get("target_accepted", 0)),
            patience=int(metadata.get("patience", 0)), resume=True,
            min_rounds=int(metadata.get("min_rounds", 1)),
            min_improvement=float(metadata.get("min_improvement", 0.0)),
            required_seeds=int(metadata.get("required_seeds", 1)),
            max_coding_requests_per_round=int(metadata.get("max_coding_requests_per_round", 1)),
        )
    finally:
        if planning_model:
            if previous_model is None:
                os.environ.pop("OMNI_AR_BOYUE_MODEL", None)
            else:
                os.environ["OMNI_AR_BOYUE_MODEL"] = previous_model


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "resume":
        resume_parser = argparse.ArgumentParser(prog="run_research resume")
        resume_parser.add_argument("resume")
        resume_parser.add_argument("--project-dir", type=Path, required=True)
        resume_args = resume_parser.parse_args(argv)
        try:
            result = resume_autoresearch(resume_args.project_dir)
            print(result)
            return 0 if _load(result).get("status") == "ok" else 2
        except (AutoResearchError, PreflightError, InitializationError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
            return 2
    parser = argparse.ArgumentParser(prog="run_research auto-run")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rough-idea", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--strategy", choices=("omni_epic", "islands"), default="omni_epic")
    parser.add_argument("--planning-mode", choices=("live", "dry_run"), default="live")
    parser.add_argument("--execution-mode", choices=("simulated", "local", "rjob_dry_run", "rjob"), default="rjob_dry_run")
    parser.add_argument("--coding-mode", choices=("off", "plan", "apply"), default="plan")
    parser.add_argument("--target-accepted", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-rounds", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--required-seeds", type=int, default=1)
    parser.add_argument("--max-coding-requests-per-round", type=int, default=1)
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if not 1 <= args.min_rounds <= args.rounds:
        parser.error("--min-rounds must be between 1 and --rounds")
    if args.required_seeds < 1 or args.max_coding_requests_per_round < 0:
        parser.error("seed count must be positive and coding request cap cannot be negative")
    try:
        result = run_autoresearch(
            args.repository, args.rough_idea, args.output_dir,
            rounds=args.rounds, strategy_name=args.strategy,
            planning_mode=args.planning_mode, execution_mode=args.execution_mode,
            coding_mode=args.coding_mode, target_accepted=args.target_accepted,
            patience=args.patience,
            min_rounds=args.min_rounds, min_improvement=args.min_improvement,
            required_seeds=args.required_seeds,
            max_coding_requests_per_round=args.max_coding_requests_per_round,
        )
        print(result)
        return 0 if _load(result).get("status") == "ok" else 2
    except (AutoResearchError, PreflightError, InitializationError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
