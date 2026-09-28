#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from task_contract import ContractError, load_task_spec, validate_proposal, validate_schema


REQUIRED_V1 = {
    "schema_version",
    "proposal_id",
    "task_name",
    "round",
    "verdict",
    "evidence_gaps",
    "idea_variants",
    "ablation_plan",
    "sweep_trials",
    "codex_tasks",
}

REQUIRED_V2 = {
    "schema_version",
    "proposal_id",
    "task_name",
    "round",
    "verdict",
    "evidence_gaps",
    "experiment_proposals",
    "risks",
}


def resolve_allowed_path(path: str, root_dir: Path, allowed_root: Path | None) -> Path:
    candidate = Path(path)
    if allowed_root is not None and not candidate.is_absolute():
        candidate = allowed_root / candidate
    elif not candidate.is_absolute():
        candidate = root_dir / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate.absolute()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suggestion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    obj = json.loads(args.suggestion.read_text(encoding="utf-8"))
    errors: list[str] = []
    version = obj.get("schema_version")
    suggestion_schema = os.environ.get("SUGGESTION_SCHEMA")
    if suggestion_schema:
        try:
            schema_path = Path(suggestion_schema).resolve()
            if version == "omni-ar-proposal/v2":
                schema_path = schema_path.with_name("proposal_v2.schema.json")
            validate_schema(obj, schema_path)
        except ContractError as exc:
            errors.append(str(exc))
    task_spec_path = os.environ.get("TASK_SPEC")
    if task_spec_path:
        try:
            spec_path = Path(task_spec_path).resolve()
            spec = load_task_spec(spec_path)
            validate_proposal(obj, spec, spec_path)
        except ContractError as exc:
            errors.append(str(exc))
    required = REQUIRED_V2 if version == "omni-ar-proposal/v2" else REQUIRED_V1
    missing = sorted(required - set(obj))
    if missing:
        errors.append(f"missing required keys: {missing}")

    list_keys = (["evidence_gaps", "experiment_proposals", "risks"]
                 if version == "omni-ar-proposal/v2"
                 else ["evidence_gaps", "idea_variants", "ablation_plan", "sweep_trials", "codex_tasks"])
    for key in list_keys:
        if key in obj and not isinstance(obj[key], list):
            errors.append(f"{key} must be a list")

    max_trials = int(os.environ.get("MAX_TRIALS_PER_ROUND", "12"))
    trials = obj.get("experiment_proposals", []) if version == "omni-ar-proposal/v2" else obj.get("sweep_trials", [])
    if isinstance(trials, list) and len(trials) > max_trials:
        errors.append(f"sweep_trials has {len(trials)} entries, limit is {max_trials}")

    root_dir = Path(os.environ.get("ROOT_DIR", ".")).resolve()
    allowed_root_raw = os.environ.get("CODEX_ALLOWED_DIR", "")
    allowed_root = Path(allowed_root_raw).resolve() if allowed_root_raw else None
    extra_allowed_roots = []
    for raw in [
        os.environ.get("PROJECT_ROOT", ""),
        os.environ.get("ABLATION_DIR", ""),
        os.environ.get("ARTIFACT_DIR", ""),
        os.environ.get("STATE_DIR", ""),
        os.environ.get("REPORT_DIR", ""),
    ]:
        if raw:
            extra_allowed_roots.append(Path(raw).resolve())
    forbidden_terms = [".env", "BOYUE_API_KEY", "OPENAI_API_KEY", "rm -rf", "git reset --hard"]
    for task in obj.get("codex_tasks", []) if isinstance(obj.get("codex_tasks"), list) else []:
        text = json.dumps(task, ensure_ascii=False)
        for term in forbidden_terms:
            if term in text:
                errors.append(f"codex task mentions forbidden term: {term}")
        for path in task.get("allowed_paths", []) if isinstance(task, dict) else []:
            resolved = resolve_allowed_path(str(path), root_dir, allowed_root)
            in_write_root = allowed_root and (resolved == allowed_root or allowed_root in resolved.parents)
            in_read_root = any(resolved == root or root in resolved.parents for root in extra_allowed_roots)
            if allowed_root and not in_write_root and not in_read_root:
                errors.append(f"codex task allowed path is outside controller policy: {path}")

    for idx, trial in enumerate(trials if isinstance(trials, list) else []):
        if not isinstance(trial, dict):
            errors.append(f"sweep_trials[{idx}] must be an object")
            continue
        if int(trial.get("train_steps", 0) or 0) > 5000:
            errors.append(f"sweep_trials[{idx}] train_steps exceeds 5000")
        if version != "omni-ar-proposal/v2" and os.environ.get("REQUIRE_ADAPTER_TRIAL_ENTRYPOINT", "0") == "1":
            entrypoint = trial.get("entrypoint")
            if "command" in trial or entrypoint not in {"run_baseline", "run_trial"}:
                errors.append(
                    f"sweep_trials[{idx}] must use adapter entrypoint "
                    "run_trial or run_baseline; custom commands are disabled"
                )

    report = {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "suggestion": str(args.suggestion),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
