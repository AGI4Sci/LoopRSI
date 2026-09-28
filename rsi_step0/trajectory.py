"""RSI Step0 trajectory transcription, annotation and replay (W1 / M7).

Reads heuresis run artifacts and emits ``rsi_trajectory/v1`` records, then
labels, dedupes, splits into SFT/RL/scheduler datasets and supports a
deterministic replay hash. Pure stdlib.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import rsi_step0.contracts as C
import rsi_step0.reward as R

REQUIRED_FILES = [
    "proposal.json",
    "trial_results.json",
    "round_summary.json",
    "execution_budget_plan.json",
    "validation.json",
]


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_events(run_dir: str) -> List[dict]:
    events_path = os.path.join(run_dir, "events.jsonl")
    events: List[dict] = []
    if not os.path.exists(events_path):
        return events
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def _parse_operator_from_report(report_text: str) -> str:
    """Heuristically infer the operator from a round suggestion report."""
    lowered = report_text.lower()
    for op in ("draft", "improve", "crossover", "debug", "tune", "evaluate", "implement", "question"):
        if op in lowered:
            return op
    return "draft"


def _build_record(
    run_id: str,
    round_: int,
    step: int,
    operator: str,
    proposal: dict,
    outcome_metrics: dict,
    resource_usage: dict,
    provenance: dict,
    raw_decision: dict,
    mse_baseline: Optional[float] = None,
) -> dict:
    task_id = proposal.get("task_id", "")
    decision = raw_decision or {}
    action = {
        "action_id": C.make_record_id(run_id, round_, step, operator),
        "run_id": run_id,
        "round": round_,
        "step": step,
        "operator": operator,
        "mode": "execution",
        "target": task_id,
        "decision": decision,
        "budget": {"max_seconds": 3600, "max_gpu_seconds": 3600, "max_cost": 1.0},
        "provenance": provenance,
    }
    pearson = float(outcome_metrics.get("pearson_delta", 0.0) or 0.0)
    status = str(outcome_metrics.get("status", "ok"))
    within_budget = bool(outcome_metrics.get("within_budget", True))
    env_failure = bool(outcome_metrics.get("env_failure", False))
    # MSE-minimize semantics: when a run reports an MSE primary metric and a
    # reference baseline is available, derive the normalized outcome from MSE
    # instead of the (vcc25/pearson semantics) delta field.
    mse = outcome_metrics.get("mse")
    normalized_override: Optional[float] = None
    if mse is not None and mse_baseline is not None:
        normalized_override = R.normalize_mse_outcome(float(mse), float(mse_baseline))

    reward, label = R.apply_reward_pipeline(
        pearson_delta=pearson,
        resource_usage=resource_usage,
        status=status,
        operator=operator,
        parent_bases=None,
        within_budget=within_budget,
        environmental_failure=env_failure,
        normalized_outcome=normalized_override,
    )

    variant = None
    iv = proposal.get("idea_variant")
    if isinstance(iv, dict):
        variant = iv.get("name")
    elif isinstance(iv, str):
        variant = iv

    return {
        "schema_version": C.SCHEMA_VERSION,
        "record_id": C.make_record_id(run_id, round_, step, operator),
        "run_id": run_id,
        "round": round_,
        "step": step,
        "operator": operator,
        "mode": "execution",
        "target": task_id,
        "decision": decision,
        "context": {
            "run_id": run_id,
            "round": round_,
            "task_id": task_id,
            "proposal_id": proposal.get("proposal_id"),
            "idea_variant": iv,
            "history": [],
            "last_outcome": {"status": status, "metrics": outcome_metrics},
            "available_actions": [],
            "resource_state": resource_usage,
        },
        "action": action,
        "outcome": {
            "status": status,
            "metrics": outcome_metrics,
            "resource_usage": resource_usage,
            "cost_aware_reward": reward["total"],
            "error": None,
        },
        "reward": reward,
        "label": label,
        "dedupe_key": C.make_dedupe_key(
            run_id, round_, proposal.get("proposal_id"), variant
        ),
        "provenance": provenance,
        "created_at": _now(),
    }


def transcribe(run_dir: str, out_dir: str) -> dict:
    """Transcribe heuresis run artifacts into rsi_trajectory/v1 records.

    Writes ``out_dir/records.jsonl`` and returns a summary dict. Missing files
    are skipped (counted in ``skipped_files``).
    """
    os.makedirs(out_dir, exist_ok=True)
    proposal = _read_json(os.path.join(run_dir, "proposal.json")) or {}
    trial = _read_json(os.path.join(run_dir, "trial_results.json")) or {}
    summary = _read_json(os.path.join(run_dir, "round_summary.json")) or {}
    budget_plan = _read_json(os.path.join(run_dir, "execution_budget_plan.json")) or {}
    validation = _read_json(os.path.join(run_dir, "validation.json")) or {}

    skipped = 0
    for fn in REQUIRED_FILES:
        if not os.path.exists(os.path.join(run_dir, fn)):
            skipped += 1

    run_id = (
        proposal.get("run_id")
        or trial.get("run_id")
        or summary.get("run_id")
        or "unknown"
    )
    round_ = int(
        proposal.get("round")
        or trial.get("round")
        or summary.get("round")
        or 0
    )
    step = int(proposal.get("step") or trial.get("step") or 1)

    metrics = dict(summary)
    metrics.setdefault("pearson_delta", trial.get("pearson_delta", 0.0))
    metrics.setdefault("status", summary.get("status", trial.get("status", "ok")))
    metrics.setdefault("within_budget", summary.get("within_budget", True))
    metrics.setdefault("env_failure", summary.get("env_failure", False))

    resource_usage = {
        "runtime_seconds": trial.get("runtime_seconds", summary.get("runtime_seconds", 0.0)),
        "gpu_count": trial.get("gpu_count", summary.get("gpu_count", 1)),
    }

    # Report-based operator inference (best effort).
    operator = "draft"
    report_dir = os.path.join(run_dir, "reports")
    if os.path.isdir(report_dir):
        for fn in sorted(os.listdir(report_dir)):
            if fn.startswith("round_") and fn.endswith(".md"):
                with open(os.path.join(report_dir, fn), "r", encoding="utf-8") as f:
                    operator = _parse_operator_from_report(f.read())
                break

    provenance = {
        "parent_action_id": None,
        "model": "deterministic",
        "seed": proposal.get("seed", 0),
    }

    decision = {k: proposal.get(k) for k in (
        "hypothesis", "change_scope", "expected_effect", "acceptance_criteria"
    ) if proposal.get(k) is not None}

    # a run that reports an MSE primary metric (tabular/minimize tasks) uses the
    # round reference MSE as the improvement baseline, so improving trials enter
    # the positive set instead of being excluded on a pearson-semantics delta.
    mse_baseline = None
    raw_baseline = (
        summary.get("mse")
        if isinstance(summary, dict) and summary.get("mse") is not None
        else (trial.get("mse") if isinstance(trial, dict) else None)
    )
    if raw_baseline is not None:
        mse_baseline = float(raw_baseline)

    record = _build_record(
        run_id, round_, step, operator, proposal, metrics,
        resource_usage, provenance, decision, mse_baseline=mse_baseline,
    )

    # Also derive records from candidate_recorded events in events.jsonl if present.
    candidates = [e for e in _iter_events(run_dir) if e.get("event") == "candidate_recorded"]
    records = [record]
    for idx, ev in enumerate(candidates, start=1):
        ev_decision = {
            "proposal_id": ev.get("proposal_id"),
            "idea_variant": ev.get("idea_variant"),
        }
        # Real engine rounds may label each candidate with its own operator
        # (draft/improve/crossover/debug). Fall back to the report-inferred
        # round-level operator when the event does not carry one.
        ev_operator = str(ev.get("operator") or operator)
        # Each event may carry its own proposal id / variant. Derive a
        # per-event proposal view so dedupe keys and context identity are
        # distinct across candidates instead of colliding with the round one.
        ev_proposal = dict(proposal)
        ev_proposal["proposal_id"] = str(
            ev.get("proposal_id") or proposal.get("proposal_id") or f"candidate_{idx}"
        )
        ev_iv = ev.get("idea_variant")
        if isinstance(ev_iv, dict):
            ev_proposal["idea_variant"] = ev_iv
        elif isinstance(ev_iv, str):
            ev_proposal["idea_variant"] = {"name": ev_iv}
        else:
            ev_proposal["idea_variant"] = {"name": ev_operator}
        ev_metrics = {
            "pearson_delta": ev.get("pearson_delta", 0.0),
            "status": ev.get("status", "ok"),
            "mse": ev.get("mse"),
        }
        ev_resource = {
            "runtime_seconds": ev.get("runtime_seconds", 0.0),
            "gpu_count": ev.get("gpu_count", 1),
        }
        rec = _build_record(
            run_id, round_, step + idx, ev_operator, ev_proposal,
            ev_metrics, ev_resource, provenance, ev_decision,
            mse_baseline=mse_baseline,
        )
        records.append(rec)

    records = annotate_and_dedupe(records)

    out_path = os.path.join(out_dir, "records.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = {"positive": 0, "negative": 0, "excluded": 0}
    for rec in records:
        counts[rec["label"]] = counts.get(rec["label"], 0) + 1

    return {
        "run_id": run_id,
        "records": len(records),
        "positive": counts["positive"],
        "negative": counts["negative"],
        "excluded": counts["excluded"],
        "deduped": len(records),
        "skipped_files": skipped,
        "output": out_path,
    }


def annotate_and_dedupe(records: List[dict]) -> List[dict]:
    """Dedupe by ``dedupe_key`` (keep first) and return the stable list."""
    seen = set()
    out = []
    for rec in records:
        key = rec.get("dedupe_key")
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def build_datasets(traj_dir: str, out_dir: str) -> dict:
    """Split labelled records into SFT / RL / scheduler dataset files."""
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(traj_dir, "records.jsonl")
    all_records: List[dict] = []
    if os.path.exists(records_path):
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        continue

    sft = [
        {"task_id": r.get("target"), "context": r.get("context"), "action": r.get("action")}
        for r in all_records if r.get("label") == "positive"
    ]
    rl = [
        {"task_id": r.get("target"), "context": r.get("context"),
         "action": r.get("action"), "outcome": r.get("outcome"),
         "reward": r.get("reward")}
        for r in all_records
    ]
    sched = [
        {"task_id": r.get("target"), "context": r.get("context"),
         "action": r.get("action"), "label": r.get("label")}
        for r in all_records
    ]

    def _write(name: str, rows: List[dict]) -> str:
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return p

    sft_path = _write("sft_records.jsonl", sft)
    rl_path = _write("rl_records.jsonl", rl)
    sched_path = _write("scheduler_records.jsonl", sched)

    return {
        "sft": len(sft),
        "rl": len(rl),
        "scheduler": len(sched),
        "sft_path": sft_path,
        "rl_path": rl_path,
        "scheduler_path": sched_path,
    }


def replay_hash(records: List[dict]) -> str:
    """Deterministic content hash over the replay-relevant fields."""
    keys = [
        "run_id", "round", "step", "operator", "target",
        "decision", "action", "outcome", "reward", "label",
    ]
    proj = []
    for rec in records:
        proj.append({k: rec.get(k) for k in keys})
    return C.hash_json(proj)


def replay(records: List[dict]) -> dict:
    """Deterministic replay: returns event sequence, hash and reward total."""
    sequence = []
    total_reward = 0.0
    for rec in records:
        outcome = rec.get("outcome") or {}
        reward = rec.get("reward") or {}
        sequence.append({
            "step": rec.get("step"),
            "operator": rec.get("operator"),
            "label": rec.get("label"),
            "status": outcome.get("status"),
            "cost_aware_reward": outcome.get("cost_aware_reward"),
        })
        total_reward += float(reward.get("total", 0.0) or 0.0)
    return {
        "event_sequence": sequence,
        "hash": replay_hash(records),
        "reward_total": round(total_reward, 8),
    }
