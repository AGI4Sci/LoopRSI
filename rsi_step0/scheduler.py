"""RSI Step0 Scheduler layer (M5).

Pure stdlib. Provides a deterministic fallback advisor, a policy-backed
learned scheduler and a budget guard that always returns a legal
``ScheduleAction`` from ``rsi_step0.contracts``.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, Optional

import rsi_step0.contracts as C


def _budget_exhausted(budget_remaining: Dict[str, Any]) -> bool:
    """True when remaining max_seconds or max_cost is exhausted (<= 0)."""
    max_seconds = budget_remaining.get("max_seconds")
    max_cost = budget_remaining.get("max_cost")
    try:
        if max_seconds is not None and float(max_seconds) <= 0.0:
            return True
        if max_cost is not None and float(max_cost) <= 0.0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _normalized_outcome(context: Dict[str, Any]) -> Optional[float]:
    """Extract last_outcome.normalized_outcome as float, or None."""
    last_outcome = context.get("last_outcome")
    if not isinstance(last_outcome, dict):
        return None
    value = last_outcome.get("normalized_outcome")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def deterministic_advise(context: dict, budget_remaining: dict) -> dict:
    """Deterministic fallback advisor mapping to stop / converge / continue."""
    if _budget_exhausted(budget_remaining):
        return {
            "decision": "stop",
            "reason": "budget exhausted",
            "budget_delta": 0.0,
            "next_task_id": "",
        }
    outcome = _normalized_outcome(context)
    if outcome is not None and outcome > 0.6:
        return {
            "decision": "converge",
            "reason": "last normalized_outcome > 0.6",
            "budget_delta": 0.0,
            "next_task_id": "",
        }
    return {
        "decision": "continue",
        "reason": "budget remaining and no convergence signal",
        "budget_delta": 0.0,
        "next_task_id": "",
    }


def guard_budget(proposed: dict, budget_remaining: dict) -> dict:
    """Guarantee a legal ScheduleAction from a proposed one.

    - decision is restricted to ``C.SCHEDULE_DECISIONS`` (invalid -> continue)
    - budget_delta is clamped to a sane range and to remaining max_seconds
    - exhausted budgets force decision == "stop"
    """
    raw_decision = proposed.get("decision") if isinstance(proposed, dict) else None
    if C.is_valid_schedule_decision(raw_decision):
        decision = raw_decision
    else:
        decision = "continue"
    if _budget_exhausted(budget_remaining):
        decision = "stop"

    raw_delta = proposed.get("budget_delta") if isinstance(proposed, dict) else None
    try:
        delta = float(raw_delta) if raw_delta is not None else 0.0
    except (TypeError, ValueError):
        delta = 0.0
    if not math.isfinite(delta):
        delta = 0.0
    delta = max(-1000.0, min(1000.0, delta))

    try:
        remaining = float(budget_remaining.get("max_seconds", 3600.0))
    except (TypeError, ValueError):
        remaining = 3600.0
    if delta > remaining:
        delta = remaining

    reason = proposed.get("reason") if isinstance(proposed, dict) else None
    if not isinstance(reason, str) or not reason:
        reason = f"guarded decision: {decision}"
    next_task_id = proposed.get("next_task_id") if isinstance(proposed, dict) else None
    if not isinstance(next_task_id, str):
        next_task_id = ""

    if decision == "stop":
        delta = 0.0

    return {
        "decision": decision,
        "reason": reason,
        "budget_delta": delta,
        "next_task_id": next_task_id,
    }


class LearnedScheduler:
    """Policy-backed scheduler; advise() always returns a legal ScheduleAction.

    When no policy is configured (or it cannot produce a suggestion) the
    scheduler falls back to ``deterministic_advise``.
    """

    def __init__(self, policy: Optional[Dict[str, Any]] = None, seed: int = 0):
        self.seed = int(seed)
        self.policy = policy
        self.rng = random.Random(self.seed)

    def _policy_suggestion(self, context: Dict[str, Any]) -> Optional[dict]:
        if self.policy is None:
            return None
        if not isinstance(self.policy, dict) or not isinstance(context, dict):
            return None
        ops = self.policy.get("operators")
        if isinstance(ops, list) and ops:
            operator = self.rng.choice(ops)
            return {
                "decision": "continue",
                "reason": f"policy operator: {operator}",
                "budget_delta": 1.0,
                "next_task_id": context.get("task_id", ""),
            }
        decision = self.policy.get("decision")
        if C.is_valid_schedule_decision(decision):
            return {
                "decision": decision,
                "reason": "learned policy suggestion",
                "budget_delta": 1.0,
                "next_task_id": context.get("task_id", ""),
            }
        return None

    def advise(self, context: dict) -> dict:
        """Return a legal ScheduleAction, falling back to deterministic advice."""
        budget_remaining: Dict[str, Any] = {}
        if isinstance(context, dict):
            resource_state = context.get("resource_state")
            if isinstance(resource_state, dict):
                budget_remaining = resource_state
        proposed = self._policy_suggestion(context)
        if proposed is None:
            return deterministic_advise(context, budget_remaining)
        return guard_budget(proposed, budget_remaining)


def replay(config: dict, **kwargs) -> dict:
    """CLI-compatible replay entrypoint.

    Wraps the deterministic scheduler into a replay pipeline over the
    configured trajectory record set, returning a bounded summary dict.
    The harness keeps final scheduling authority; this only produces an
    advisory schedule action.
    """
    import json as _json
    import os as _os

    run_cfg = config.get("run", {}) if isinstance(config, dict) else {}
    traj_dir = (
        (config.get("trajectory") or {}).get("output_dir")
        if isinstance(config, dict)
        else None
    ) or run_cfg.get("trajectory_dir") or "output/rsi_step0/trajectories"

    records: list = []
    records_path = _os.path.join(traj_dir, "records.jsonl")
    if _os.path.isfile(records_path):
        with open(records_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(_json.loads(line))
                    except Exception:
                        continue

    seed = int(run_cfg.get("seed", 0))
    task_id = str(run_cfg.get("task_id", "vcc25_h1"))
    budget = dict(run_cfg.get("budget", {"max_seconds": 3600, "max_gpu_seconds": 3600, "max_cost": 1.0}))

    context = {
        "run_id": run_cfg.get("run_id", "replay"),
        "round": int(run_cfg.get("rounds", 1)),
        "task_id": task_id,
        "history": [],
        "last_outcome": records[-1].get("outcome") if records else None,
        "available_actions": [],
        "resource_state": {},
    }
    advice = deterministic_advise(context, budget)
    guarded = guard_budget(advice, budget)
    return {
        "ok": True,
        "seed": seed,
        "task_id": task_id,
        "n_records": len(records),
        "fold": "replay",
        "schedule": guarded,
    }
