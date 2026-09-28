"""RSI Step0 Router layer (M2/M4).

Pure stdlib. Provides ``make_action`` plus two ``Router`` implementations
(``GuidedRouter`` and ``TrainedRouter``) that emit valid ``Action`` records
using the shared ``rsi_step0.contracts``.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional

import rsi_step0.contracts as C


def _default_budget() -> Dict[str, Any]:
    return {"max_seconds": 3600, "max_gpu_seconds": 3600, "max_cost": 1.0}


def _default_provenance() -> Dict[str, Any]:
    return {"parent_action_id": None, "model": "deterministic", "seed": 0}


def make_action(
    run_id: str,
    round_no: int,
    step: int,
    operator: str,
    mode: str,
    target: str,
    decision: Dict[str, Any],
    budget: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> dict:
    """Construct a legal Action, filling defaults for id/budget/provenance."""
    if not C.is_valid_operator(operator):
        raise ValueError(f"invalid operator: {operator!r}")
    if not C.is_valid_mode(mode):
        raise ValueError(f"invalid mode: {mode!r}")
    budget_resolved = dict(_default_budget())
    if budget is not None:
        budget_resolved.update(budget)
    provenance_resolved = dict(_default_provenance())
    if provenance is not None:
        provenance_resolved.update(provenance)
    return {
        "action_id": C.new_action_id(),
        "run_id": str(run_id),
        "round": int(round_no),
        "step": int(step),
        "operator": operator,
        "mode": mode,
        "target": str(target),
        "decision": dict(decision) if decision is not None else {},
        "budget": budget_resolved,
        "provenance": provenance_resolved,
    }


def _renormalized_w2_probs(operator_probs: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Return probabilities over W2_TRAIN_OPERATORS that sum to 1.0."""
    if operator_probs is None:
        probs: Dict[str, float] = {
            op: float(C.DEFAULT_OPERATOR_PROBS.get(op, 0.0)) for op in C.W2_TRAIN_OPERATORS
        }
    else:
        probs = {op: float(operator_probs.get(op, 0.0)) for op in C.W2_TRAIN_OPERATORS}
    total = sum(probs.values())
    if total <= 0.0:
        raise ValueError("operator probabilities must include a positive W2 operator")
    return {op: p / total for op, p in probs.items()}


def _softmax(weights: Dict[str, float]) -> Dict[str, float]:
    """Numerically stable softmax over operator weights."""
    keys = list(weights.keys())
    vals = [float(weights[k]) for k in keys]
    if not vals:
        return {}
    max_w = max(vals)
    exp_w = [math.exp(v - max_w) for v in vals]
    total = sum(exp_w)
    if total <= 0.0:
        raise ValueError("policy weights must not all be zero")
    return {k: e / total for k, e in zip(keys, exp_w)}


def _hash_based_score(candidate: Dict[str, Any]) -> float:
    """Stable heuristic score in [0.0, 1.0) from the content hash."""
    digest = C.hash_json(candidate)
    return int(digest, 16) / float(2 ** (len(digest) * 4))


def _sample_operator(probs: Dict[str, float], rng: random.Random) -> str:
    """Sample a W2 operator from a normalized probability dict."""
    r = rng.random()
    cumulative = 0.0
    for op in C.W2_TRAIN_OPERATORS:
        cumulative += probs[op]
        if r <= cumulative:
            return op
    return C.W2_TRAIN_OPERATORS[-1]


def _decision_payload(operator: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Build the operator-specific ``decision`` payload."""
    task_id = context.get("task_id", "")
    history = context.get("history") or []
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    last_action_id = last.get("action_id", "")
    last_task = last.get("target", task_id)

    if operator == "draft":
        return {
            "hypothesis": f"draft hypothesis for {task_id}",
            "change_scope": f"module-level:{task_id}",
            "expected_effect": "improve normalized_outcome",
            "acceptance_criteria": "normalized_outcome > baseline",
        }
    if operator == "improve":
        return {
            "target_action_id": last_action_id,
            "diagnosis": f"heuristic diagnosis for {last_task}",
            "patch": f"incremental patch for {last_task}",
        }
    if operator == "crossover":
        parents: List[str] = []
        candidates = context.get("available_actions") or history
        for entry in candidates[-2:] if isinstance(candidates, list) else []:
            if isinstance(entry, dict) and entry.get("action_id"):
                parents.append(entry["action_id"])
        if not parents:
            parents = [last_action_id] if last_action_id else []
        return {
            "parent_action_ids": parents,
            "recombination_strategy": "uniform",
        }
    return {"operator": operator}


class GuidedRouter:
    """Router sampling draft/improve/crossover from renormalized probabilities."""

    def __init__(
        self,
        seed: int = 0,
        operator_probs: Optional[Dict[str, float]] = None,
    ):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.operator_probs = _renormalized_w2_probs(operator_probs)

    def _next_step(self, context: Dict[str, Any]) -> int:
        if context.get("step") is not None:
            return int(context["step"])
        history = context.get("history") or []
        return len(history) + 1

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        operator = _sample_operator(self.operator_probs, self.rng)
        payload = _decision_payload(operator, context)
        return make_action(
            run_id=context.get("run_id", ""),
            round_no=context.get("round", 0),
            step=self._next_step(context),
            operator=operator,
            mode="execution",
            target=context.get("task_id", ""),
            decision=payload,
        )

    def score(
        self, context: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> List[float]:
        return [_hash_based_score(c) for c in candidates]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {"seed": self.seed, "operator_probs": self.operator_probs},
                fh,
                sort_keys=True,
            )

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.seed = int(data.get("seed", self.seed))
        raw = data.get("operator_probs")
        self.operator_probs = (
            _renormalized_w2_probs(raw) if isinstance(raw, dict) else self.operator_probs
        )
        self.rng = random.Random(self.seed)


class TrainedRouter:
    """Router sampling operators from softmax-normalized policy weights."""

    def __init__(self, policy: Optional[Dict[str, float]] = None, seed: int = 0):
        self.seed = int(seed)
        self.policy: Dict[str, float] = (
            {str(k): float(v) for k, v in policy.items()} if policy else {}
        )
        self.rng = random.Random(self.seed)

    def _operators(self) -> List[str]:
        if self.policy:
            return list(self.policy.keys())
        return list(C.W2_TRAIN_OPERATORS)

    def _probabilities(self) -> Dict[str, float]:
        ops = self._operators()
        if not ops:
            return {}
        weights = {op: max(float(self.policy.get(op, 0.0)), 0.0) for op in ops}
        if sum(weights.values()) <= 0.0:
            return {op: 1.0 / len(ops) for op in ops}
        return _softmax(weights)

    def _next_step(self, context: Dict[str, Any]) -> int:
        if context.get("step") is not None:
            return int(context["step"])
        history = context.get("history") or []
        return len(history) + 1

    def decide(self, context: Dict[str, Any]) -> Dict[str, Any]:
        probs = self._probabilities()
        ops = list(probs.keys())
        if not ops:
            raise ValueError("TrainedRouter policy is empty")
        r = self.rng.random()
        cumulative = 0.0
        operator = ops[-1]
        for op in ops:
            cumulative += probs[op]
            if r <= cumulative:
                operator = op
                break
        payload = _decision_payload(operator, context)
        return make_action(
            run_id=context.get("run_id", ""),
            round_no=context.get("round", 0),
            step=self._next_step(context),
            operator=operator,
            mode="execution",
            target=context.get("task_id", ""),
            decision=payload,
        )

    def score(
        self, context: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> List[float]:
        return [float(self.policy.get(c.get("operator"), 0.0)) for c in candidates]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"seed": self.seed, "policy": self.policy}, fh, sort_keys=True)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.seed = int(data.get("seed", self.seed))
        raw = data.get("policy")
        self.policy = (
            {str(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        )
        self.rng = random.Random(self.seed)
