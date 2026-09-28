"""RSI Step0 unified contracts (M1-M7).

Single source of truth for the cross-module types, enums and helper validators used
by every other ``rsi_step0`` module. Pure stdlib, Python 3.10+.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict

SCHEMA_VERSION = "rsi_trajectory/v1"

# --------------------------------------------------------------------------- enums
OPERATORS: tuple = (
    "draft",
    "improve",
    "crossover",
    "tune",
    "evaluate",
    "question",
    "implement",
    "debug",
    "schedule",
)
# W2 trains only these; W4 adds `schedule`. (Step0 minimal closed loop.)
W2_TRAIN_OPERATORS: tuple = ("draft", "improve", "crossover")
MODES: tuple = ("thinking", "execution", "scheduling")
STATUSES: tuple = ("ok", "failed", "skipped", "timeout", "excluded")
SCHEDULE_DECISIONS: tuple = ("continue", "converge", "switch_direction", "stop")
LABELS: tuple = ("positive", "negative", "excluded")
MODELS: tuple = ("glm", "deepseek", "trained_pi", "deterministic")

# OpenRSI operator sampling probabilities (kept for provenance; W2 subset is
# renormalized over draft/improve/crossover).
DEFAULT_OPERATOR_PROBS: Dict[str, float] = {
    "draft": 0.50,
    "improve": 0.17,
    "crossover": 0.17,
    "debug": 0.16,
}

DEFAULT_BASELINE = 0.30614


# --------------------------------------------------------------------------- typed dicts
class Action(TypedDict):
    action_id: str
    run_id: str
    round: int
    step: int
    operator: str
    mode: str
    target: str
    decision: Dict[str, Any]
    budget: Dict[str, Any]  # max_seconds / max_gpu_seconds / max_cost
    provenance: Dict[str, Any]  # parent_action_id / model / seed


class DecisionContext(TypedDict):
    run_id: str
    round: int
    task_id: str
    history: List[Dict[str, Any]]
    last_outcome: Optional[Dict[str, Any]]
    available_actions: List[Dict[str, Any]]
    resource_state: Dict[str, Any]


class Outcome(TypedDict):
    status: str
    metrics: Dict[str, float]
    resource_usage: Dict[str, Any]
    cost_aware_reward: float
    error: Optional[str]


class Reward(TypedDict):
    normalized_outcome: float
    cost_compute: float
    time_cost: float
    total: float
    lambda_c: float
    lambda_t: float
    metadata: Dict[str, Any]


class ScheduleAction(TypedDict):
    decision: str
    reason: str
    budget_delta: float
    next_task_id: str


class TrajectoryRecord(TypedDict):
    schema_version: str
    record_id: str
    run_id: str
    round: int
    step: int
    operator: str
    mode: str
    target: str
    decision: Dict[str, Any]
    context: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    reward: Dict[str, Any]
    label: str
    dedupe_key: str
    provenance: Dict[str, Any]
    created_at: str


class EvaluationReport(TypedDict):
    run_id: str
    task_id: str
    evaluator: str
    metrics: Dict[str, Any]
    cost_aware_reward: float
    resource_usage: Dict[str, Any]
    decision: Optional[str]
    error: Optional[str]


# --------------------------------------------------------------------------- protocols
class Router(Protocol):
    def decide(self, context: DecisionContext) -> Action: ...
    def score(self, context: DecisionContext, candidates: List[Action]) -> List[float]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


class Environment(Protocol):
    def reset(self, seed: int, task_id: str) -> Dict[str, Any]: ...
    def step(self, action: Action) -> Outcome: ...


class Evaluator(Protocol):
    def evaluate(self, checkpoint: Dict[str, Any], context: DecisionContext) -> EvaluationReport: ...


# --------------------------------------------------------------------------- helpers
def make_dedupe_key(run_id: str, round_: int, proposal_id: Optional[str], variant: Optional[str]) -> str:
    """Stable dedupe key: (run_id, round, proposal_id, idea_variant.name)."""
    return "|".join(
        [
            str(run_id),
            str(round_),
            str(proposal_id if proposal_id is not None else ""),
            str(variant if variant is not None else ""),
        ]
    )


def make_record_id(run_id: str, round_: int, step: int, operator: str) -> str:
    base = f"{run_id}:{round_}:{step}:{operator}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def new_action_id() -> str:
    return uuid.uuid4().hex[:16]


def hash_json(obj: Any) -> str:
    """Deterministic content hash for provenance / replay checks."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


VALID_OPERATORS = frozenset(OPERATORS)
VALID_MODES = frozenset(MODES)
VALID_STATUSES = frozenset(STATUSES)
VALID_LABELS = frozenset(LABELS)
VALID_SCHEDULE_DECISIONS = frozenset(SCHEDULE_DECISIONS)


def is_valid_operator(op: Any) -> bool:
    return isinstance(op, str) and op in VALID_OPERATORS


def is_valid_mode(mode: Any) -> bool:
    return isinstance(mode, str) and mode in VALID_MODES


def is_valid_status(status: Any) -> bool:
    return isinstance(status, str) and status in VALID_STATUSES


def is_valid_label(label: Any) -> bool:
    return isinstance(label, str) and label in VALID_LABELS


def is_valid_schedule_decision(decision: Any) -> bool:
    return isinstance(decision, str) and decision in VALID_SCHEDULE_DECISIONS
