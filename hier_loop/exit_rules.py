"""Per-layer exit rules for the hierarchical loop.

Three exit types:
- ``converge_or_budget``: stop when the tracked metric's last improvement is
  below ``converge_delta`` OR when the step budget is exhausted.
- ``threshold``: stop once the tracked metric reaches ``min`` (or ``max``).
- ``budget``: stop when steps/budget are exhausted.
Every exit returns an ``ExitDecision`` used to emit a ``kind=exit`` message and
an experience card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

EXIT_TYPES = ("converge_or_budget", "threshold", "budget")


@dataclass
class ExitDecision:
    triggered: bool
    reason: str  # one of: converge | threshold | budget | user
    detail: str
    metric_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reason": self.reason,
            "detail": self.detail,
            "metric_value": self.metric_value,
        }


def _metric(step: Dict[str, Any], metric: Optional[str]) -> Optional[float]:
    if not metric:
        return step.get("score")
    value = step.get("metrics", {}).get(metric)
    if value is None:
        value = step.get("score")
    return value


def evaluate_exit(
    rule: Dict[str, Any],
    step: Dict[str, Any],
    max_steps: Optional[int] = None,
    steps_done: int = 0,
    metric_history: Optional[List[float]] = None,
) -> ExitDecision:
    """``step`` carries ``metrics``/``score``; ``rule`` carries the exit config."""
    rule = dict(rule or {})
    exit_type = rule.get("type", "budget")
    if exit_type not in EXIT_TYPES:
        raise ValueError(f"invalid exit type {exit_type!r}; expected one of {EXIT_TYPES}")
    metric = rule.get("metric")
    history = metric_history or []

    if max_steps is not None and steps_done >= max_steps:
        return ExitDecision(True, "budget", f"reached max_steps={max_steps}")

    if exit_type == "threshold":
        val = _metric(step, metric)
        threshold = rule.get("min")
        if val is not None and threshold is not None and val >= float(threshold):
            return ExitDecision(True, "threshold", f"{metric or 'score'} {val} >= {threshold}", val)
        return ExitDecision(False, "", "threshold not reached", val)

    if exit_type == "converge_or_budget":
        val = _metric(step, metric)
        delta = rule.get("converge_delta")
        if val is not None and delta is not None and len(history) >= 2:
            improvement = abs(history[-1] - history[-2])
            if improvement < float(delta):
                return ExitDecision(
                    True, "converge", f"improvement {improvement} < converge_delta {delta}", val
                )
        if rule.get("budget_seconds") and step.get("cost", {}).get("seconds", 0) >= float(
            rule["budget_seconds"]
        ):
            return ExitDecision(True, "budget", "budget_seconds exhausted", val)
        return ExitDecision(False, "", "continue", val)

    # budget type (default)
    if step.get("cost", {}).get("seconds", 0) >= float(rule.get("budget_seconds", 0)):
        return ExitDecision(True, "budget", "budget_seconds exhausted")
    return ExitDecision(False, "", "continue")
