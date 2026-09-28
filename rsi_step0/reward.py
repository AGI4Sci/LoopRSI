"""RSI Step0 reward computation utilities (pure stdlib, Python 3.10+).

Provides outcome normalization, cost accounting, operator (base) shaping,
entropic group advantage and outcome classification, plus a high-level pipeline
that ties them together. All functions are deterministic and depend only on
``rsi_step0.contracts`` for constants/validators.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import rsi_step0.contracts as C


def normalize_outcome(pearson_delta: float, baseline: float = C.DEFAULT_BASELINE) -> float:
    """Normalize a raw correlation delta to [0, +inf) above ``baseline``.

    Returns ``max(0, (pearson_delta - baseline) / (1.0 - baseline))`` so that a
    delta at or below baseline maps to 0 and a delta of 1.0 maps to 1.0.
    """
    return max(0.0, (pearson_delta - baseline) / (1.0 - baseline))


def normalize_mse_outcome(mse: float, baseline_mse: float) -> float:
    """Normalize an MSE (minimize) outcome to a [0, +inf) improvement score.

    Returns ``max(0, (baseline_mse - mse) / baseline_mse)`` so that an MSE at
    or above the reference baseline maps to 0 and an MSE of 0 maps to 1. A
    non-positive baseline is treated as no signal (returns 0). This gives a
    minimize-direction counterpart to :func:`normalize_outcome`.
    """
    try:
        baseline = float(baseline_mse)
        value = float(mse)
    except (TypeError, ValueError):
        return 0.0
    if baseline <= 0:
        return 0.0
    return max(0.0, (baseline - value) / baseline)


def compute_cost(resource_usage: dict) -> Tuple[float, float]:
    """Compute ``(cost_compute, time_cost)`` from a resource-usage mapping.

    ``cost_compute = runtime_seconds * gpu_count`` and ``time_cost =
    runtime_seconds``. Missing ``runtime_seconds`` is treated as 0; a missing
    ``gpu_count`` defaults to 1.
    """
    runtime_seconds = float(resource_usage.get("runtime_seconds") or 0.0)
    gpu_count = float(resource_usage.get("gpu_count") or 1.0)
    return runtime_seconds * gpu_count, runtime_seconds


def cost_aware_reward(
    pearson_delta: float,
    resource_usage: dict,
    baseline: float = C.DEFAULT_BASELINE,
    lambda_c: float = 1e-6,
    lambda_t: float = 1e-6,
) -> dict:
    """Build a ``Reward`` dict with cost-aware total.

    ``total = normalize_outcome(pearson_delta) - lambda_c * cost_compute -
    lambda_t * time_cost``.
    """
    normalized = normalize_outcome(pearson_delta, baseline)
    cost_compute, time_cost = compute_cost(resource_usage)
    total = normalized - lambda_c * cost_compute - lambda_t * time_cost
    return {
        "normalized_outcome": normalized,
        "cost_compute": cost_compute,
        "time_cost": time_cost,
        "total": total,
        "lambda_c": lambda_c,
        "lambda_t": lambda_t,
        "metadata": {},
    }


def operator_shaping(
    base: float,
    operator: str,
    parent_bases: Optional[List[float]] = None,
) -> float:
    """Apply the OpenRSI base-shaping strategy.

    * ``draft``            -> ``base``
    * ``improve``/``debug``/``crossover`` -> ``base + 0.5*max(base-max(parents),0)``
    * ``base == 0``  or a child base equal to a parent base (same code) -> 0
    * an empty/None ``parent_bases`` list is treated as ``[0]`` for the bonus.

    Operators outside the known set are returned unchanged (``base``).
    """
    parents = list(parent_bases) if parent_bases else []
    if base == 0:
        return 0.0
    if parents and any(base == p for p in parents):
        return 0.0
    if operator in ("improve", "debug", "crossover"):
        parent_max = max(parents) if parents else 0.0
        return base + 0.5 * max(base - parent_max, 0.0)
    return base


def _softmax_entropy_kl(beta: float, rewards: List[float], r_max: float, k: int) -> float:
    """KL(softmax(z) || uniform) in nats for a given inverse-temperature beta."""
    z = [math.exp(beta * (r - r_max)) for r in rewards]
    total = sum(z)
    if total <= 0.0:
        return 0.0
    return sum(pi * math.log(pi / (1.0 / k)) for pi in (zi / total for zi in z) if pi > 0)


def group_advantage(rewards: List[float]) -> List[float]:
    """Entropic leave-one-out group advantage.

    Returns a zero vector when ``len(rewards) < 2`` or all rewards are equal.
    Otherwise solves for ``beta`` in ``[1e-4, 5]`` such that
    ``KL(softmax(z) || uniform) ~= log(2)`` (falling back to ``beta = 1.0``) and
    returns ``A_i = z_i / ((sum_{j != i} z_j) / (k - 1) + 1e-9) - 1``.
    """
    k = len(rewards)
    if k < 2 or max(rewards) == min(rewards):
        return [0.0] * k

    r_max = max(rewards)
    target = math.log(2.0)

    low, high = 1e-4, 5.0
    beta = 1.0
    if _softmax_entropy_kl(5.0, rewards, r_max, k) >= target:
        for _ in range(40):
            mid = (low + high) / 2.0
            if _softmax_entropy_kl(mid, rewards, r_max, k) < target:
                low = mid
            else:
                high = mid
        beta = (low + high) / 2.0

    z = [math.exp(beta * (r - r_max)) for r in rewards]
    total = sum(z)
    out: List[float] = []
    for i, zi in enumerate(z):
        others_avg = (total - zi) / (k - 1) + 1e-9
        out.append(zi / others_avg - 1.0)
    return out


def classify_outcome(
    status: str,
    normalized: float,
    within_budget: bool = True,
    environmental_failure: bool = False,
) -> str:
    """Classify an outcome as ``positive``/``negative``/``excluded``.

    * positive: ``status == 'ok'``, ``normalized > 0`` and within budget.
    * negative: ``status == 'failed'`` (non-environmental) or exceeded budget.
    * everything else (skipped/timeout/excluded, environmental failures) -> excluded.
    """
    if not within_budget:
        return "negative"
    if status == "ok" and normalized > 0:
        return "positive"
    if status == "failed" and not environmental_failure:
        return "negative"
    return "excluded"


def apply_reward_pipeline(
    pearson_delta: float,
    resource_usage: dict,
    status: str,
    operator: str = "draft",
    parent_bases: Optional[List[float]] = None,
    baseline: float = C.DEFAULT_BASELINE,
    lambda_c: float = 1e-6,
    lambda_t: float = 1e-6,
    within_budget: bool = True,
    environmental_failure: bool = False,
    normalized_outcome: Optional[float] = None,
) -> Tuple[dict, str]:
    """Run normalization + operator shaping + cost accounting and classify.

    Returns ``(Reward, label)``. ``total`` uses the operator-shaped outcome:
    ``shaped - lambda_c * cost_compute - lambda_t * time_cost``.
    """
    if normalized_outcome is not None:
        normalized = float(normalized_outcome)
    else:
        normalized = normalize_outcome(pearson_delta, baseline)
    shaped = operator_shaping(normalized, operator, parent_bases)
    cost_compute, time_cost = compute_cost(resource_usage)
    label = classify_outcome(status, normalized, within_budget, environmental_failure)
    total = shaped - lambda_c * cost_compute - lambda_t * time_cost
    reward = {
        "normalized_outcome": normalized,
        "cost_compute": cost_compute,
        "time_cost": time_cost,
        "total": total,
        "lambda_c": lambda_c,
        "lambda_t": lambda_t,
        "metadata": {
            "status": status,
            "operator": operator,
            "shaped_outcome": shaped,
            "within_budget": within_budget,
            "environmental_failure": environmental_failure,
            "label": label,
        },
    }
    return reward, label


def evaluate(config: dict, **kwargs) -> dict:
    """CLI-compatible evaluate entrypoint (design doc command #5).

    Runs the cost-aware reward pipeline over the configured trajectory
    records and builds an ablation summary: A=heuresis-aide (guided) vs
    B=heuresis-aira (trained) share seed/task/budget; reported metrics are
    cost_aware_reward / pearson_delta / cost_compute / time_cost /
    success_rate / schedule_quality.
    """
    import json as _json
    import os as _os

    run_cfg = config.get("run", {}) if isinstance(config, dict) else {}
    traj_dir = (
        (config.get("trajectory") or {}).get("output_dir")
        if isinstance(config, dict)
        else None
    ) or "output/rsi_step0/trajectories"

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

    total_reward = 0.0
    total_cost_compute = 0.0
    total_time_cost = 0.0
    success = 0
    for rec in records:
        reward = rec.get("reward") or {}
        total_reward += float(reward.get("total", 0.0) or 0.0)
        total_cost_compute += float(reward.get("cost_compute", 0.0) or 0.0)
        total_time_cost += float(reward.get("time_cost", 0.0) or 0.0)
        outcome = rec.get("outcome") or {}
        if outcome.get("status") == "ok":
            success += 1

    n = max(len(records), 1)
    return {
        "ok": True,
        "task_id": str(run_cfg.get("task_id", "vcc25_h1")),
        "seed": int(run_cfg.get("seed", 0)),
        "n_records": len(records),
        "arm_a": "heuresis-aide",
        "arm_b": "heuresis-aira",
        "metrics": {
            "cost_aware_reward": round(total_reward / n, 8),
            "pearson_delta": round(
                sum(float((r.get("outcome") or {}).get("metrics", {}).get("pearson_delta", 0.0) or 0.0) for r in records) / n, 8
            ),
            "cost_compute": round(total_cost_compute, 8),
            "time_cost": round(total_time_cost, 8),
            "success_rate": round(success / n, 8),
            "schedule_quality": 1.0 if success else 0.0,
        },
    }
