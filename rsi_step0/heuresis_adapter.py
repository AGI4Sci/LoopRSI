"""RSI Step0 heuresis thin adapter layer.

Provides a fully deterministic local environment/evaluator and thin wrappers
over the real heuresis engine (``omni_ar.autoresearch`` /
``tasks.vcc25.native_evaluator``) when it is importable. Pure stdlib.

All heuresis imports are lazy and guarded: importing this module never raises
when heuresis (or torch) is absent. The deterministic surface mirrors the
shared env/evaluator protocol from ``rsi_step0.contracts``.
"""

from __future__ import annotations

import importlib
import random
from typing import Any, Dict, List, Optional

import rsi_step0.contracts as C

ENGINE_MODULES = ("omni_ar.autoresearch", "tasks.vcc25.native_evaluator")
DEFAULT_TASK_ID = "vcc25_h1"
DEFAULT_LAMBDA_C = 1e-6
DEFAULT_LAMBDA_T = 1e-6

NOT_AVAILABLE = {
    "ok": False,
    "available": False,
    "reason": "heuresis engine not available",
    "result": None,
}


# --------------------------------------------------------------------------- engine probe
def engine_available() -> bool:
    """Return True when both real heuresis modules import successfully."""
    for name in ENGINE_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            return False
    return True


# --------------------------------------------------------------------------- reward helpers
def _reward_module() -> Any:
    """Lazily import the reward module; None when it does not exist yet."""
    try:
        import rsi_step0.reward as R
        return R
    except Exception:
        return None


def _cost_aware_reward(pearson_delta: float, resource_usage: Dict[str, Any]) -> float:
    """Outcome-level cost-aware reward (float).

    Prefers ``rsi_step0.reward.cost_aware_reward`` (a Reward dict whose
    ``total`` is the cost-aware value). If the reward module is unavailable or
    the call fails, falls back to a deterministic mirror of the same formula so
    the environment still works on torch-/reward-less remotes.
    """
    R = _reward_module()
    if R is not None and hasattr(R, "cost_aware_reward"):
        try:
            reward = R.cost_aware_reward(
                pearson_delta,
                resource_usage,
                baseline=C.DEFAULT_BASELINE,
                lambda_c=DEFAULT_LAMBDA_C,
                lambda_t=DEFAULT_LAMBDA_T,
            )
            if isinstance(reward, dict) and "total" in reward:
                return float(reward["total"])
            return float(reward)
        except Exception:
            pass
    normalized = max(0.0, (pearson_delta - C.DEFAULT_BASELINE) / (1.0 - C.DEFAULT_BASELINE))
    runtime = float(resource_usage.get("runtime_seconds") or 0.0)
    gpus = float(resource_usage.get("gpu_count") or 1.0)
    return normalized - DEFAULT_LAMBDA_C * runtime * gpus - DEFAULT_LAMBDA_T * runtime


# --------------------------------------------------------------------------- deterministic env
class DeterministicEnv:
    """Deterministic environment implementing the ``Environment`` protocol.

    ``pearson_delta`` evolves deterministically from ``C.DEFAULT_BASELINE``:
    ``draft`` adds a seed-derived increment; ``improve``/``crossover`` add a
    further increment on top of the parent (most recent) metric. Resources are
    a seeded ``runtime_seconds`` and ``gpu_count`` of 1, and the outcome's
    ``cost_aware_reward`` is computed via ``R.cost_aware_reward`` (fallback
    included). All state is sealed inside the instance: the same seed and
    action sequence always reproduce the same outcomes.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._task_id = DEFAULT_TASK_ID
        self._rng = random.Random(self.seed)
        self._base_metric = C.DEFAULT_BASELINE
        self._last_metric = C.DEFAULT_BASELINE
        self._history: List[Dict[str, Any]] = []

    # -- Environment protocol --------------------------------------------
    def reset(self, seed: Optional[int] = None, task_id: str = DEFAULT_TASK_ID) -> Dict[str, Any]:
        """Re-seal instance state; returns the task observation dict."""
        if seed is not None:
            self.seed = int(seed)
        self._task_id = str(task_id)
        self._rng = random.Random(f"{self.seed}:{self._task_id}")
        self._base_metric = C.DEFAULT_BASELINE
        self._last_metric = C.DEFAULT_BASELINE
        self._history = []
        return self._observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Advance the environment; returns an ``Outcome`` dict."""
        operator = action.get("operator") if isinstance(action, dict) else None
        operator = operator if isinstance(operator, str) else "draft"
        metric, resource = self._deterministic_draw(operator)
        outcome = {
            "status": "ok",
            "metrics": {"pearson_delta": metric},
            "resource_usage": resource,
            "cost_aware_reward": _cost_aware_reward(metric, resource),
            "error": None,
        }
        self._history.append({"action": action, "outcome": outcome})
        self._last_metric = metric
        return outcome

    # -- internals --------------------------------------------------------
    def _observation(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "seed": self.seed,
            "baseline": self._base_metric,
            "history": [entry["outcome"] for entry in self._history],
            "last_outcome": self._history[-1]["outcome"] if self._history else None,
            "available_actions": [{"operator": op} for op in C.W2_TRAIN_OPERATORS],
            "resource_state": {
                "gpu_count": 1,
                "max_seconds": 3600,
                "max_gpu_seconds": 3600,
                "max_cost": 1.0,
            },
        }

    def _deterministic_draw(self, operator: str):
        """Compute the next metric and resources from the sealed rng state."""
        parent = self._last_metric if self._history else self._base_metric
        if operator == "draft":
            base = self._base_metric
            metric_inc = self._rng.uniform(0.0, 0.05)
        elif operator == "improve":
            base = parent
            metric_inc = self._rng.uniform(0.01, 0.08)
        elif operator == "crossover":
            base = parent
            metric_inc = self._rng.uniform(0.005, 0.06)
        else:
            base = parent
            metric_inc = self._rng.uniform(0.0, 0.02)
        metric = round(base + metric_inc, 6)
        runtime = round(30.0 + self._rng.uniform(0.0, 240.0), 1)
        resource = {"runtime_seconds": runtime, "gpu_count": 1}
        return metric, resource


# --------------------------------------------------------------------------- heuresis env
class HeuresisEnv:
    """Environment wrapper: real heuresis when available, else DeterministicEnv.

    ``reset``/``step`` follow the shared ``Environment`` protocol. If the real
    engine import succeeds, its Environment is wrapped and used; any runtime
    failure drops back to the deterministic implementation and records the
    reason in ``real_error``.
    """

    def __init__(self, seed: int = 0, prefer_real: Optional[bool] = None) -> None:
        self.seed = int(seed)
        self.real_error: Optional[str] = None
        self._real = None
        self._deterministic = DeterministicEnv(seed=self.seed)
        want_real = engine_available() if prefer_real is None else bool(prefer_real)
        if want_real:
            try:
                mod = importlib.import_module("omni_ar.autoresearch")
                self._real = mod.Environment(seed=self.seed)
            except Exception as exc:
                self._real = None
                self.real_error = f"{type(exc).__name__}: {exc}"

    @property
    def using_real_engine(self) -> bool:
        return self._real is not None

    def reset(self, seed: Optional[int] = None, task_id: str = DEFAULT_TASK_ID) -> Dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        if self._real is not None:
            try:
                obs = self._real.reset(self.seed, task_id)
                return self._asdict(obs)
            except Exception as exc:
                self._real = None
                self.real_error = f"{type(exc).__name__}: {exc}"
        return self._deterministic.reset(self.seed, task_id)

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._real is not None:
            try:
                outcome = self._real.step(action)
                return self._normalize_outcome(outcome)
            except Exception as exc:
                self._real = None
                self.real_error = f"{type(exc).__name__}: {exc}"
        return self._deterministic.step(action)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _asdict(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {"result": str(obj)}

    @staticmethod
    def _normalize_outcome(outcome: Any) -> Dict[str, Any]:
        if not isinstance(outcome, dict):
            outcome = HeuresisEnv._asdict(outcome)
        out = {
            "status": outcome.get("status", "ok"),
            "metrics": outcome.get("metrics") or {},
            "resource_usage": outcome.get("resource_usage") or {},
            "cost_aware_reward": float(outcome.get("cost_aware_reward") or 0.0),
            "error": outcome.get("error"),
        }
        return out


# --------------------------------------------------------------------------- evaluator
class HeuresisEvaluator:
    """Evaluator: heuresis native evaluator when available, else deterministic.

    ``evaluate(checkpoint, context)`` returns an ``EvaluationReport`` with a
    deterministic ``cost_aware_reward`` derived from the shared constants and
    (when present) the last history outcome.
    """

    def __init__(self, prefer_real: Optional[bool] = None) -> None:
        self.native_error: Optional[str] = None
        self._try_real = engine_available() if prefer_real is None else bool(prefer_real)

    def evaluate(self, checkpoint: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if self._try_real:
            try:
                mod = importlib.import_module("tasks.vcc25.native_evaluator")
                report = mod.evaluate(checkpoint, context)
                if isinstance(report, dict) and "cost_aware_reward" in report:
                    return report
            except Exception as exc:
                self.native_error = f"{type(exc).__name__}: {exc}"
                return self._deterministic_report(checkpoint, context, self.native_error)
        return self._deterministic_report(checkpoint, context)

    def _deterministic_report(
        self,
        checkpoint: Dict[str, Any],
        context: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        ctx = context if isinstance(context, dict) else {}
        history = ctx.get("history") or []
        last = history[-1] if history and isinstance(history[-1], dict) else {}
        metrics = last.get("metrics") if isinstance(last, dict) else None
        if isinstance(metrics, dict) and isinstance(metrics.get("pearson_delta"), (int, float)):
            pearson = float(metrics["pearson_delta"])
        else:
            pearson = C.DEFAULT_BASELINE
        resource = {"runtime_seconds": 0.0, "gpu_count": 1}
        return {
            "run_id": str(ctx.get("run_id") or "evaluation"),
            "task_id": str(ctx.get("task_id") or DEFAULT_TASK_ID),
            "evaluator": "heuresis_deterministic",
            "metrics": {"pearson_delta": pearson},
            "cost_aware_reward": _cost_aware_reward(pearson, resource),
            "resource_usage": resource,
            "decision": None,
            "error": error,
        }


# --------------------------------------------------------------------------- thin adapters
class SkillRouterAdapter:
    """Thin, documented adapter around the heuresis skill router.

    ``run`` / ``invoke`` / ``train`` lazily resolve the real engine when
    ``engine_available()``; otherwise they return an explicit ``not available``
    payload. An injected ``engine`` object (duck-typed) takes precedence.
    """

    name = "skill_router"

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._module = None

    def available(self) -> bool:
        return self._engine is not None or engine_available()

    def _resolve(self, attr: str) -> Any:
        if self._engine is not None:
            return getattr(self._engine, attr, None) if callable(getattr(self._engine, attr, None)) else None
        if not engine_available():
            return None
        try:
            if self._module is None:
                self._module = importlib.import_module("omni_ar.autoresearch")
            return getattr(self._module, attr, None)
        except Exception:
            return None

    def _call(self, attr: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        target = self._resolve(attr)
        if target is None:
            if not self.available():
                return dict(NOT_AVAILABLE)
            return {"ok": False, "available": True,
                    "reason": f"heuresis engine has no callable {attr!r}", "result": None}
        try:
            result = target(*args, **kwargs)
        except Exception as exc:
            return {"ok": False, "available": True,
                    "reason": f"{attr} failed: {type(exc).__name__}: {exc}", "result": None}
        return {"ok": True, "available": True, "result": result}

    def run(self, task: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run a skill-router task (skill selection + execution) end to end."""
        return self._call("run", task, **kwargs)

    def invoke(self, task: Any, **kwargs: Any) -> Dict[str, Any]:
        """Invoke the skill router on a task and return its plan/decision."""
        return self._call("invoke", task, **kwargs)

    def train(self, dataset: Any, **kwargs: Any) -> Dict[str, Any]:
        """Fit/update the router's policy from an experience dataset."""
        return self._call("train", dataset, **kwargs)


class ModelFactoryAdapter:
    """Thin, documented adapter around the heuresis model factory.

    ``run`` / ``invoke`` / ``train`` map to factory-level callables once the
    real engine is available; otherwise they return an explicit
    ``not available`` payload.
    """

    name = "model_factory"

    def __init__(self, factory: Any = None) -> None:
        self._factory = factory
        self._module = None

    def available(self) -> bool:
        return self._factory is not None or engine_available()

    def _resolve(self, attr: str) -> Any:
        if self._factory is not None:
            return getattr(self._factory, attr, None) if callable(getattr(self._factory, attr, None)) else None
        if not engine_available():
            return None
        try:
            if self._module is None:
                self._module = importlib.import_module("omni_ar.autoresearch")
            return getattr(self._module, attr, None)
        except Exception:
            return None

    def _call(self, attr: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        target = self._resolve(attr)
        if target is None:
            if not self.available():
                return dict(NOT_AVAILABLE)
            return {"ok": False, "available": True,
                    "reason": f"heuresis engine has no callable {attr!r}", "result": None}
        try:
            result = target(*args, **kwargs)
        except Exception as exc:
            return {"ok": False, "available": True,
                    "reason": f"{attr} failed: {type(exc).__name__}: {exc}", "result": None}
        return {"ok": True, "available": True, "result": result}

    def run(self, model_spec: Any, **kwargs: Any) -> Dict[str, Any]:
        """Create/fit a model described by ``model_spec``."""
        return self._call("run", model_spec, **kwargs)

    def invoke(self, checkpoint: Any, context: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Run inference for ``context`` using a trained ``checkpoint``."""
        return self._call("invoke", checkpoint, context, **kwargs)

    def train(self, dataset: Any, **kwargs: Any) -> Dict[str, Any]:
        """Train a model from ``dataset`` and return its checkpoint metadata."""
        return self._call("train", dataset, **kwargs)
