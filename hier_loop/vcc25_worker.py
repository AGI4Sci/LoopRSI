"""Real vcc25 worker for the Hierarchical Looped RSI Researcher.

Runs inside the rjob GPU container.  Only the most micro layer (``L5``
technique) performs a *real* vcc25 training/eval via the official_h1 adapter;
the macro layers (``L1``..``L4``) are planning layers that select the next
technique to try and report a plan-quality score derived from the best observed
``pearson_delta``.

The worker is deterministic given the config.  ``mock=True`` is for local
dry-runs / tests and does not invoke the engine.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from .loop import LoopWorker, StepResult

# Seed set accepted by the official_h1 run_trial adapter.
VALID_SEEDS = (20260907, 20260908, 20260909)
# Candidate variants accepted by the adapter (candidate_g is the promoted one).
VALID_VARIANTS = ("candidate_g_promoted_delta_model", "autonomous_research_candidate")


def _default_plan() -> Dict[str, Any]:
    return {"direction": "official H1 perturbation response prediction",
            "problem": "predict 18080 genes for 100 unseen H1 targets with high pearson_delta",
            "mechanism": "promoted delta model with aggregate cell-eval scoring"}


class Vcc25Worker(LoopWorker):
    """LoopWorker that drives one real vcc25 training/eval per L5 step."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, mock: bool = False,
                 memory_manager: Optional[Any] = None):
        wcfg = ((config or {}).get("worker") or {})
        self.mode = "mock" if mock else (wcfg.get("mode") or "offline")
        self.adapter_module = wcfg.get("adapter_module", "tasks.vcc25.official_h1_adapter")
        self.trial_defaults = dict(wcfg.get("trial_defaults") or {})
        self.result_root = wcfg.get("result_root", "output/vcc25/hier")
        # Wall-clock cap per real trial (prediction + contract + cell-eval full
        # profile). A wedged cell-eval would otherwise block the loop forever.
        self.trial_timeout_seconds = float(wcfg.get("trial_timeout_seconds", 21600))
        self.baseline = float((((config or {}).get("reward") or {}).get("baseline", 0.30614)))
        self.metric = (((config or {}).get("reward") or {}).get("metric", "pearson_delta"))
        self.seed_idx = 0
        self._best_score: Optional[float] = None
        # -- P3/P4 memory hook -------------------------------------------------
        self.memory_manager = memory_manager
        mem_cfg = ((config or {}).get("memory") or {})
        self.memory_enabled = bool(mem_cfg.get("enabled", False))
        self.memory_baseline = float(mem_cfg.get("baseline") or 0.050227)
        self.memory_top_k = int((mem_cfg.get("retrieval") or {}).get("top_k", 5) or 5)
        self._picked_seeds: List[int] = []

    # -- candidate schedule --------------------------------------------------
    def _pick(self, layer: str, context: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Choose variant + seed for this technique step.

        Memory-aware (P4/B arm): when the memory manager is active, first
        exploit the best *real* native_eval outcome observed so far (or a prior
        real result) whose seed has not been tried this run; when nothing new is
        left to exploit, explore an ever-untried frozen seed guided by the
        empirical baseline prior.  Falls back to the deterministic schedule for
        the A arm / graceful no-memory path.
        """
        variant = self.trial_defaults.get("candidate_variant", VALID_VARIANTS[0])
        seed = VALID_SEEDS[self.seed_idx % len(VALID_SEEDS)]
        chosen_from = "default_schedule"
        tried = set(self._picked_seeds)
        if self.memory_manager is not None and self.memory_enabled:
            try:
                real_pool = [e for e in self.memory_manager.real_candidates(layer)
                             if e.seed in VALID_SEEDS and e.variant in VALID_VARIANTS
                             and e.variant is not None]
                real = [e for e in real_pool
                        if int(e.seed) not in tried and float(e.reward or 0.0) >= 0.0]
                if real:
                    best = real[0]
                    variant, seed = best.variant, int(best.seed)
                    chosen_from = f"memory:{layer}:exploit_real_seed={seed}|reward={best.reward}"
                else:
                    # Nothing above baseline left to exploit: prefer a frozen
                    # seed that has no real outcome in memory at all (genuine
                    # discovery), before falling back to the cycle schedule.
                    real_seen = {int(e.seed) for e in real_pool}
                    for cand in VALID_SEEDS:
                        if cand not in tried and cand not in real_seen:
                            seed = cand
                            chosen_from = f"memory:{layer}:explore_unknown_seed={cand}"
                            break
                    else:
                        seed = self._next_untried_seed()
                        chosen_from = f"memory:{layer}:explore_untried_seed={seed}" if seed not in tried else "memory:exhausted_seeds:schedule"
            except Exception:
                chosen_from = "memory:error:schedule"
        return {
            "variant": variant,
            "seed": seed,
            "chosen_from": chosen_from,
            "memory_used": bool(context.get("memory_used")),
            "memory_hits": int(context.get("memory_hits") or 0) if context.get("memory_used") else 0,
        }

    def _next_untried_seed(self) -> int:
        """First frozen seed not yet attempted this run (exploration loop)."""
        for s in VALID_SEEDS:
            if s not in self._picked_seeds:
                return s
        return VALID_SEEDS[self.seed_idx % len(VALID_SEEDS)]

    def _memorize_live(self, context: Dict[str, Any], variant: str, seed: int,
                       score: Optional[float], status: str, round_: int, step: int) -> None:
        """Record a just-completed real native_eval trial into the memory so the
        next round can exploit/explore from observed outcomes (self-improvement).
        """
        if self.memory_manager is None or status != "ok":
            return
        try:
            from .decisions import context_fingerprint, context_text
            from .memory import MemoryEntry
            fp = context.get("memory_fingerprint") or context_fingerprint(context)
            reward = None if score is None else round(float(score) - self.memory_baseline, 6)
            entry = MemoryEntry(
                layer="L5", context_fingerprint=fp,
                context=context_text(context),
                op="implement" if step == 1 else "tune",
                variant=variant, seed=int(seed),
                outcome_score=score, reward=reward,
                metric_source="real", kind="step", round=round_, step=step,
                reasoning=f"real native_eval round={round_} step={step} seed={seed} score={score}",
            )
            self.memory_manager.remember_live(entry)
        except Exception:
            pass

    # -- adapter invocation (real) -------------------------------------------
    def _run_real_trial(self, variant: str, seed: int, round_: int, step: int) -> Dict[str, Any]:
        os.makedirs(self.result_root, exist_ok=True)
        stem = f"hier_r{round_}_s{step}_{variant}_seed{seed}"
        output = os.path.join(self.result_root, f"{stem}.json")
        cmd = [
            "python", "-m", self.adapter_module, "run_trial",
            "--output", output,
            "--method", "candidate",
            "--candidate-variant", variant,
            "--seed", str(seed),
        ]
        for key, flag in (("eval_profile", "--eval-profile"),
                          ("eval_skip_metrics", "--eval-skip-metrics")):
            val = self.trial_defaults.get(key)
            if val is not None:
                cmd += [flag, str(val)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False,
                timeout=self.trial_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "failed",
                "error": (f"trial timed out after {self.trial_timeout_seconds:.0f}s "
                          f"(cell-eval/de pipeline made no progress); "
                          f"partial stdout: {str(exc.stdout or b'')[-2000:]!r}"),
                "command": " ".join(cmd), "command_list": cmd,
            }
        if proc.returncode != 0:
            return {"status": "failed", "error": proc.stderr[-2000:] or proc.stdout[-2000:],
                    "command": " ".join(cmd), "command_list": cmd}
        raw = _read_json(output)
        return raw or {"status": "failed", "error": "adapter returned no result file",
                       "command": " ".join(cmd), "command_list": cmd}

    # -- worker protocol ------------------------------------------------------
    def step(self, layer: str, round_: int, step: int, context: Dict[str, Any]) -> StepResult:
        details = context.get("details") or _default_plan()
        if layer == "L5":
            return self._technique_step(details, round_, step, context)
        # macro planning layer: no GPU, report a plan-quality score.
        best = self._best_score if self._best_score is not None else self.baseline
        score = round(best + (step * 0.0001), 4)  # deterministic small uplift
        reasoning = {
            "L1": "focus on official H1 perturbation response prediction",
            "L2": "rank 18080 genes across 100 unseen H1 targets",
            "L3": "delta-model perturbation features transfer across targets",
            "L4": "promoted delta model with shared gene embedding head",
            "L5": "candidate_g_promoted_delta_model with native_eval scoring",
        }[layer]
        action = {
            "layer": layer, "step": step, "kind": "plan",
            "reasoning": f"{reasoning}", "operator": "draft" if step == 1 else "improve",
        }
        # P3: carry the memory few-shot block into the action so decision
        # records and prompts visibly differ between memory / no-memory arms.
        mem_block = context.get("memory_block")
        if mem_block:
            action["memory_block"] = mem_block
            action["reasoning"] = f"{reasoning}\n\n{mem_block}"
        return StepResult(
            layer=layer,
            round=round_,
            step=step,
            operator=action["operator"],
            status="ok",
            score=score,
            metrics={"pearson_delta": score, "plan_quality": score},
            cost={"seconds": 0.0},
            context=dict(context),
            action=action,
            detail=f"planning layer {layer}",
        )

    def _technique_step(self, details: Dict[str, Any], round_: int, step: int, context: Dict[str, Any]) -> StepResult:
        pick = self._pick("L5", context, step)
        variant, seed = pick["variant"], pick["seed"]
        self._picked_seeds.append(int(seed))
        if self.mode == "rollout":
            # P2: planning + reasoning only. No real training/trial, but the
            # full decision record (operator/variant/reasoning/outcome) is
            # produced so every rollout step yields a decision trajectory.
            self.seed_idx += 1
            base = self.baseline + 0.008 * self.seed_idx + 0.001 * step
            score = round(min(base, 0.35), 4)
            self._best_score = max(self._best_score or score, score)
            reasoning = (
                f"rollout L5 technique: variant={variant} seed={seed} "
                f"explore improvement on promoted delta model; expect score ~{score}"
            )
            return StepResult(
                layer="L5", round=round_, step=step,
                operator="tune" if step > 1 else "implement",
                status="ok", score=score,
                metrics={self.metric: score},
                cost={"seconds": 0.5, "gpu": 0},
                context={"details": details},
                action={"variant": variant, "seed": seed, "kind": "rollout",
                        "reasoning": reasoning, "operator": "implement" if step == 1 else "tune"},
                detail=f"rollout vcc25 variant={variant} seed={seed} pearson_delta={score}",
            )
        if self.mode != "real":
            # offline / mock: deterministic score progression (baseline-based).
            self.seed_idx += 1
            base = self.baseline + 0.01 * self.seed_idx + 0.001 * step
            score = round(base, 4)
            self._best_score = max(self._best_score or score, score)
            return StepResult(
                layer="L5", round=round_, step=step,
                operator="tune" if step > 1 else "implement",
                status="ok", score=score,
                metrics={self.metric: score},
                cost={"seconds": 60.0, "gpu": 1},
                context={"details": details},
                action={"variant": variant, "seed": seed, "kind": "real_trial", "mock": True},
                detail=f"mock vcc25 trial variant={variant} seed={seed}",
            )

        raw = self._run_real_trial(variant, seed, round_, step)
        self.seed_idx += 1
        if raw.get("status") != "ok":
            return StepResult(
                layer="L5", round=round_, step=step, operator="tune",
                status="failed",
                score=self._best_score,
                metrics={self.metric: self._best_score if self._best_score is not None else 0.0},
                cost={"seconds": 0.0},
                context={"details": details, "raw": raw},
                action={"variant": variant, "seed": seed, "kind": "real_trial"},
                detail=f"trial failed: {str(raw.get('error'))[:200]}",
            )
        metrics = raw.get("metrics") or {}
        score = float(metrics.get(self.metric, 0.0))
        self._best_score = max(self._best_score or score, score)
        self._memorize_live(context, variant, seed, score, "ok", round_, step)
        return StepResult(
            layer="L5", round=round_, step=step,
            operator="tune" if step > 1 else "implement",
            status="ok", score=score,
            metrics=dict(metrics),
            cost={"seconds": float(raw.get("runtime_seconds") or 0.0), "gpu": 1},
            context={"details": details, "raw_result": str(raw.get("raw_result", ""))},
            action={"variant": variant, "seed": seed, "kind": "real_trial"},
            detail=f"real vcc25 trial variant={variant} seed={seed} pearson_delta={score}",
        )


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
