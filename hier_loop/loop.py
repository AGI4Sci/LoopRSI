"""HierarchicalLoop orchestrator for the Hierarchical Looped RSI Researcher.

Runs a contiguous ``loop_range`` of layers from macro to micro.  Each layer is an
inner loop with its own max steps and exit rule.  On exit the layer emits an
``exit`` message and an experience card.  The card is routed onward:

- by default to the next **adjacent** in-range layer, or
- directly to any layer named by a whitelisted ``direct_whitelist`` edge whose
  source is the current layer (this is the cross-layer direct connect, which may
  skip over layers or reach a layer outside the loop range).

Every message is appended to an append-only JSONL manifest so any run is
replayable.  Pure stdlib, Python 3.10+.
"""

from __future__ import annotations

import json
import os
import random
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import decisions as dec
from . import layers
from . import experience as exp
from . import messages as msg
from . import routing
from .exit_rules import ExitDecision, evaluate_exit


@dataclass
class StepResult:
    """One worker step for one layer inside a loop round."""

    layer: str
    round: int
    step: int
    operator: str
    status: str
    score: Optional[float]
    metrics: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_step_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "round": self.round,
            "step": self.step,
            "operator": self.operator,
            "status": self.status,
            "score": self.score,
            "metrics": dict(self.metrics),
            "cost": dict(self.cost),
            "context": dict(self.context),
            "action": dict(self.action),
            "detail": self.detail,
        }


class LoopWorker:
    """Worker protocol: one ``step`` invocation per layer-round-step."""

    def step(self, layer: str, round_: int, step: int, context: Dict[str, Any]) -> StepResult:
        raise NotImplementedError


class DeterministicWorker(LoopWorker):
    """Deterministic worker that reports a monotonically improving score unless
    ``improves=False``.  Used for local dry-runs and unit tests."""

    def __init__(self, seed: int = 0, improves: bool = True):
        self.rng = random.Random(seed)
        self.improves = improves

    def step(self, layer: str, round_: int, step: int, context: Dict[str, Any]) -> StepResult:
        base = {"direction": 0.30, "problem": 0.31, "hypothesis": 0.32, "mechanism": 0.33, "technique": 0.34}
        seed = base.get(layer, 0.30)
        if self.improves:
            score = round(min(seed + step * 0.005, 0.40), 4)
        else:
            score = round(seed - step * 0.005, 4)
        return StepResult(
            layer=layer,
            round=round_,
            step=step,
            operator="draft" if step == 1 else "improve",
            status="ok",
            score=score,
            metrics={"pearson_delta": score},
            cost={"seconds": step * 30.0},
            context=dict(context),
            action={"layer": layer, "step": step},
            detail="deterministic step",
        )


@dataclass
class LoopSummary:
    run_id: str
    loop_range: List[str]
    rounds: int
    messages: List[Dict[str, Any]] = field(default_factory=list)
    route_decisions: List[Dict[str, Any]] = field(default_factory=list)
    cards: List[Dict[str, Any]] = field(default_factory=list)
    per_layer: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    best_score: Optional[float] = None
    best_layer: Optional[str] = None
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "rsi.hier-summary/v1",
            "run_id": self.run_id,
            "loop_range": list(self.loop_range),
            "rounds": self.rounds,
            "messages": list(self.messages),
            "route_decisions": list(self.route_decisions),
            "cards": list(self.cards),
            "per_layer": dict(self.per_layer),
            "errors": list(self.errors),
            "best_score": self.best_score,
            "best_layer": self.best_layer,
            "memory_usage": dict(self.memory_usage),
            "created_at": self.created_at,
        }


class HierarchicalLoop:
    """Top-level orchestrator.

    ``config`` shape::

        {
          "run": {"id": ..., "seed": 42, "rounds": N, "task_id": "vcc25"},
          "loop_range": {"start": "L3", "end": "L5"},
          "direct_whitelist": [{"from": "L5", "to": "L1"}, ...],
          "loops": {
             "L3": {"max_steps": 4, "exit": {"type": "converge_or_budget", ...}},
             ...
          },
          "router": {"default_route": "adjacent"},
          "reward": {"metric": "pearson_delta", "baseline": 0.30614},
          "out": {"dir": ..., "manifest": "hier_progress.jsonl"}
        }
    """

    def __init__(
        self,
        config: Dict[str, Any],
        worker: Optional[LoopWorker] = None,
        run_id: Optional[str] = None,
        decision_writer: Optional[dec.DecisionWriter] = None,
        card_repo: Optional[exp.CardRepository] = None,
        memory_manager: Optional[Any] = None,
    ):
        self.config = config
        run_cfg = config.get("run") or {}
        self.worker = worker or DeterministicWorker(seed=int(run_cfg.get("seed", 0)))
        self.run_id = run_id or run_cfg.get("id") or f"hier_{int(_time.time() * 1000)}"
        self.seed = int(run_cfg.get("seed", 42))
        self.rounds = max(1, int(run_cfg.get("rounds", 1)))

        self.loop_range = layers.resolve_loop_range(config.get("loop_range"))
        self.table = routing.build_routing_table(config.get("direct_whitelist"))

        # P2/P3 hooks
        self.decision_writer = decision_writer
        self.card_repo = card_repo
        self.memory_manager = memory_manager
        # memory block (defaults to disabled => fully backward-compatible)
        mem_cfg = config.get("memory") or {}
        self.memory_enabled = bool(mem_cfg.get("enabled", False))
        self.memory_baseline = float(mem_cfg.get("baseline") or 0.050227)
        mem_cfg.setdefault("capacity_threshold", 0)
        self.memory_capacity = int(mem_cfg.get("capacity_threshold") or 0)
        mem_ret = mem_cfg.get("retrieval") or {}
        self.memory_top_k = int(mem_ret.get("top_k", 5))
        self.memory_usage: Dict[str, Any] = {"layer_hits": {}, "n_memory_injections": 0, "queries_total": 0, "hits_total": 0}

        raw_loops = config.get("loops") or {}
        self.loop_cfg: Dict[str, Dict[str, Any]] = {}
        for lid in self.loop_range.ids():
            raw = raw_loops.get(lid) or {}
            if "max_steps" not in raw:
                raw = {"max_steps": 3, "exit": {"type": "budget", "budget_seconds": 600}}
            self.loop_cfg[lid] = dict(raw)
        self.out_cfg = dict(config.get("out") or {})
        self.summary = LoopSummary(
            run_id=self.run_id, loop_range=self.loop_range.ids(), rounds=self.rounds
        )
        self._msg_counter = 0

    # -- persistence ---------------------------------------------------------
    def _out_dir(self) -> str:
        return self.out_cfg.get("dir", "output/rsi_step0/hier")

    def _manifest_path(self) -> str:
        return os.path.join(self._out_dir(), self.out_cfg.get("manifest", "hier_progress.jsonl"))

    def _append(self, record: Dict[str, Any]) -> None:
        os.makedirs(self._out_dir(), exist_ok=True)
        with open(self._manifest_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")

    # -- routing -------------------------------------------------------------
    def route(self, src: str, dst: str) -> routing.RouteDecision:
        d = self.table.resolve(src, dst)
        self.summary.route_decisions.append(d.to_dict())
        return d

    def _direct_targets(self, src: str) -> List[str]:
        """Layers reachable from ``src`` via whitelisted direct edges."""
        targets = []
        for (s, d) in sorted(self.table.direct_edges):
            if s == src:
                targets.append(d)
        return targets

    # -- core ----------------------------------------------------------------
    def _emit(self, m: msg.Msg) -> Dict[str, Any]:
        m.validate()
        d = m.to_dict()
        self._msg_counter += 1
        d["_seq"] = self._msg_counter
        self.summary.messages.append(d)
        self._append(d)
        return d

    def _new_msg(
        self,
        from_layer: str,
        to_layer: str,
        kind: str,
        round_: int,
        hop: int,
        route: str = "adjacent",
        payload: Optional[msg.Payload] = None,
        parent_msg_id: Optional[str] = None,
        exit_reason: Optional[str] = None,
    ) -> msg.Msg:
        return msg.make_msg(
            from_layer=from_layer,
            to_layer=to_layer,
            kind=kind,
            round_=round_,
            hop=hop,
            route=route,
            payload=payload or msg.Payload(),
            parent_msg_id=parent_msg_id,
            exit_reason=exit_reason,
        )

    def _route_experience(self, lid: str, parent_id: str, hop: int, cards: Dict[str, Any], round_: int) -> List:
        """Emit the experience-card transfer for a completed layer.

        - Adjacent next in-range layer gets an ``adjacent`` transfer.
        - Every whitelisted direct edge out of ``lid`` gets a ``direct`` transfer
          (may skip layers or reach a layer outside the loop range).
        A blocked route edges are recorded (not silently dropped).

        Returns a list of ``(msg_dict, target_layer)`` for the emitted transfers
        so callers can persist decision records for transfer messages too.
        """
        emitted: List = []
        # 1) adjacent within-range flow
        next_layer = self.loop_range.next_in_range(lid)
        if next_layer is not None:
            d = self.route(lid, next_layer)
            m = self._emit(
                self._new_msg(
                    from_layer=lid,
                    to_layer=next_layer,
                    kind="transfer",
                    round_=round_,
                    hop=hop,
                    route=d.route,
                    payload=msg.Payload(context={"experience_cards": cards}),
                    parent_msg_id=parent_id,
                )
            )
            emitted.append((m, next_layer))

        # 2) whitelisted direct edges (cross-layer / out-of-range)
        for dst in self._direct_targets(lid):
            d = self.route(lid, dst)
            if not d.allowed:
                self.summary.errors.append(f"direct edge {lid}->{dst} blocked: {d.reason}")
                continue
            m = self._emit(
                self._new_msg(
                    from_layer=lid,
                    to_layer=dst,
                    kind="transfer",
                    round_=round_,
                    hop=hop,
                    route="direct",
                    payload=msg.Payload(context={"experience_cards": cards, "direct": True}),
                    parent_msg_id=parent_id,
                )
            )
            emitted.append((m, dst))
        return emitted

    def _is_real_worker(self) -> bool:
        """True when the active worker performs real native_eval (L5 real mode)."""
        w = getattr(self.worker, "mode", "offline")
        return w == "real"

    def _reward_for(self, score: Optional[float]) -> Optional[float]:
        """Reward = improvement over the empirical memory baseline.

        Computed whenever a baseline is configured (memory block present), so
        rollout decisions carry a reward for P3 memory building even before the
        memory index is enabled for inference.
        """
        if score is None:
            return None
        return round(float(score) - self.memory_baseline, 6)

    def _inject_memory(self, context: Dict[str, Any], layer: str) -> Dict[str, Any]:
        """Add a memory prompt block for ``layer`` to the step context.

        Graceful: returns an unmodified context when memory is disabled, the
        manager/index is missing, or retrieval has no hit for this fingerprint.
        """
        if not self.memory_enabled or self.memory_manager is None:
            return context
        fp = dec.context_fingerprint(context)
        ct = dec.context_text(context)
        self.memory_usage["queries_total"] = self.memory_usage.get("queries_total", 0) + 1
        try:
            block = self.memory_manager.render_prompt_block(
                layer,
                context=ct,
                fingerprint=fp,
                top_k=self.memory_top_k,
            )
        except Exception:
            return context
        usage = self.memory_manager.usage()
        for key in ("embedding_backend", "min_similarity", "similarity",
                    "semantic_queries", "semantic_hits", "fp_queries", "fp_hits"):
            if key in usage:
                self.memory_usage[key] = usage[key]
        if not block:
            return context
        ctx = dict(context)
        ctx["memory_prompt"] = block
        ctx["memory_block"] = block
        ctx["memory_used"] = True
        ctx["memory_hits"] = int(self.memory_top_k)
        ctx["memory_fingerprint"] = fp
        layer_hits = self.memory_usage.setdefault("layer_hits", {})
        layer_hits[layer] = layer_hits.get(layer, 0) + 1
        self.memory_usage["n_memory_injections"] = self.memory_usage.get("n_memory_injections", 0) + 1
        self.memory_usage["hits_total"] = self.memory_usage.get("hits_total", 0) + 1
        return ctx

    def _record_decision(
        self,
        m: Dict[str, Any],
        layer: str,
        reward: Optional[float] = None,
        card_id: Optional[str] = None,
    ) -> None:
        """Append one decision record when a decision writer is configured."""
        if self.decision_writer is None:
            return
        try:
            if reward is None:
                payload = m.get("payload") or {}
                outcome = payload.get("outcome") or {}
                metrics = outcome.get("metrics") or {}
                sc = outcome.get("best_score")
                if sc is None:
                    sc = outcome.get("score")
                if sc is None:
                    sc = metrics.get((self.config.get("reward") or {}).get("metric", "pearson_delta"))
                reward = self._reward_for(sc)
            metric_source = "real" if (layer == "L5" and self._is_real_worker()) else "plan_proxy"
            d = dec.make_decision(
                m,
                run_id=self.run_id,
                reward=reward,
                metric=(self.config.get("reward") or {}).get("metric", "pearson_delta"),
                card_id=card_id,
                metric_source=metric_source,
            )
            self.decision_writer.write(d)
        except Exception:
            pass

    def run(self) -> LoopSummary:
        start = _time.time()
        active_layers = self.loop_range.ids()
        context: Dict[str, Any] = {
            "task_id": (self.config.get("run") or {}).get("task_id", "vcc25"),
            "seed": self.seed,
            "baseline": (self.config.get("reward") or {}).get("baseline"),
            "metric": (self.config.get("reward") or {}).get("metric", "pearson_delta"),
            "active_layers": list(active_layers),
        }

        parent_id: Optional[str] = None
        hop = 0
        for round_no in range(self.rounds):
            hop += 1
            # Round-entry message carries the accumulated task context forward.
            entry = self._new_msg(
                from_layer=active_layers[0],
                to_layer=active_layers[0],
                kind="step",
                round_=round_no,
                hop=hop,
                payload=msg.Payload(
                    context={"task": context, "round": round_no, "entry": True},
                    action={}, outcome={}, cost={}, time={},
                ),
            )
            self._emit(entry)
            parent_id = entry.msg_id

            for lid in active_layers:
                hop += 1
                layer_summary: Dict[str, Any] = {
                    "layer": lid,
                    "round": round_no,
                    "n_steps": 0,
                    "best_score": None,
                    "exit_reason": None,
                    "exit_detail": None,
                }
                cfg = self.loop_cfg[lid]
                max_steps = int(cfg.get("max_steps", 3))
                exit_rule = cfg.get("exit") or {}
                metric_history: List[float] = []
                step_count = 0
                decision: Optional[ExitDecision] = None

                for step in range(1, max_steps + 1):
                    step_ctx = self._inject_memory(context, lid)
                    step_result = self.worker.step(lid, round_no, step, step_ctx)
                    step_count += 1
                    layer_summary["n_steps"] = step_count
                    score = step_result.score
                    if score is not None:
                        metric_history.append(float(score))
                        layer_summary["best_score"] = max(
                            layer_summary["best_score"] if layer_summary["best_score"] is not None else -1e18,
                            float(score),
                        )
                    payload = msg.Payload(
                        context=dict(step_ctx),
                        action=dict(step_result.action),
                        outcome={"status": step_result.status, "metrics": dict(step_result.metrics)},
                        cost=dict(step_result.cost),
                        time={"seconds_since_start": round(_time.time() - start, 3)},
                    )
                    step_msg = self._emit(
                        self._new_msg(
                            from_layer=lid,
                            to_layer=lid,
                            kind="step",
                            round_=round_no,
                            hop=hop,
                            payload=payload,
                            parent_msg_id=parent_id,
                        )
                    )
                    parent_id = step_msg["msg_id"]
                    self._record_decision(step_msg, lid)

                    decision = evaluate_exit(
                        exit_rule,
                        step_result.to_step_dict(),
                        max_steps=max_steps,
                        steps_done=step_count,
                        metric_history=metric_history,
                    )
                    if decision.triggered:
                        break

                if decision is None or not decision.triggered:
                    decision = ExitDecision(True, "budget", f"max_steps={max_steps} exhausted")

                layer_summary["exit_reason"] = decision.reason
                layer_summary["exit_detail"] = decision.detail

                # Track the per-layer aggregate across rounds, preserving each
                # round's own best score / exit so the run is fully auditable.
                agg = self.summary.per_layer.setdefault(
                    lid, {"layer": lid, "rounds": [], "best_score": None, "best_round": None, "n_steps": 0}
                )
                agg["rounds"].append(
                    {"round": round_no, "best_score": layer_summary["best_score"], "exit_reason": decision.reason}
                )
                agg["n_steps"] += step_count
                if layer_summary["best_score"] is not None and (
                    agg["best_score"] is None or layer_summary["best_score"] > agg["best_score"]
                ):
                    agg["best_score"] = layer_summary["best_score"]
                    agg["best_round"] = round_no

                if layer_summary["best_score"] is not None and (
                    self.summary.best_score is None
                    or layer_summary["best_score"] > self.summary.best_score
                ):
                    self.summary.best_score = layer_summary["best_score"]
                    self.summary.best_layer = lid

                # exit message + experience card
                exit_msg = self._new_msg(
                    from_layer=lid,
                    to_layer=lid,
                    kind="exit",
                    round_=round_no,
                    hop=hop,
                    payload=msg.Payload(
                        context=dict(context),
                        action={},
                        outcome={"exit": decision.to_dict(), "best_score": layer_summary["best_score"]},
                        cost={},
                        time={"seconds_since_start": round(_time.time() - start, 3)},
                    ),
                    parent_msg_id=parent_id,
                    exit_reason=decision.reason,
                )
                exit_msg = self._emit(exit_msg)
                parent_id = exit_msg["msg_id"]
                self._record_decision(exit_msg, lid)

                # metric_source: L5 (technique) is a real native_eval outcome in
                # real mode; macro planning layers L1..L4 are plan-quality proxies.
                metric_source = "real" if (lid == "L5" and self._is_real_worker()) else "plan_proxy"
                delta = (metric_history[-1] - metric_history[-2]) if len(metric_history) >= 2 else None
                reward = self._reward_for(layer_summary["best_score"])
                card = exp.new_experience_card(
                    layer=lid,
                    summary=f"{lid}:{layers.layer_names()[lid]} round{round_no} exit({decision.reason}) best_score={layer_summary['best_score']}",
                    metric=(self.config.get("reward") or {}).get("metric"),
                    metric_value=layer_summary["best_score"],
                    delta=delta,
                    evidence_msg_ids=[self.summary.messages[-1]["msg_id"]],
                    cost={"reward": reward} if reward is not None else {},
                    exit_reason=decision.reason,
                    metric_source=metric_source,
                )
                if self.card_repo is not None:
                    self.card_repo.append(card)
                card_dict = card.to_dict()
                self.summary.cards.append(card_dict)
                context["experience_cards"] = exp.rollup_cards(
                    [exp.ExperienceCard.from_dict(c) for c in self.summary.cards]
                )
                # internalise signal when per-layer card count reaches capacity
                if self.memory_capacity > 0:
                    layer_cards = (
                        self.card_repo.load(lid) if self.card_repo is not None else []
                    )
                    sig = exp.capacity_signal(layer_cards, self.memory_capacity)
                    context.setdefault("internalise_signals", []).append(
                        {"layer": lid, **sig}
                    )

                transfer_decision = self._route_experience(
                    lid, parent_id, hop, context["experience_cards"], round_no
                )
                for tmsg, layer in transfer_decision:
                    self._record_decision(tmsg, layer)

        self.summary.created_at = _time.strftime("%Y-%m-%dT%H:%M:%S%z")
        mu = dict(self.memory_usage)
        queries = int(mu.get("queries_total") or 0)
        hits = int(mu.get("hits_total") or 0)
        mu["hit_rate"] = round(hits / queries, 4) if queries else 0.0
        self.summary.memory_usage = mu
        self._write_summary()
        return self.summary

    def _write_summary(self) -> None:
        os.makedirs(self._out_dir(), exist_ok=True)
        with open(os.path.join(self._out_dir(), "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.summary.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    def validate(self) -> List[str]:
        """Return a list of configuration errors (empty = valid)."""
        errors: List[str] = []
        try:
            self.loop_range = layers.resolve_loop_range(self.config.get("loop_range"))
        except ValueError as exc:
            errors.append(str(exc))
        try:
            self.table = routing.build_routing_table(self.config.get("direct_whitelist"))
        except ValueError as exc:
            errors.append(str(exc))
        return errors
