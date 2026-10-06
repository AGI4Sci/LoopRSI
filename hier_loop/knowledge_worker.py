"""Knowledge-driven five-layer worker that plans without fabricating scores."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .knowledge_bridge import KnowledgeBridge
from .layer_agent import (
    LayerDecisionBackend,
    LayerDecisionRequest,
    build_layer_prompt,
    validate_layer_decision,
)
from .loop import LoopWorker, StepResult


class KnowledgeDrivenVcc25Worker(LoopWorker):
    def __init__(self, bridge: KnowledgeBridge, backend: LayerDecisionBackend) -> None:
        self.bridge = bridge
        self.backend = backend
        self._decisions: list[dict[str, Any]] = []

    def step(self, layer: str, round_: int, step: int, context: Dict[str, Any]) -> StepResult:
        knowledge = self.bridge.inject(layer, context)
        prior = tuple(dict(decision) for decision in self._decisions)
        prompt = build_layer_prompt(layer, round_, step, knowledge, context, prior)
        request = LayerDecisionRequest(
            layer=layer,
            round=round_,
            step=step,
            knowledge=knowledge,
            context=dict(context),
            prior_decisions=prior,
            prompt=prompt,
        )
        decision = validate_layer_decision(layer, self.backend.decide(request))
        self._decisions.append(decision)
        provenance = asdict(knowledge)
        provenance["card_ids"] = list(knowledge.card_ids)
        action = {
            "layer": layer,
            "step": step,
            "kind": "knowledge_decision",
            "decision": decision,
            "knowledge": provenance,
            "prior_decision_count": len(prior),
            "evaluation_authority": None,
        }
        return StepResult(
            layer=layer,
            round=round_,
            step=step,
            operator="draft" if step == 1 else "improve",
            status="ok",
            score=None,
            metrics={},
            cost={"seconds": 0.0, "gpu": 0},
            context=dict(context),
            action=action,
            detail=f"knowledge-driven {layer} decision",
        )
