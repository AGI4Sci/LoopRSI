"""Experience cards: layer exit summaries that are explicitly accumulated and,
at a capacity threshold, signal internalisation into the trained router.

A card is written when a layer loop exits.  ``rollup_cards`` feeds the cards of
a completed layer into the next (more macro) layer's context so the upper layer
sees evidence instead of only the decoded summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

METRIC_SOURCES = ("real", "plan_proxy")


@dataclass
class ExperienceCard:
    card_id: str
    layer: str
    summary: str
    metric: Optional[str]
    metric_value: Optional[float]
    delta: Optional[float]
    evidence_msg_ids: List[str] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    exit_reason: Optional[str] = None
    metric_source: str = "plan_proxy"
    created_at: Optional[str] = None


    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ExperienceCard":
        return cls(
            card_id=raw["card_id"],
            layer=raw["layer"],
            summary=raw.get("summary", ""),
            metric=raw.get("metric"),
            metric_value=raw.get("metric_value"),
            delta=raw.get("delta"),
            evidence_msg_ids=list(raw.get("evidence_msg_ids") or []),
            cost=dict(raw.get("cost") or {}),
            exit_reason=raw.get("exit_reason"),
            metric_source=raw.get("metric_source", "plan_proxy"),
            created_at=raw.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "schema": "rsi.experience.v1",
            "card_id": self.card_id,
            "layer": self.layer,
            "summary": self.summary,
            "metric": self.metric,
            "metric_value": self.metric_value,
            "delta": self.delta,
            "evidence_msg_ids": list(self.evidence_msg_ids),
            "cost": dict(self.cost),
            "exit_reason": self.exit_reason,
            "metric_source": self.metric_source,
            "created_at": self.created_at or _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }


def new_experience_card(
    layer: str,
    summary: str,
    metric: Optional[str] = None,
    metric_value: Optional[float] = None,
    delta: Optional[float] = None,
    evidence_msg_ids: Optional[List[str]] = None,
    cost: Optional[Dict[str, Any]] = None,
    exit_reason: Optional[str] = None,
    metric_source: str = "plan_proxy",
) -> ExperienceCard:
    basis = f"{layer}|{metric_value}|{len(evidence_msg_ids or [])}|{_time.time()}"
    card_id = "card_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return ExperienceCard(
        card_id=card_id,
        layer=layer,
        summary=summary,
        metric=metric,
        metric_value=metric_value,
        delta=delta,
        evidence_msg_ids=list(evidence_msg_ids or []),
        cost=dict(cost or {}),
        exit_reason=exit_reason,
        metric_source=metric_source if metric_source in METRIC_SOURCES else "plan_proxy",
    )


def rollup_cards(cards: List[ExperienceCard]) -> Dict[str, Any]:
    """Roll a layer's cards into a compact, evidence-carrying context block."""
    return {
        "n_cards": len(cards),
        "layers": sorted({c.layer for c in cards}),
        "best": max((c.metric_value or 0.0 for c in cards), default=0.0),
        "cards": [c.to_dict() for c in cards],
    }


def capacity_signal(cards: List[ExperienceCard], capacity: int) -> Dict[str, Any]:
    """Return an internalise signal when the explicit card count reaches capacity."""
    return {
        "internalise": len(cards) >= int(capacity) if capacity else False,
        "capacity": int(capacity or 0),
        "n_cards": len(cards),
        "action": "internalise" if (capacity and len(cards) >= int(capacity)) else "accumulate",
    }

# ---------------------------------------------------------------------------
# Per-layer experience-card repository (P1).
# ---------------------------------------------------------------------------
class CardRepository:
    """Append-only per-layer experience-card repo: ``out/experience_cards/<layer>.jsonl``."""

    def __init__(self, root: str):
        self.root = root

    def path(self, layer: str) -> str:
        return os.path.join(self.root, f"{layer}.jsonl")

    def append(self, card: ExperienceCard) -> ExperienceCard:
        os.makedirs(self.root, exist_ok=True)
        with open(self.path(card.layer), "a", encoding="utf-8") as f:
            f.write(json.dumps(card.to_dict(), ensure_ascii=False) + "\n")
        return card

    def load(self, layer: str) -> List[ExperienceCard]:
        if not os.path.isfile(self.path(layer)):
            return []
        cards: List[ExperienceCard] = []
        with open(self.path(layer), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cards.append(ExperienceCard.from_dict(json.loads(line)))
                except Exception:
                    continue
        return cards

    def count(self, layer: str) -> int:
        return len(self.load(layer))
