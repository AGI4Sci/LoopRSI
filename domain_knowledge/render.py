"""Bounded, provenance-preserving rendering for prompt injection."""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .cards import KnowledgeCard


@dataclass(frozen=True)
class RenderedKnowledge:
    content: str
    card_ids: Tuple[str, ...]
    estimated_tokens: int
    content_hash: str


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / 4)


def _join(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "; ".join(str(item) for item in value)
    return str(value)


def _render_card(card: KnowledgeCard) -> str:
    lines = [
        f"card_id: {card.id}",
        f"asset_type: {card.asset_type}",
        f"title: {card.title}",
        f"review_status: {card.review_status}",
        f"review_basis: {card.review_basis}",
        f"summary: {card.summary_plain}",
    ]
    fit_reason = card.get("fit_reason")
    if fit_reason:
        lines.append(f"fit_reason: {fit_reason}")
    if card.asset_type == "paper":
        if card.get("mechanism"):
            lines.append(f"mechanism: {card.get('mechanism')}")
        if card.get("limitations"):
            lines.append(f"limitations: {_join(card.get('limitations'))}")
    elif card.asset_type == "repository":
        lines.append(f"revision: {card.get('revision')}")
        lines.append(f"entrypoints: {_join(card.get('entrypoints', ())) }")
        if card.get("limitations"):
            lines.append(f"limitations: {_join(card.get('limitations'))}")
    elif card.asset_type == "model":
        if card.get("research_role"):
            lines.append(f"research_role: {card.get('research_role')}")
        lines.append(f"execution_readiness: {card.execution_readiness}")
        recipe: Mapping[str, Any] = card.get("execution_recipe", {})
        lines.append(f"adapter_id: {recipe.get('adapter_id')}")
        lines.append(f"actions: {_join(recipe.get('actions', ())) }")
        compatibility: Mapping[str, Any] = card.get("vcc25_compatibility", {})
        lines.append(f"compatibility: {compatibility.get('level')}")
        lines.append(f"gaps: {_join(compatibility.get('gaps', ())) }")
        evidence = card.get("smoke_evidence", ())
        for item in evidence:
            if isinstance(item, Mapping):
                lines.append(
                    "smoke_evidence: "
                    f"{item.get('status', 'unknown')} "
                    f"({item.get('authority', 'unknown')}); "
                    f"source={item.get('source_manifest', 'unspecified')}"
                )
    for source in card.sources:
        lines.append(f"source: {source['url']}")
    return "\n".join(lines)


def render_cards(cards: Sequence[KnowledgeCard], token_budget: int) -> RenderedKnowledge:
    """Render only complete cards that fit; provenance is never truncated."""
    if token_budget < 0:
        raise ValueError("token_budget must not be negative")
    blocks = []
    card_ids = []
    used = 0
    for card in cards:
        block = _render_card(card)
        block_tokens = _estimate_tokens(block)
        if used + block_tokens > token_budget:
            continue
        blocks.append(block)
        card_ids.append(card.id)
        used += block_tokens
    content = "\n\n---\n\n".join(blocks)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return RenderedKnowledge(
        content=content,
        card_ids=tuple(card_ids),
        estimated_tokens=used,
        content_hash=digest,
    )
