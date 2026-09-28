"""Deterministic prompt composition with injection provenance."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping

from .plugin_protocols import PromptFragment


@dataclass(frozen=True)
class ComposedPrompt:
    text: str
    fragments: tuple[PromptFragment, ...]


class PromptComposer:
    def compose(self, fragments: Iterable[PromptFragment], context_budget: int | None = None) -> ComposedPrompt:
        ordered = tuple(sorted(fragments, key=lambda item: (item.priority, item.skill_id, item.stage)))
        selected: list[PromptFragment] = []
        used = 0
        for fragment in ordered:
            budget = fragment.token_budget or len(fragment.content.split())
            if context_budget is not None and used + budget > context_budget:
                continue
            if not fragment.content_hash:
                fragment = PromptFragment(
                    skill_id=fragment.skill_id,
                    stage=fragment.stage,
                    content=fragment.content,
                    priority=fragment.priority,
                    token_budget=fragment.token_budget,
                    activation_reason=fragment.activation_reason,
                    source=fragment.source,
                    content_hash=sha256(fragment.content.encode()).hexdigest(),
                )
            selected.append(fragment)
            used += budget
        return ComposedPrompt("\n\n".join(item.content for item in selected), tuple(selected))
