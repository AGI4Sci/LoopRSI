"""Small reusable base for domain-specific research knowledge skills."""

from __future__ import annotations

import hashlib
import json
import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai4ai.contracts import SkillContext, SkillResult
from ai4ai.plugin_protocols import PromptFragment, ValidationResult
from ai4ai.skill import Skill


class ResearchKnowledgeSkill(Skill):
    """Load a declared JSON knowledge file and return a stable SkillResult.

    Domain adapters only provide ``knowledge_filename`` and ``next_actions``;
    task-specific knowledge remains outside this generic layer.
    """

    expected_task = ""
    knowledge_package_name = "vcc25"
    knowledge_filename = ""
    source = ""
    next_actions: tuple[dict[str, Any], ...] = ()

    _knowledge_loads = 0

    def __init__(self, knowledge_root: str | Path | None = None) -> None:
        # ``knowledge_root`` is retained as a compatibility hook for the
        # original ResearchSkill protocol.  The new SkillContext API uses the
        # package-owned knowledge files by default.
        self.knowledge_root = Path(knowledge_root) if knowledge_root else None

    @property
    def skill_id(self) -> str:
        return self.manifest.skill_id

    def can_activate(self, context: dict[str, Any]) -> bool:
        return context.get("task_id") == self.expected_task and context.get("layer") in self.layers

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(getattr(self, "supported_layers", ("L1", "L2", "L3", "L4", "L5")))

    def inject(self, context: dict[str, Any]) -> PromptFragment | None:
        if not self.can_activate(context):
            return None
        from domain_knowledge import KnowledgeQuery, KnowledgeStore
        from domain_knowledge.render import render_cards
        from domain_knowledge.store import assert_safe_knowledge

        root = self.knowledge_root or Path(__file__).resolve().parents[1] / "knowledge" / "vcc25"
        assert_safe_knowledge(context, "research_skill_context")
        store = KnowledgeStore.from_directory(root)
        layer = context["layer"]
        asset_types = ("paper",) if layer in {"L1", "L2"} else (("model", "repository") if layer == "L5" else ("paper", "repository", "model"))
        cards = list(store.query(KnowledgeQuery(task_id=self.expected_task, layer=layer, text=str(context.get("query", "")), asset_types=asset_types, max_results=100)))
        if layer == "L5" and int(context.get("token_budget", 1200)) >= 1500:
            # L5 needs both executable model facts and the repository that
            # supplies the implementation. Preserve repository-before-model
            # provenance for the legacy ResearchSkill contract.
            related = []
            for card in cards[:2]:
                for relation in card.relations:
                    if relation.startswith("kb:repo:"):
                        related.extend(store.query(KnowledgeQuery(task_id=self.expected_task, layer=layer, text=relation.split(":")[-1], asset_types=("repository",), max_results=1)))
            seen = set()
            cards = [card for card in related + cards if not (card.id in seen or seen.add(card.id))]
        rendered = render_cards(cards, int(context.get("token_budget", 1200)))
        if not rendered.card_ids:
            return None
        return PromptFragment(self.skill_id, layer, rendered.content, token_budget=int(context.get("token_budget", 1200)), source=f"knowledge:vcc25:{','.join(rendered.card_ids)}", content_hash=rendered.content_hash)

    def validate_proposal(self, proposal: dict[str, Any], context: dict[str, Any]) -> ValidationResult:
        from domain_knowledge.store import assert_safe_knowledge
        try:
            assert_safe_knowledge(proposal, "research_skill_proposal")
        except Exception as exc:
            return ValidationResult(False, errors=(str(exc),))
        return ValidationResult(True)

    @staticmethod
    @lru_cache(maxsize=128)
    def _cached_knowledge(path: str, mtime_ns: int, size: int) -> tuple[dict[str, Any], str]:
        ResearchKnowledgeSkill._knowledge_loads += 1
        raw = Path(path).read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    @classmethod
    def clear_knowledge_cache(cls) -> None:
        cls._cached_knowledge.cache_clear()
        ResearchKnowledgeSkill._knowledge_loads = 0

    @classmethod
    def knowledge_cache_stats(cls) -> dict[str, int]:
        return {"loads": ResearchKnowledgeSkill._knowledge_loads}

    def run(self, context: SkillContext) -> SkillResult:
        if context.research_state.task_id != self.expected_task:
            return SkillResult(
                self.manifest.skill_id,
                "rejected",
                {"reason": "task_mismatch", "expected_task": self.expected_task},
                provenance={"skill_version": self.manifest.version},
            )
        path = Path(__file__).resolve().parent / self.knowledge_package_name / self.knowledge_filename
        stat = path.stat()
        knowledge, knowledge_sha256 = self._cached_knowledge(
            str(path), stat.st_mtime_ns, stat.st_size
        )
        provenance = {
            "skill_version": self.manifest.version,
            "knowledge_sha256": knowledge_sha256,
            "source": self.source,
        }
        if isinstance(knowledge, dict) and knowledge.get("schema_version"):
            provenance["knowledge_schema"] = knowledge["schema_version"]
        return SkillResult(
            self.manifest.skill_id,
            "ok",
            copy.deepcopy(knowledge),
            next_actions=self.next_actions,
            provenance=provenance,
        )
