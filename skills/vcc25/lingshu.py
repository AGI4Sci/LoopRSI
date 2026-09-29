"""Manifest-declared knowledge skills for the VCC25 hierarchy."""

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from ai4ai.plugin_protocols import PromptFragment, ValidationResult
from domain_knowledge import KnowledgeQuery, KnowledgeStore, KnowledgeValidationError
from domain_knowledge.render import render_cards
from domain_knowledge.store import assert_safe_knowledge


DEFAULT_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "knowledge" / "vcc25"


class _KnowledgeSkill:
    _skill_id = ""
    _layers: Tuple[str, ...] = ()
    _asset_types: Tuple[str, ...] = ()
    _priority = 50

    def __init__(self, knowledge_root: Optional[Path] = None) -> None:
        self._root = Path(knowledge_root) if knowledge_root is not None else DEFAULT_KNOWLEDGE_ROOT
        self._store = KnowledgeStore.from_directory(self._root)

    @property
    def skill_id(self) -> str:
        return self._skill_id

    def can_activate(self, context: Mapping[str, Any]) -> bool:
        return context.get("task_id") == "vcc25" and context.get("layer") in self._layers

    def inject(self, context: Mapping[str, Any]) -> Optional[PromptFragment]:
        if not self.can_activate(context):
            return None
        assert_safe_knowledge(context, "skill_context")
        budget = int(context.get("token_budget", 1200))
        text = str(context.get("query", context.get("text", "")))
        cards = self._store.query(
            KnowledgeQuery(
                task_id="vcc25",
                layer=str(context["layer"]),
                text=text,
                asset_types=self._asset_types,
                max_results=10,
            )
        )
        rendered = render_cards(cards, token_budget=budget)
        if not rendered.card_ids:
            return None
        return PromptFragment(
            skill_id=self.skill_id,
            stage=str(context["layer"]),
            content=rendered.content,
            priority=self._priority,
            token_budget=rendered.estimated_tokens,
            activation_reason=f"VCC25 {context['layer']} domain knowledge",
            source="knowledge:vcc25:" + ",".join(rendered.card_ids),
            content_hash=rendered.content_hash,
        )

    def validate_proposal(
        self,
        proposal: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> ValidationResult:
        try:
            assert_safe_knowledge(proposal, "proposal")
            assert_safe_knowledge(context, "proposal_context")
        except KnowledgeValidationError as exc:
            return ValidationResult(passed=False, errors=(str(exc),))
        return ValidationResult(
            passed=True,
            checks=({"check": "restricted_identifiers", "passed": True},),
        )


class LingshuAlignmentSkill(_KnowledgeSkill):
    _skill_id = "vcc25.lingshu.alignment"
    _layers = ("L1", "L2")
    _asset_types = ("paper",)
    _priority = 70


class LingshuDataProcessingInsightSkill(_KnowledgeSkill):
    _skill_id = "vcc25.lingshu.data_processing"
    _layers = ("L3", "L4")
    _asset_types = ("paper",)
    _priority = 65


class LingshuModelDesignInsightSkill(_KnowledgeSkill):
    _skill_id = "vcc25.lingshu.model_design"
    _layers = ("L5",)
    _asset_types = ("repository", "model")
    _priority = 80
