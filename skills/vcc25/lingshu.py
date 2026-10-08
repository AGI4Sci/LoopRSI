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
        # L5 (multi-asset) needs a larger budget to show multiple model
        # candidates for autonomous comparison; planning layers (L1-L4)
        # stay at the leaner default.
        default_budget = 2400 if len(self._asset_types) > 1 else 1200
        budget = int(context.get("token_budget", default_budget))
        text = str(context.get("query", context.get("text", "")))
        if len(self._asset_types) > 1:
            # Query all L5-eligible models so the decision agent can compare
            # the full candidate space.  The token budget in render_cards
            # controls how many cards actually fit in the prompt — we do not
            # pre-truncate with max_results here.
            models = self._store.query(KnowledgeQuery(
                task_id="vcc25", layer=str(context["layer"]), text=text,
                asset_types=("model",), max_results=50,
            ))
            repositories = self._store.query(KnowledgeQuery(
                task_id="vcc25", layer=str(context["layer"]), text=text,
                asset_types=("repository",), max_results=20,
            ))
            repo_by_id = {r.id: r for r in repositories}
            # Interleave model+repository pairs so that each model is rendered
            # immediately after its related repository.  This ensures that
            # models (the primary decision candidates) are rendered before the
            # token budget is exhausted by repositories alone.
            cards = []
            seen_repos = set()
            for model in models:
                repo = repo_by_id.get(model.id.replace("kb:model:", "kb:repo:"))
                # Fall back to relation lookup if the ID heuristic fails.
                if repo is None:
                    repo = next(
                        (r for r in repositories if model.id in r.relations),
                        None,
                    )
                if repo is not None and repo.id not in seen_repos:
                    cards.append(repo)
                    seen_repos.add(repo.id)
                cards.append(model)
            # Append any remaining repos that were not related to a model.
            for repo in repositories:
                if repo.id not in seen_repos:
                    cards.append(repo)
            cards = tuple(cards)
        else:
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
