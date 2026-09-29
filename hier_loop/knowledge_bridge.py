"""Validated knowledge injection for the five-layer VCC25 hierarchy."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai4ai.plugin_manifest import TaskPluginManifest
from ai4ai.plugin_registry import load_skill_bundle


class KnowledgePreflightError(ValueError):
    """Raised when a hierarchy layer cannot receive auditable knowledge."""


def resolve_skill_id(skill: object) -> str:
    """Return an ID for either a ResearchSkill or a legacy manifest skill."""

    direct = getattr(skill, "skill_id", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    manifest = getattr(skill, "manifest", None)
    legacy = getattr(manifest, "skill_id", None)
    if isinstance(legacy, str) and legacy.strip():
        return legacy.strip()
    raise ValueError("skill must expose skill_id or manifest.skill_id")


@dataclass(frozen=True)
class KnowledgeInjection:
    layer: str
    skill_id: str
    card_ids: tuple[str, ...]
    content: str
    content_hash: str
    token_budget: int
    source: str


class KnowledgeBridge:
    """Route a hierarchy layer to one manifest-declared ResearchSkill."""

    def __init__(
        self,
        manifest: TaskPluginManifest,
        skills: Sequence[object] | None = None,
    ) -> None:
        self.manifest = manifest
        self.skills = tuple(load_skill_bundle(manifest) if skills is None else skills)

    def inject(self, layer: str, context: Mapping[str, Any]) -> KnowledgeInjection:
        if layer not in {"L1", "L2", "L3", "L4", "L5"}:
            raise KnowledgePreflightError(f"unsupported hierarchy layer: {layer}")
        task_id = str(context.get("task_id") or self.manifest.task_id)
        if task_id != self.manifest.task_id:
            raise KnowledgePreflightError(
                f"knowledge task mismatch: expected {self.manifest.task_id}, got {task_id}"
            )
        skill_context = dict(context)
        skill_context.update({"task_id": task_id, "layer": layer})
        active = [
            skill
            for skill in self.skills
            if callable(getattr(skill, "can_activate", None))
            and skill.can_activate(skill_context)
        ]
        if len(active) != 1:
            raise KnowledgePreflightError(
                f"expected exactly one knowledge skill for {layer}, found {len(active)}"
            )

        skill = active[0]
        fragment = skill.inject(skill_context)
        if fragment is None or not str(getattr(fragment, "content", "")).strip():
            raise KnowledgePreflightError(f"empty knowledge fragment for {layer}")

        source = str(getattr(fragment, "source", ""))
        prefix = "knowledge:vcc25:"
        if not source.startswith(prefix):
            raise KnowledgePreflightError(f"invalid knowledge source for {layer}: {source!r}")
        card_ids = tuple(value for value in source[len(prefix) :].split(",") if value)
        if not card_ids:
            raise KnowledgePreflightError(f"empty knowledge provenance for {layer}")

        content = str(fragment.content)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        declared_hash = str(getattr(fragment, "content_hash", ""))
        if declared_hash != content_hash:
            raise KnowledgePreflightError(f"knowledge content hash mismatch for {layer}")
        token_budget = getattr(fragment, "token_budget", None)
        if not isinstance(token_budget, int) or token_budget <= 0:
            raise KnowledgePreflightError(f"invalid knowledge token budget for {layer}")

        return KnowledgeInjection(
            layer=layer,
            skill_id=resolve_skill_id(skill),
            card_ids=card_ids,
            content=content,
            content_hash=content_hash,
            token_budget=token_budget,
            source=source,
        )
