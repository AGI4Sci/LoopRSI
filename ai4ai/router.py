from __future__ import annotations

from typing import Iterable
from .contracts import SkillContext, SkillManifest, SkillResult
from .skill import Skill
from .external import ExternalSkill


class SkillRouter:
    """In-process v0 router; future workers can implement the same contract over JSON."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: Skill) -> SkillManifest:
        manifest = skill.manifest
        manifest.validate()
        if manifest.skill_id in self._skills:
            raise ValueError(f"duplicate skill: {manifest.skill_id}")
        self._skills[manifest.skill_id] = skill
        return manifest

    def register_external(self, manifest, command, timeout_s=30.0):
        return self.register(ExternalSkill(manifest, tuple(command), timeout_s))

    def backend(self, skill_id):
        return "subprocess" if isinstance(self._skills[skill_id], ExternalSkill) else "in-process"

    def discover(self, capability: str | None = None) -> list[SkillManifest]:
        values = [s.manifest for s in self._skills.values()]
        return [m for m in values if capability is None or capability in m.capabilities]

    def run(self, skill_id: str, context: SkillContext) -> SkillResult:
        try:
            skill = self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {skill_id}") from exc
        result = skill.run(context)
        if result.provenance.get("skill_version") not in {None, skill.manifest.version}:
            raise ValueError("skill version mismatch")
        if result.skill_id != skill_id:
            raise ValueError("skill result identity mismatch")
        if result.status not in {"ok", "blocked", "rejected"}:
            raise ValueError(f"invalid skill status: {result.status}")
        return result

