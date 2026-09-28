from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from .contracts import SkillContext, SkillResult
from .router import SkillRouter


@dataclass
class ResearchLoop:
    router: SkillRouter
    lineage: list[dict[str, Any]] = field(default_factory=list)

    def invoke(self, skill_id: str, context: SkillContext) -> SkillResult:
        result = self.router.run(skill_id, context)
        self.lineage.append({"skill_id": skill_id, "status": result.status, "evidence": dict(result.evidence)})
        return result

