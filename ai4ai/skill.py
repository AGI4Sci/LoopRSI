from __future__ import annotations

from abc import ABC, abstractmethod
from .contracts import SkillContext, SkillManifest, SkillResult


class Skill(ABC):
    manifest: SkillManifest

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if hasattr(cls, "manifest"):
            cls.manifest.validate()

    @abstractmethod
    def run(self, context: SkillContext) -> SkillResult:
        """Consume a bounded research context and return structured evidence."""

