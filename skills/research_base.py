"""Small reusable base for domain-specific research knowledge skills."""

from __future__ import annotations

import hashlib
import json
import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai4ai.contracts import SkillContext, SkillResult
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
