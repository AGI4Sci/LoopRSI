"""Small reusable base for domain-specific research knowledge skills."""

from __future__ import annotations

import hashlib
import json
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

    def run(self, context: SkillContext) -> SkillResult:
        if context.research_state.task_id != self.expected_task:
            return SkillResult(
                self.manifest.skill_id,
                "rejected",
                {"reason": "task_mismatch", "expected_task": self.expected_task},
                provenance={"skill_version": self.manifest.version},
            )
        path = Path(__file__).resolve().parent / self.knowledge_package_name / self.knowledge_filename
        raw = path.read_bytes()
        knowledge = json.loads(raw)
        provenance = {
            "skill_version": self.manifest.version,
            "knowledge_sha256": hashlib.sha256(raw).hexdigest(),
            "source": self.source,
        }
        if isinstance(knowledge, dict) and knowledge.get("schema_version"):
            provenance["knowledge_schema"] = knowledge["schema_version"]
        return SkillResult(
            self.manifest.skill_id,
            "ok",
            knowledge,
            next_actions=self.next_actions,
            provenance=provenance,
        )
