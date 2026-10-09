from ai4ai.contracts import ResearchState, SkillContext
from skills.vcc25.lingshu import (
    LingshuAlignmentSkill,
    LingshuDataProcessingInsightSkill,
    LingshuModelDesignInsightSkill,
)
from skills.research_base import ResearchKnowledgeSkill


def _context(task_id="vcc25"):
    return SkillContext(ResearchState(task_id=task_id, question="test"))


def test_lingshu_skills_reuse_common_base_without_changing_entrypoints():
    assert issubclass(LingshuAlignmentSkill, ResearchKnowledgeSkill)
    assert issubclass(LingshuDataProcessingInsightSkill, ResearchKnowledgeSkill)
    assert issubclass(LingshuModelDesignInsightSkill, ResearchKnowledgeSkill)
    assert LingshuAlignmentSkill().run(_context()).status == "ok"


def test_common_base_keeps_task_mismatch_behavior():
    result = LingshuAlignmentSkill().run(_context("other"))
    assert result.status == "rejected"
    assert result.evidence["reason"] == "task_mismatch"

