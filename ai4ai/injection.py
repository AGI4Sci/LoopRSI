from __future__ import annotations

from typing import Any
from .contracts import SkillContext


def inject_skill_context(context: SkillContext, skill_id: str, evidence: dict[str, Any]) -> SkillContext:
    """Return a new planner context; injected evidence never mutates prior state."""
    state = context.research_state
    record = {"source": skill_id, **evidence}
    return SkillContext(
        research_state=type(state)(
            task_id=state.task_id,
            question=state.question,
            evidence=(*state.evidence, record),
            open_hypotheses=state.open_hypotheses,
            rejected_hypotheses=state.rejected_hypotheses,
            blockers=state.blockers,
        ),
        constraints=context.constraints,
    )

