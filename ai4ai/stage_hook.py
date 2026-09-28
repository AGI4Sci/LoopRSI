from __future__ import annotations
from dataclasses import dataclass
from .contracts import ResearchState, SkillContext, SkillResult
from .router import SkillRouter

@dataclass(frozen=True)
class StageInjection:
 stage: str
 result: SkillResult

class SkillStageHook:
 """Adapter layer: adds bounded skill evidence to an existing stage context."""
 def __init__(self, router: SkillRouter): self.router=router
 def inject(self, stage: str, skill_id: str, state: ResearchState, constraints=None) -> tuple[ResearchState, StageInjection]:
  result=self.router.run(skill_id, SkillContext(state,constraints or {}))
  forbidden={"proposal","proposal_id","candidate","candidate_id","experiment","rjob","evaluator_result"}
  if forbidden.intersection(result.evidence):
   raise ValueError("skill output crosses bounded evidence boundary")
  bounded={"source":skill_id,"stage":stage,"evidence":dict(result.evidence)}
  augmented=ResearchState(state.task_id,state.question,(*state.evidence,bounded),state.open_hypotheses,state.rejected_hypotheses,state.blockers)
  return augmented,StageInjection(stage,result)
