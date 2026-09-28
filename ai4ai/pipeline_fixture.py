from __future__ import annotations
from dataclasses import dataclass
from .contracts import ResearchState
from .router import SkillRouter
from .stage_hook import SkillStageHook
@dataclass(frozen=True)
class PipelineTrace:
 stages: tuple[str,...]
 state: ResearchState
 skill_injected: bool
 def as_dict(self): return {"stages":list(self.stages),"evidence":list(self.state.evidence),"skill_injected":self.skill_injected}
def run_original_fixture(state: ResearchState) -> PipelineTrace:
 stages=("ResearchStudio", "QA", "IdeaCard", "Heuresis", "Proposal", "Candidate", "TaskAdapter", "Experiment", "Evaluator", "Archive")
 return PipelineTrace(stages,state,False)
def run_skill_fixture(state: ResearchState,router: SkillRouter) -> PipelineTrace:
 stages=("ResearchStudio", "QA", "SkillRouter", "SkillInjection", "IdeaCard", "Heuresis", "Proposal", "Candidate", "TaskAdapter", "Experiment", "Evaluator", "Archive")
 state,_=SkillStageHook(router).inject("qa_to_idea_card","example.evidence_summary",state,{"bounded":True})
 return PipelineTrace(stages,state,True)
