from ai4ai.contracts import SkillContext, SkillManifest, SkillResult
from ai4ai.skill import Skill
class EvidenceSummarySkill(Skill):
    manifest=SkillManifest(skill_id="example.evidence_summary", version="0.1.0", kind="research_knowhow", capabilities=("summarize_evidence","extract_failures"), input_schema="ai4ai/skill-context/v1", output_schema="ai4ai/skill-result/v1", entrypoint="skills.example.evidence:EvidenceSummarySkill", context_requirements=("research_state.evidence",), provenance={"owner":"ai4ai-core","source":"new-project"})
    def run(self, context):
        evidence=list(context.research_state.evidence); failures=[e for e in evidence if e.get("status") in {"failed","rejected","blocked"} or e.get("decision") in {"reject","rejected","blocked"}]
        return SkillResult(self.manifest.skill_id,"ok",{"evidence_count":len(evidence),"failure_count":len(failures),"failure_ids":[e.get("id") for e in failures if e.get("id")]},({"type":"planner_input","source":self.manifest.skill_id},),{"skill_version":self.manifest.version})
