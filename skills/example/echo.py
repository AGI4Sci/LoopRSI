from ai4ai.contracts import SkillContext, SkillManifest, SkillResult
from ai4ai.skill import Skill


class EchoSkill(Skill):
    manifest = SkillManifest(
        skill_id="example.echo",
        version="0.1.0",
        kind="research_knowhow",
        capabilities=("inspect_context",),
        input_schema="ai4ai/skill-context/v1",
        output_schema="ai4ai/skill-result/v1",
        entrypoint="skills.example.echo:EchoSkill",
        provenance={"owner":"ai4ai-core","source":"new-project"},
    )

    def run(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            skill_id=self.manifest.skill_id,
            status="ok",
            evidence={"question": context.research_state.question, "evidence_count": len(context.research_state.evidence)},
            next_actions=({"type": "record_evidence", "source": self.manifest.skill_id},),
            provenance={"skill_version": self.manifest.version},
        )

