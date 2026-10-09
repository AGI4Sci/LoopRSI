from ai4ai.contracts import ResearchState, SkillContext
from skills.vcc25.lingshu import (
    LingshuAlignmentSkill,
    LingshuDataProcessingInsightSkill,
    LingshuModelDesignInsightSkill,
)
from skills.research_base import ResearchKnowledgeSkill
from controller_bash.scripts.apply_skill_context import compile_skill_constraints
from controller_bash.scripts.heuresis_suggest import skill_constraint_error


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


def test_knowledge_payload_and_hash_are_reused_between_calls():
    ResearchKnowledgeSkill.clear_knowledge_cache()
    skill = LingshuAlignmentSkill()
    first = skill.run(_context())
    second = skill.run(_context())
    assert first.evidence == second.evidence
    assert first.provenance["knowledge_sha256"] == second.provenance["knowledge_sha256"]
    assert ResearchKnowledgeSkill.knowledge_cache_stats()["loads"] == 1


def test_skill_actions_compile_to_machine_checkable_heuresis_constraints():
    actions = (
        {"type": "falsification_gate", "control": "degree-matched-random-graph"},
        {"type": "preflight_gate", "protocol": "gene_order_raw_count_and_leakage_audit"},
        {"type": "stability_gate", "protocol": "multifold_multi_seed_validation"},
    )
    constraints = compile_skill_constraints(actions)
    assert constraints["controls"] == ["degree-matched-random-graph"]
    assert "gene_order_raw_count_and_leakage_audit" in constraints["acceptance"]["required_gates"]
    assert "multifold_multi_seed_validation" in constraints["acceptance"]["required_gates"]
    assert constraints["acceptance"]["test_expression_used"] is False


def test_proposal_validator_rejects_dropped_skill_constraints():
    context = {"heuresis_constraints": {
        "controls": ["shuffle"],
        "acceptance": {"required_gates": ["multi_seed"]},
    }}
    proposal = {"experiment_proposals": [{"mechanism_alignment": {
        "required_controls": [], "required_gates": [],
    }}]}
    error = skill_constraint_error(context, proposal)
    assert error and "SkillConstraintError" in error
