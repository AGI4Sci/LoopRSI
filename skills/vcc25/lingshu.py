"""VCC25/Lingshu domain adapters built on generic research skills."""
from __future__ import annotations

from ai4ai.contracts import SkillManifest
from skills.research_base import ResearchKnowledgeSkill


class LingshuAlignmentSkill(ResearchKnowledgeSkill):
    expected_task = "vcc25"
    knowledge_filename = "knowledge.json"
    source = "validated_lingshu_alignment_reports"
    next_actions = (
        {"type": "planner_input", "priority": "new_target_biological_information"},
        {"type": "representation_gate", "protocol": "official_target_go_bp_features"},
        {"type": "objective_gate", "protocol": "train_only_de_direction_and_ranking"},
        {"type": "neighborhood_gate", "protocol": "learned_attention_vs_shuffled_and_random_controls"},
        {"type": "calibration_gate", "protocol": "validation_only_direction_amplitude_calibration"},
        {"type": "falsification_gate", "control": "degree_matched_random_graph"},
        {"type": "stability_gate", "protocol": "multifold_multi_seed_validation"},
        {"type": "evaluation_gate", "protocol": "official_h1_cell_eval"},
    )
    manifest = SkillManifest(
        skill_id="vcc25.lingshu_alignment", version="1.2.0", kind="scientific_knowledge",
        capabilities=("benchmark_alignment", "failure_memory", "research_prioritization"),
        input_schema="ai4ai/skill-context/v1", output_schema="ai4ai/skill-result/v1",
        entrypoint="skills.vcc25.lingshu:LingshuAlignmentSkill",
        context_requirements=("research_state.task_id",),
        provenance={"owner": "vcc25", "source": "validated_lingshu_alignment_reports"},
    )


class LingshuDataProcessingInsightSkill(ResearchKnowledgeSkill):
    expected_task = "vcc25"
    knowledge_filename = "lingshu_data_processing_knowledge.json"
    source = "lingshu_release_and_vcc25_data_processing_audits"
    next_actions = (
        {"type": "planner_input", "priority": "target_specific_gene_level_representation"},
        {"type": "preflight_gate", "protocol": "gene_order_raw_count_and_leakage_audit"},
        {"type": "ablation_gate", "protocol": "single_data_processing_variable_only"},
    )
    manifest = SkillManifest(
        skill_id="vcc25.lingshu_data_processing_insight", version="0.1.0", kind="research_knowhow",
        capabilities=("data_representation_insight", "hypothesis_generation", "failure_memory"),
        input_schema="ai4ai/skill-context/v1", output_schema="ai4ai/skill-result/v1",
        entrypoint="skills.vcc25.lingshu:LingshuDataProcessingInsightSkill",
        context_requirements=("research_state.task_id",),
        provenance={"owner": "vcc25", "source": "lingshu_release_and_vcc25_data_processing_audits"},
    )


class LingshuModelDesignInsightSkill(ResearchKnowledgeSkill):
    expected_task = "vcc25"
    knowledge_filename = "lingshu_model_design_knowledge.json"
    source = "lingshu_release_and_vcc25_model_design_audits"
    next_actions = (
        {"type": "planner_input", "priority": "condition_aware_full_gene_hypotheses"},
        {"type": "design_gate", "protocol": "do_not_copy_lingshu_or_claim_lingshu_training"},
        {"type": "falsification_gate", "protocol": "shuffled_condition_or_prior_control"},
    )
    manifest = SkillManifest(
        skill_id="vcc25.lingshu_model_design_insight", version="0.1.0", kind="research_knowhow",
        capabilities=("model_design_insight", "hypothesis_generation", "ablation_design"),
        input_schema="ai4ai/skill-context/v1", output_schema="ai4ai/skill-result/v1",
        entrypoint="skills.vcc25.lingshu:LingshuModelDesignInsightSkill",
        context_requirements=("research_state.task_id",),
        provenance={"owner": "vcc25", "source": "lingshu_release_and_vcc25_model_design_audits"},
    )
