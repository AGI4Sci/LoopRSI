import hashlib
import unittest
from pathlib import Path

from ai4ai.plugin_manifest import load_task_plugin_manifest
from ai4ai.plugin_registry import load_skill_bundle
from domain_knowledge import KnowledgeQuery, KnowledgeStore
from domain_knowledge.render import render_cards
from skills.vcc25.lingshu import (
    LingshuAlignmentSkill,
    LingshuDataProcessingInsightSkill,
    LingshuModelDesignInsightSkill,
)


ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = ROOT / "knowledge" / "vcc25"


class LingshuSkillTests(unittest.TestCase):
    def test_real_manifest_entrypoints_instantiate(self):
        manifest = load_task_plugin_manifest(ROOT / "tasks" / "vcc25" / "task_plugin.yaml")
        skills = load_skill_bundle(manifest)
        self.assertEqual(
            [skill.skill_id for skill in skills],
            [
                "vcc25.lingshu.alignment",
                "vcc25.lingshu.data_processing",
                "vcc25.lingshu.model_design",
            ],
        )

    def test_non_vcc25_context_never_activates(self):
        skills = [
            LingshuAlignmentSkill(KNOWLEDGE_ROOT),
            LingshuDataProcessingInsightSkill(KNOWLEDGE_ROOT),
            LingshuModelDesignInsightSkill(KNOWLEDGE_ROOT),
        ]
        self.assertTrue(all(not skill.can_activate({"task_id": "other", "layer": "L5"}) for skill in skills))

    def test_l1_and_l2_alignment_returns_paper_evidence(self):
        skill = LingshuAlignmentSkill(KNOWLEDGE_ROOT)
        fragment = skill.inject({"task_id": "vcc25", "layer": "L1", "token_budget": 900})
        self.assertIsNotNone(fragment)
        self.assertIn("asset_type: paper", fragment.content)
        self.assertIn("review_basis: provisional_initial_release", fragment.content)
        self.assertIn("card_id: kb:paper:", fragment.content)
        self.assertIn("source: https://", fragment.content)

    def test_l3_and_l4_data_skill_exposes_mechanisms_and_limitations(self):
        skill = LingshuDataProcessingInsightSkill(KNOWLEDGE_ROOT)
        fragment = skill.inject({"task_id": "vcc25", "layer": "L3", "token_budget": 900})
        self.assertIsNotNone(fragment)
        self.assertIn("mechanism:", fragment.content)
        self.assertIn("limitations:", fragment.content)

    def test_l5_model_skill_returns_repository_and_model_execution_facts(self):
        skill = LingshuModelDesignInsightSkill(KNOWLEDGE_ROOT)
        fragment = skill.inject({"task_id": "vcc25", "layer": "L5", "token_budget": 1600})
        self.assertIsNotNone(fragment)
        self.assertIn("asset_type: repository", fragment.content)
        self.assertIn("asset_type: model", fragment.content)
        self.assertIn("execution_readiness:", fragment.content)
        self.assertIn("adapter_id:", fragment.content)
        # L5 now injects multiple model candidates for autonomous selection.
        # The highest-ranked model (lingshu-vcc-85m) must always be present.
        self.assertIn("kb:model:lingshu-vcc-85m", fragment.source)
        self.assertIn("kb:repo:lingshu-cell", fragment.source)
        # At least two model candidates should be returned for comparison.
        model_ids = [part for part in fragment.source.split(":") if part.startswith("kb:model")]
        model_lines = [line for line in fragment.content.splitlines() if line.startswith("card_id: kb:model:")]
        self.assertGreaterEqual(len(model_lines), 2)

    def test_l5_targeted_gears_uses_matching_repository(self):
        skill = LingshuModelDesignInsightSkill(KNOWLEDGE_ROOT)
        fragment = skill.inject({"task_id": "vcc25", "layer": "L5", "query": "GEARS", "token_budget": 1600})
        self.assertIsNotNone(fragment)
        self.assertEqual(fragment.source, "knowledge:vcc25:kb:repo:gears,kb:model:gears")
        self.assertIn("smoke_evidence: passed (smoke_only)", fragment.content)
        self.assertIn("gears-h1-gpu-smoke-20261005.json", fragment.content)

    def test_renderer_is_deterministic_hashes_content_and_never_truncates_a_card(self):
        store = KnowledgeStore.from_directory(KNOWLEDGE_ROOT)
        card = store.query(
            KnowledgeQuery(task_id="vcc25", asset_types=("paper",), max_results=1)
        )[0]
        first = render_cards([card], token_budget=1000)
        second = render_cards([card], token_budget=1000)
        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, hashlib.sha256(first.content.encode("utf-8")).hexdigest())
        self.assertEqual(first.card_ids, (card.id,))
        too_small = render_cards([card], token_budget=max(first.estimated_tokens - 1, 0))
        self.assertEqual(too_small.content, "")
        self.assertEqual(too_small.card_ids, ())

    def test_inject_returns_none_when_no_whole_card_fits(self):
        skill = LingshuAlignmentSkill(KNOWLEDGE_ROOT)
        self.assertIsNone(skill.inject({"task_id": "vcc25", "layer": "L1", "token_budget": 1}))

    def test_proposal_validation_recursively_rejects_restricted_identifiers(self):
        skill = LingshuModelDesignInsightSkill(KNOWLEDGE_ROOT)
        context = {"task_id": "vcc25", "layer": "L5"}
        result = skill.validate_proposal(
            {"steps": [{"inputs": {"path": "/private/competition_test/data"}}]},
            context,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("competition_test" in error for error in result.errors))
        self.assertTrue(skill.validate_proposal({"model_id": "kb:model:lingshu-vcc-85m"}, context).passed)


if __name__ == "__main__":
    unittest.main()
