import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai4ai.plugin_manifest import load_task_plugin_manifest
from ai4ai.plugin_protocols import PromptFragment
from controller_bash.scripts.apply_skill_context import build_skill_injections
from hier_loop.knowledge_bridge import (
    KnowledgeBridge,
    KnowledgePreflightError,
    resolve_skill_id,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "tasks" / "vcc25" / "task_plugin.yaml"


class _NewResearchSkill:
    skill_id = "vcc25.new"


class _LegacySkill:
    manifest = SimpleNamespace(skill_id="vcc25.legacy")


class _EmptySkill:
    skill_id = "vcc25.empty"

    def can_activate(self, context):
        return context.get("layer") == "L1"

    def inject(self, context):
        return None


class KnowledgeBridgeTests(unittest.TestCase):
    def test_new_research_skill_id_is_supported(self):
        self.assertEqual(resolve_skill_id(_NewResearchSkill()), "vcc25.new")

    def test_legacy_manifest_skill_id_is_supported(self):
        self.assertEqual(resolve_skill_id(_LegacySkill()), "vcc25.legacy")

    def test_each_layer_receives_nonempty_vcc25_knowledge(self):
        bridge = KnowledgeBridge(load_task_plugin_manifest(MANIFEST_PATH))
        expected = {
            "L1": "vcc25.lingshu.alignment",
            "L2": "vcc25.lingshu.alignment",
            "L3": "vcc25.lingshu.data_processing",
            "L4": "vcc25.lingshu.data_processing",
            "L5": "vcc25.lingshu.model_design",
        }

        for layer, skill_id in expected.items():
            with self.subTest(layer=layer):
                injection = bridge.inject(layer, {"query": "VCC25 perturbation prediction"})
                self.assertEqual(injection.layer, layer)
                self.assertEqual(injection.skill_id, skill_id)
                self.assertTrue(injection.content.strip())
                self.assertTrue(injection.card_ids)

    def test_injection_records_card_ids_and_content_hash(self):
        bridge = KnowledgeBridge(load_task_plugin_manifest(MANIFEST_PATH))
        injection = bridge.inject("L5", {"token_budget": 1600})

        self.assertTrue(all(card_id.startswith("kb:") for card_id in injection.card_ids))
        self.assertEqual(
            injection.content_hash,
            hashlib.sha256(injection.content.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(injection.source, "knowledge:vcc25:" + ",".join(injection.card_ids))
        self.assertGreater(injection.token_budget, 0)

    def test_empty_fragment_fails_preflight(self):
        manifest = load_task_plugin_manifest(MANIFEST_PATH)
        bridge = KnowledgeBridge(manifest, skills=(_EmptySkill(),))

        with self.assertRaisesRegex(KnowledgePreflightError, "empty knowledge fragment"):
            bridge.inject("L1", {})

    def test_old_controller_hook_accepts_new_research_skills(self):
        injections = build_skill_injections(
            {
                "task_spec": {"task": {"name": "vcc25"}},
                "scientific_question": "Improve VCC25 perturbation prediction",
            }
        )

        self.assertEqual([item["layer"] for item in injections], ["L1", "L2", "L3", "L4", "L5"])
        self.assertEqual(
            [item["skill_id"] for item in injections],
            [
                "vcc25.lingshu.alignment",
                "vcc25.lingshu.alignment",
                "vcc25.lingshu.data_processing",
                "vcc25.lingshu.data_processing",
                "vcc25.lingshu.model_design",
            ],
        )


if __name__ == "__main__":
    unittest.main()
