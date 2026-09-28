from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omni_ar.feature_planner import FeaturePlanningError, plan_features, run_coding_fallback


class FeaturePlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = Path(__file__).resolve().parents[2]

    def test_minds_plan_exposes_branches_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            first = plan_features(
                self.repository, "minds14-fresh-20260824@1",
                rough_idea="improve intent classification", output=output,
            )
            second = plan_features(
                self.repository, "minds14-fresh-20260824@1",
                rough_idea="improve intent classification", output=output,
            )
        ids = {item["id"] for item in first["search_combinations"]}
        self.assertIn("audio_text_basic_v1:text", ids)
        self.assertIn("audio_text_basic_v1:combined", ids)
        self.assertEqual(second["cache_status"], "reused")
        self.assertEqual(first["recommended_builds"][0]["status"], "already_declared")

    def test_unknown_builder_route_is_fail_closed_coding_fallback(self) -> None:
        # universal-weird-demo now has a Builder produced by the accepted
        # Coding fallback.  Mock an actually unseen modality instead of
        # relying on a registered dataset remaining unknown forever.
        with patch(
            "omni_ar.feature_planner.FeatureBuilderRegistry.compatible",
            return_value=[],
        ):
            plan = plan_features(self.repository, "universal-weird-demo@1")
        self.assertEqual(plan["fallback"]["route"], "coding_agent")
        self.assertTrue(plan["fallback"]["requires_user_confirmation"])

    def test_vcc_plan_compiles_go_feature_to_verified_task_methods(self) -> None:
        plan = plan_features(self.repository, "vcc25@2")
        bindings = {
            item.get("task_method"): item.get("task_parameters")
            for item in plan["search_combinations"] if item.get("task_method")
        }
        self.assertIn("pseudobulk_go_hierarchy", bindings)
        self.assertIn("pseudobulk_go_hierarchy_flow", bindings)
        self.assertIn("pseudobulk_go_hierarchy_prototype_flow", bindings)
        self.assertEqual(bindings["pseudobulk_go_hierarchy"]["hierarchy_annotation"], "guide_hierarchy")
        verification = next(
            item for item in plan["search_combinations"]
            if item.get("task_method") == "pseudobulk_go_hierarchy_prototype_flow_verification"
        )
        self.assertEqual(verification["role"], "activation_verification_only")

    def test_coding_fallback_requires_explicit_authorization(self) -> None:
        with self.assertRaises(FeaturePlanningError):
            run_coding_fallback(
                self.repository, self.repository / "fresh_inputs",
                dataset_id="blocked-demo", version="1", blueprint={}, authorized=False,
            )


if __name__ == "__main__":
    unittest.main()
