from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from datasets.registry import DatasetRegistry
from omni_ar.loop import ResearchLoop
from tasks.vcc25.adapter import AdapterError, VCC25TaskAdapter


REPOSITORY = Path(__file__).resolve().parents[2]


class VCC25HierarchyContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = DatasetRegistry(REPOSITORY / "datasets").load_adapter("vcc25@2")

    def test_registered_dag_is_complete_and_hashed(self) -> None:
        result = self.dataset.validate_hierarchy(deep=True)
        self.assertTrue(result["valid"])
        self.assertTrue(result["acyclic"])
        self.assertTrue(result["all_nodes_reachable_from_root"])
        self.assertTrue(result["target_coverage_complete"])
        self.assertTrue(result["guide_coverage_complete"])
        self.assertTrue(all(item["sha256_matches"] for item in result["hash_checks"]))

    def test_negative_controls_are_deterministic_and_distinct(self) -> None:
        shuffled = self.dataset.build_negative_control(
            strategy="shuffle_target_annotations", seed=20250805
        )
        repeated = self.dataset.build_negative_control(
            strategy="shuffle_target_annotations", seed=20250805
        )
        degree = self.dataset.build_negative_control(
            strategy="degree_preserving_target_swap", seed=20250805
        )
        self.assertEqual(shuffled["edge_sha256"], repeated["edge_sha256"])
        self.assertNotEqual(shuffled["edge_sha256"], degree["edge_sha256"])
        self.assertLess(shuffled["unchanged_annotation_fraction"], 1.0)
        self.assertLess(degree["unchanged_annotation_fraction"], 1.0)

    def test_loop_tool_redacts_physical_paths(self) -> None:
        result = ResearchLoop(REPOSITORY).dataset_tool(
            "vcc25@2", "load_hierarchy", limit=2
        )
        self.assertNotIn(str(REPOSITORY), str(result["payload"]))

    def test_invalid_hierarchy_parameters_fail_closed(self) -> None:
        spec = yaml.safe_load((REPOSITORY / "tasks/vcc25/task_spec.yaml").read_text())
        adapter = VCC25TaskAdapter()
        proposal = {
            "hypothesis": "invalid layer mode",
            "change_scope": ["representation", "model"],
            "parameters": {
                "method": "pseudobulk_go_hierarchy_verification",
                "hierarchy_layer_mode": "invented",
            },
            "expected_effect": {"mean_delta_pearson": {"direction": "increase", "minimum_change": 0.0}},
            "acceptance_criteria": {"mean_delta_pearson": {"operator": ">=", "value": 0.0}},
            "resource_request": {"gpu_count": 1, "cpu": 8, "memory_mb": 20000, "max_runtime_minutes": 20},
        }
        with self.assertRaises(AdapterError):
            adapter.proposal_to_trial(proposal, spec["adapter"]["trial_defaults"], "invalid")


if __name__ == "__main__":
    unittest.main()
