from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "controller_bash/scripts/heuresis_suggest.py"
SPEC = importlib.util.spec_from_file_location("heuresis_suggest_coverage", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HeuresisFeatureCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = {"dataset": {"feature_plan": {"search_combinations": [
            {"id": "raw", "feature_ids": []},
            {"id": "basic:text", "feature_ids": ["basic"], "feature_branch": "text"},
            {"id": "basic:combined", "feature_ids": ["basic"], "feature_branch": "combined"},
        ]}}}

    def test_rejects_parameter_only_proposal_when_features_are_available(self) -> None:
        proposal = {"experiment_proposals": [
            {"parameters": {"method": "baseline"}},
            {"parameters": {"method": "baseline", "rank": 32}},
        ]}
        error = MODULE.feature_coverage_error(self.context, proposal)
        self.assertIn("registered feature candidate", error)

    def test_requires_raw_single_and_combined_branches(self) -> None:
        proposal = {"experiment_proposals": [
            {"parameters": {"method": "baseline", "feature_id": "raw"}},
            {"parameters": {"method": "feature", "feature_id": "basic", "feature_branch": "text"}},
            {"parameters": {"method": "feature", "feature_id": "basic", "feature_branch": "combined"}},
        ]}
        self.assertIsNone(MODULE.feature_coverage_error(self.context, proposal))

    def test_task_method_binding_counts_as_registered_feature(self) -> None:
        context = {"dataset": {"feature_plan": {"search_combinations": [
            {"id": "raw", "feature_ids": []},
            {"id": "go", "feature_ids": ["go"], "task_method": "go_model",
             "task_parameters": {"hierarchy": "go"}},
        ]}}}
        proposal = {"experiment_proposals": [
            {"parameters": {"method": "baseline"}},
            {"parameters": {"method": "go_model", "hierarchy": "go"}},
        ]}
        self.assertIsNone(MODULE.feature_coverage_error(context, proposal))


if __name__ == "__main__":
    unittest.main()
