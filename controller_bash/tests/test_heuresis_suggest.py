from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "controller_bash/scripts/heuresis_suggest.py"
SPEC = importlib.util.spec_from_file_location("heuresis_suggest", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HeuresisSuggestionRetryTest(unittest.TestCase):
    def test_complete_v2_proposal_is_accepted(self) -> None:
        proposal = {
            "schema_version": "omni-ar-proposal/v2",
            "proposal_id": "task-round-0-test",
            "task_name": "task",
            "round": 0,
            "verdict": "test",
            "evidence_gaps": [],
            "experiment_proposals": [],
            "risks": [],
        }
        self.assertTrue(MODULE.is_complete_proposal(proposal))

    def test_json_fragment_is_rejected_for_retry(self) -> None:
        self.assertFalse(MODULE.is_complete_proposal({"{": "  :"}))
        self.assertFalse(MODULE.is_complete_proposal({"schema_version": "omni-ar-proposal/v2"}))

    def test_prompt_exposes_bounded_dataset_tool(self) -> None:
        context = {"dataset": {"ref": "vcc25@1"}, "task_spec": {}}
        prompt = MODULE.prompt_for(context)
        self.assertIn('"name": "dataset_adapter"', prompt)
        self.assertIn('"maximum": 20', prompt)
        self.assertIn("omni-ar-tool-call/v1", prompt)

    def test_prompt_includes_contract_feedback_for_retry(self) -> None:
        prompt = MODULE.prompt_for(
            {"dataset": {"ref": "sst2-small@1"}, "task_spec": {}},
            validation_feedback=["parameter C must use snake_case"],
        )
        self.assertIn("parameter C must use snake_case", prompt)

    def test_task_adapter_rejects_unknown_direct_method_before_execution(self) -> None:
        context = {
            "round": 3,
            "paths": {"task_spec": str(ROOT / "tasks/sst2-small/task_spec.yaml")},
        }
        proposal = {
            "schema_version": "omni-ar-proposal/v2",
            "proposal_id": "task-round-3-invalid",
            "task_name": "sst2_small",
            "round": 3,
            "verdict": "test validation",
            "evidence_gaps": [],
            "experiment_proposals": [{
                "hypothesis": "unknown direct method should be rejected",
                "change_scope": ["model"],
                "parameters": {"method": "unknown_model"},
                "expected_effect": {},
                "acceptance_criteria": {},
                "resource_request": {"gpu_count": 0},
            }],
            "implementation_requests": [],
            "risks": [],
        }
        error = MODULE.proposal_contract_error(context, proposal)
        self.assertIsNotNone(error)
        self.assertIn("unknown_model", error)
