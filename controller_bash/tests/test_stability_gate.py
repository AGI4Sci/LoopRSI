from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controller_bash/scripts"))

from stability_gate import archive_decision, classify_failure  # noqa: E402


class StabilityGateTests(unittest.TestCase):
    def result(self):
        return {
            "status": "ok",
            "resource_usage": {"within_budget": True},
            "protocol": {"dataset_ref": "vcc25@2", "requested_split": "heldout_guide", "seed": 7},
        }

    def test_valid_fixed_context_is_accepted(self):
        record = {"status": "ok", "result_contract_status": "ok", "acceptance": {"passed": True}}
        decision = archive_decision(
            record, self.result(),
            expected_context={"dataset_ref": "vcc25@2", "requested_split": "heldout_guide", "seed": 7},
        )
        self.assertTrue(decision["eligible"])
        self.assertEqual(decision["archive_bucket"], "accepted")
        self.assertEqual(decision["outcome"], "accepted")

    def test_protocol_drift_and_missing_budget_fail_closed(self):
        result = self.result()
        result["protocol"]["seed"] = 8
        result["resource_usage"]["within_budget"] = None
        decision = archive_decision(
            {"status": "ok", "result_contract_status": "ok"}, result,
            expected_context={"seed": 7},
        )
        self.assertFalse(decision["eligible"])
        self.assertIn("fixed_context_mismatch:seed", decision["reasons"])
        self.assertIn("budget_not_proven", decision["reasons"])

    def test_timeout_is_non_retryable_to_avoid_duplicate_training(self):
        failure = classify_failure({"status": "timeout"})
        self.assertEqual(failure["failure_class"], "execution_timeout")
        self.assertFalse(failure["retryable"])

    def test_training_failure_is_distinct_from_invalid_result(self):
        training = classify_failure({"status": "failed", "error": "worker exited 1"})
        invalid = classify_failure({
            "status": "ok", "result_contract_error": "standard result contract missing",
        })
        self.assertEqual(training["failure_class"], "training_failed")
        self.assertEqual(training["archive_bucket"], "failed_train")
        self.assertEqual(invalid["failure_class"], "result_invalid")
        self.assertEqual(invalid["outcome"], "result_invalid")
        self.assertEqual(invalid["archive_bucket"], "invalid")

    def test_inactive_method_is_invalid(self):
        failure = classify_failure({"status": "ok", "error": "activation diagnostic inactive"})
        self.assertEqual(failure["failure_class"], "method_inactive")
        self.assertEqual(failure["archive_bucket"], "invalid")

    def test_completed_trial_below_threshold_is_rejected(self):
        record = {
            "status": "ok", "result_contract_status": "ok",
            "acceptance": {"passed": False},
        }
        decision = archive_decision(record, self.result())
        self.assertEqual(decision["failure_class"], "below_baseline")
        self.assertEqual(decision["archive_bucket"], "rejected")
        self.assertNotEqual(decision["archive_bucket"], "failed_train")
        self.assertEqual(decision["outcome"], "below_acceptance")

    def test_selected_feature_without_activation_is_invalid(self):
        result = self.result()
        result["features"] = {"status": "invalid", "records": [{"enabled": False}]}
        decision = archive_decision(
            {"status": "ok", "result_contract_status": "ok", "acceptance": {"passed": True}},
            result,
        )
        self.assertEqual(decision["failure_class"], "method_inactive")
        self.assertEqual(decision["outcome"], "method_inactive")


if __name__ == "__main__":
    unittest.main()
