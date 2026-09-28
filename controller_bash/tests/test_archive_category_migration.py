from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controller_bash/scripts"))

from migrate_archive_categories import migrate_gate, migrate_non_search_entry  # noqa: E402


class ArchiveCategoryMigrationTests(unittest.TestCase):
    def test_completed_acceptance_failure_becomes_rejected(self) -> None:
        gate = {
            "reasons": ["acceptance_criteria_failed"],
            "failure_class": "train_failed", "archive_bucket": "failed_train",
        }
        self.assertTrue(migrate_gate(gate, status="ok"))
        self.assertEqual(gate["failure_class"], "below_baseline")
        self.assertEqual(gate["archive_bucket"], "rejected")

    def test_real_training_failure_is_not_migrated(self) -> None:
        gate = {
            "reasons": ["execution_status_not_ok"],
            "failure_class": "train_failed", "archive_bucket": "failed_train",
        }
        self.assertFalse(migrate_gate(gate, status="failed"))
        self.assertEqual(gate["archive_bucket"], "failed_train")

    def test_activation_verification_is_not_labeled_training_failure(self) -> None:
        entry = {
            "bucket": "verification", "strategy_bucket": "failed_train",
            "strategy_metadata": {"omniepic_failure_mode": "training failed or invalid"},
        }
        self.assertTrue(migrate_non_search_entry(entry))
        self.assertEqual(entry["strategy_bucket"], "activation_verified")
        self.assertEqual(entry["outcome"], "activation_verified")
        self.assertTrue(entry["strategy_metadata"]["excluded_from_search_archive"])
        self.assertNotIn("omniepic_failure_mode", entry["strategy_metadata"])

    def test_below_acceptance_is_not_labeled_training_failure(self) -> None:
        entry = {
            "bucket": "rejected", "strategy_bucket": "failed_train",
            "archive_gate": {
                "failure_class": "below_baseline",
                "reasons": ["acceptance_criteria_failed"],
            },
            "strategy_metadata": {},
        }
        self.assertTrue(migrate_non_search_entry(entry))
        self.assertEqual(entry["strategy_bucket"], "below_acceptance")
        self.assertEqual(entry["outcome"], "below_acceptance")


if __name__ == "__main__":
    unittest.main()
