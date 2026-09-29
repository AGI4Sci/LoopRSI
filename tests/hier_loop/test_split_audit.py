import json
import tempfile
import unittest
from pathlib import Path

from hier_loop.split_audit import SplitAudit, SplitLeakageError


class SplitAuditTests(unittest.TestCase):
    def test_target_overlap_fails(self):
        with self.assertRaisesRegex(SplitLeakageError, "overlap"):
            SplitAudit.from_targets(
                train=("GENE1", "GENE2"),
                validation=("GENE2", "GENE3"),
                test=("GENE4",),
            )

    def test_audit_stores_only_target_hashes_and_counts(self):
        audit = SplitAudit.from_targets(
            train=("GENE1", "GENE2"),
            validation=("GENE3",),
            test=("GENE4", "GENE5"),
        )
        value = audit.to_dict()
        serialized = json.dumps(value)

        self.assertEqual(value["splits"]["train"]["count"], 2)
        self.assertEqual(len(value["splits"]["test"]["target_sha256"]), 2)
        self.assertNotIn("GENE1", serialized)
        self.assertNotIn("expression", serialized.lower())
        self.assertTrue(value["target_splits_disjoint"])

    def test_split_audit_file_is_deterministic(self):
        first = SplitAudit.from_targets(
            train=("B", "A"), validation=("C",), test=("D",)
        )
        second = SplitAudit.from_targets(
            train=("A", "B"), validation=("C",), test=("D",)
        )
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split_audit.json"
            first.write(path)
            self.assertEqual(json.loads(path.read_text()), first.to_dict())


if __name__ == "__main__":
    unittest.main()
