import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hier_loop.paper_reproduction import METHODS, run_paper_reproduction


ROOT = Path(__file__).resolve().parents[2]


class PaperReproductionTests(unittest.TestCase):
    def test_eight_remote_attempts_record_network_timeout_without_claiming_reproduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment"
            shutil.copytree(ROOT / "knowledge" / "vcc25", experiment / "knowledge" / "vcc25")
            (experiment / "hier_loop").mkdir()
            prior = experiment / "artifacts" / "phase-d-validation" / "stream-cuda-full-20260930-r2" / "lineage_summary.json"
            prior.parent.mkdir(parents=True)
            prior.write_text(json.dumps({
                "status": "pass", "official_score_claim": False,
                "data_contract": {"test_expression_read": False},
                "results": {"candidate_a": {"aggregate": {"delta_correlation_mean": 0.1}}},
            }))

            def timeout(command, **kwargs):
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])

            output = experiment / "artifacts" / "paper-attempt"
            summary = run_paper_reproduction(experiment, output, runner=timeout, code_root=experiment)
            self.assertEqual(summary["evidence_cards"], 8)
            self.assertFalse(summary["candidate_request_generated"])
            self.assertEqual(summary["methods"]["lingshu-cell"], "partial")
            self.assertEqual(set(summary["methods"]), set(METHODS))
            for method in METHODS:
                with self.subTest(method=method):
                    card = json.loads((output / "cards" / f"{method}.json").read_text())
                    self.assertEqual(card["status"], summary["methods"][method])
                    self.assertIn("timed out", card["commands_attempted"][0]["error"])
                    self.assertTrue(card["paper_card_id"].startswith("kb:paper:"))
                    self.assertTrue(card["code_card_id"].startswith("kb:repo:"))
                    self.assertTrue(card["model_card_id"].startswith("kb:model:"))
                    self.assertFalse(card["test_expression_read"])
                    self.assertFalse(card["gpu_used"])
                    self.assertIsNone(card["validation_metrics"])

    def test_rejects_output_outside_independent_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "output must be inside"):
                run_paper_reproduction(root / "experiment", root / "elsewhere", code_root=root / "experiment")


if __name__ == "__main__":
    unittest.main()
