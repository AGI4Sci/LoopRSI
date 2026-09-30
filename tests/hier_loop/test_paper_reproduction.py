import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from hier_loop.paper_reproduction import METHODS, STATIC_ENTRYPOINTS, run_paper_reproduction


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

    def test_local_source_manifest_records_pinned_cpu_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment"
            shutil.copytree(ROOT / "knowledge" / "vcc25", experiment / "knowledge" / "vcc25")
            records = []
            for method in METHODS:
                repository = json.loads((experiment / "knowledge" / "vcc25" / "repositories" / f"{method}.json").read_text())
                source = experiment / "sources" / method
                entrypoint = source / STATIC_ENTRYPOINTS[method]
                entrypoint.parent.mkdir(parents=True, exist_ok=True)
                entrypoint.write_text("pass\n")
                archive = experiment / "sources" / f"{method}.tar.gz"
                archive.write_bytes(method.encode())
                records.append({
                    "method": method, "status": "fetched", "official_url": repository["official_url"],
                    "source_dir": str(source), "archive": str(archive),
                    "archive_sha256": hashlib.sha256(method.encode()).hexdigest(),
                    "revision": "a" * 40,
                })
            manifest = experiment / "sources" / "manifest.json"
            manifest.write_text(json.dumps({"records": records}))

            def successful_check(command, **kwargs):
                return subprocess.CompletedProcess(command, 0, "syntax-ok\n", "")

            output = experiment / "artifacts" / "local-source-check"
            summary = run_paper_reproduction(
                experiment, output, runner=successful_check,
                code_root=experiment, source_manifest=manifest,
            )
            self.assertEqual(summary["authority"], "local_looprsi_attempt")
            self.assertFalse(summary["method_reproduction_complete"])
            self.assertEqual(summary["evidence_cards"], 8)
            for method in METHODS:
                card = json.loads((output / "cards" / f"{method}.json").read_text())
                self.assertEqual(card["observed_source_revision"], "a" * 40)
                self.assertTrue(card["card_id"].endswith(":local"))
                self.assertEqual(card["commands_attempted"][0]["exit_code"], 0)
                self.assertFalse(card["gpu_used"])


if __name__ == "__main__":
    unittest.main()
