import json
import tempfile
import unittest
from pathlib import Path

from hier_loop.cli import main as cli_main
from hier_loop.knowledge_experiment import ExperimentGateError, KnowledgeExperiment


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rsi_step0/hier_vcc25_knowledge.json"


class KnowledgeExperimentTests(unittest.TestCase):
    def test_preflight_runs_all_layers_and_writes_audit_artifacts_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "preflight"
            result = KnowledgeExperiment.from_config(CONFIG, workspace).run_preflight()

            self.assertEqual(result["layers_completed"], ["L1", "L2", "L3", "L4", "L5"])
            self.assertFalse(result["gpu_invoked"])
            self.assertEqual(result["executed_commands"], [])
            expected = {
                "run_manifest.json",
                "layer_decisions.jsonl",
                "candidate_request.json",
                "candidate.patch",
                "candidate_code_sha256.json",
                "split_audit.json",
                "preflight_result.json",
                "summary.json",
            }
            self.assertTrue(expected.issubset({path.name for path in workspace.iterdir()}))

    def test_each_layer_artifact_contains_required_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "preflight"
            KnowledgeExperiment.from_config(CONFIG, workspace).run_preflight()
            rows = [
                json.loads(line)
                for line in (workspace / "layer_decisions.jsonl").read_text().splitlines()
            ]

            self.assertEqual([row["layer"] for row in rows], ["L1", "L2", "L3", "L4", "L5"])
            for row in rows:
                self.assertTrue(row["skill_id"].startswith("vcc25.lingshu."))
                self.assertTrue(row["card_ids"])
                self.assertEqual(len(row["content_hash"]), 64)
                self.assertGreater(row["token_budget"], 0)
                self.assertTrue(row["source"].startswith("knowledge:vcc25:"))
                self.assertIsNone(row["evaluation_authority"])

    def test_preflight_refuses_destination_equal_to_source_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            config = Path(temporary) / "config.json"
            raw = json.loads(CONFIG.read_text())
            raw["source_checkouts"] = [str(source), str(Path(temporary) / "other")]
            config.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(ExperimentGateError, "source checkout"):
                KnowledgeExperiment.from_config(config, source).run_preflight()

    def test_final_refuses_without_frozen_candidate_and_promotion_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "experiment"
            experiment = KnowledgeExperiment.from_config(CONFIG, workspace)
            with self.assertRaisesRegex(ExperimentGateError, "frozen candidate"):
                experiment.run_final()

            workspace.mkdir(parents=True)
            (workspace / "candidate_code_sha256.json").write_text(
                json.dumps({"candidate_hash": "sha256:" + "a" * 64, "frozen": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ExperimentGateError, "promotion record"):
                experiment.run_final()

    def test_cli_knowledge_preflight_uses_recording_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "cli-preflight"
            exit_code = cli_main(
                [
                    "knowledge-preflight",
                    "--config",
                    str(CONFIG),
                    "--workspace",
                    str(workspace),
                ]
            )
            self.assertEqual(exit_code, 0)
            manifest = json.loads((workspace / "run_manifest.json").read_text())
            self.assertEqual(manifest["decision_backend"], "RecordingDecisionBackend")
            self.assertEqual(manifest["phase"], "preflight")

    def test_archived_copy_uses_configured_source_revision_without_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "archived.json"
            raw = json.loads(CONFIG.read_text())
            raw["source_revision"] = "tested-commit-123"
            config.write_text(json.dumps(raw), encoding="utf-8")
            experiment = KnowledgeExperiment.from_config(
                config,
                Path(temporary) / "preflight",
            )
            experiment._git_revision = lambda: None
            experiment.run_preflight()
            manifest = json.loads(
                (Path(temporary) / "preflight/run_manifest.json").read_text()
            )
            self.assertEqual(manifest["source_revision"], "tested-commit-123")


if __name__ == "__main__":
    unittest.main()
