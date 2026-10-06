import json
import tempfile
import unittest
from pathlib import Path

from ai4ai.plugin_manifest import load_task_plugin_manifest
from hier_loop.knowledge_bridge import KnowledgeBridge
from hier_loop.knowledge_worker import KnowledgeDrivenVcc25Worker
from hier_loop.layer_agent import HistoricalDecisionBackend, LayerDecisionError
from hier_loop.memory import MemoryEntry, MemoryManager, MemoryStore
from hier_loop.validation_record import make_validation_record
from hier_loop.vcc25_worker import Vcc25Worker


ROOT = Path(__file__).resolve().parents[2]


class ValidationSelectionTests(unittest.TestCase):
    def test_autonomous_runner_is_passed_to_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = Vcc25Worker({"worker": {"mode": "real", "result_root": tmp,
                "trial_defaults": {"autonomous_runner": "tasks/vcc25/implementation/scripts/run_official_h1_autonomous_candidate.sh"}}})
            from unittest.mock import patch
            import subprocess
            with patch("hier_loop.vcc25_worker.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 2, "", "expected stop")
                worker._run_real_trial("autonomous_research_candidate", 20260907, 0, 1)
            self.assertIn("--autonomous-runner", run.call_args.args[0])

    def test_record_rejects_unverified_and_missing_split_audit(self):
        args = dict(method="official_h1", variant="candidate_g_promoted_delta_model",
                    seed=20260907, metric="pearson_delta", elapsed_seconds=2.0)
        good = make_validation_record(**args, raw={
            "status": "ok", "metrics": {"pearson_delta": 0.2},
            "protocol": {"test_expression_used_for_selection": False},
        })
        self.assertEqual(good["metric_value"], 0.2)
        self.assertTrue(good["selection_feedback_allowed"])
        bad = make_validation_record(**args, raw={"status": "ok", "metrics": {"pearson_delta": 0.9}})
        self.assertIsNone(bad["metric_value"])
        self.assertFalse(bad["selection_feedback_allowed"])
        other = make_validation_record(**{**args, "method": "gears"}, raw={
            "status": "ok", "metrics": {"pearson_delta": 0.9},
            "test_expression_read": False,
        })
        self.assertFalse(other["selection_feedback_allowed"])

    def test_real_round_failure_then_switch_and_persist_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(MemoryStore(str(Path(tmp) / "memory")), str(Path(tmp) / "decisions"))
            config = {"worker": {"mode": "real", "result_root": str(Path(tmp) / "results")},
                      "memory": {"enabled": True, "baseline": 0.05}}
            execution = Vcc25Worker(config, memory_manager=manager)
            outcomes = [
                {"status": "failed", "error": "missing runner", "test_expression_read": False},
                {"status": "ok", "metrics": {"pearson_delta": 0.12},
                 "protocol": {"test_expression_used_for_selection": False}},
            ]
            execution._run_real_trial = lambda *args: outcomes.pop(0)
            bridge = KnowledgeBridge(load_task_plugin_manifest(ROOT / "tasks/vcc25/task_plugin.yaml"))
            worker = KnowledgeDrivenVcc25Worker(bridge, HistoricalDecisionBackend(manager), execution)
            first = worker.step("L5", 0, 1, {"task_id": "vcc25"})
            second = worker.step("L5", 1, 1, {"task_id": "vcc25"})
            self.assertEqual(first.status, "failed")
            self.assertIsNone(first.score)
            self.assertEqual(first.action["variant"], "candidate_g_promoted_delta_model")
            self.assertEqual(second.status, "ok")
            self.assertEqual(second.action["variant"], "autonomous_research_candidate")
            self.assertEqual(second.score, 0.12)
            records = sorted((Path(tmp) / "results").glob("*.validation.json"))
            self.assertEqual(len(records), 2)
            self.assertEqual({json.loads(path.read_text())["status"] for path in records}, {"failed", "passed"})
            entries = manager.store.load("L5")
            self.assertEqual([entry.status for entry in entries], ["failed", "passed"])
            self.assertIsNone(entries[0].reward)
            self.assertAlmostEqual(entries[1].reward, 0.07)
            self.assertIsNotNone(entries[0].wall_seconds)

    def test_all_failed_variants_stop_before_reexecution(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(MemoryStore(str(Path(tmp) / "memory")), str(Path(tmp) / "decisions"))
            for variant in ("candidate_g_promoted_delta_model", "autonomous_research_candidate"):
                manager.remember_live(MemoryEntry(layer="L5", context_fingerprint="x", variant=variant,
                                                  seed=20260907, status="failed", metric_source="real"))
            bridge = KnowledgeBridge(load_task_plugin_manifest(ROOT / "tasks/vcc25/task_plugin.yaml"))
            worker = KnowledgeDrivenVcc25Worker(bridge, HistoricalDecisionBackend(manager))
            with self.assertRaisesRegex(LayerDecisionError, "all executable"):
                worker.step("L5", 2, 1, {"task_id": "vcc25"})


if __name__ == "__main__":
    unittest.main()
