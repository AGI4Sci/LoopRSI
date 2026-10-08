import json
import subprocess
import unittest
from pathlib import Path

from ai4ai.plugin_manifest import load_task_plugin_manifest
from hier_loop.knowledge_bridge import KnowledgeBridge
from hier_loop.knowledge_worker import KnowledgeDrivenVcc25Worker
from hier_loop.vcc25_worker import Vcc25Worker
from hier_loop.layer_agent import (
    CodexJsonDecisionBackend,
    LayerDecisionError,
    RecordingDecisionBackend,
    StructuredDecisionBackend,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = load_task_plugin_manifest(ROOT / "tasks" / "vcc25" / "task_plugin.yaml")
SCHEMA = ROOT / "hier_loop" / "schemas" / "layer_decision.schema.json"
L1_SCHEMA = ROOT / "hier_loop" / "schemas" / "layer_decision.L1.schema.json"


class _MalformedBackend:
    def decide(self, request):
        return {"layer": request.layer, "decision_type": "direction"}


class KnowledgeWorkerTests(unittest.TestCase):
    def make_worker(self, backend=None):
        return KnowledgeDrivenVcc25Worker(
            KnowledgeBridge(MANIFEST),
            backend or RecordingDecisionBackend(),
        )

    def test_layers_produce_typed_decisions_and_l5_coding_request(self):
        worker = self.make_worker()
        fields = {
            "L1": "direction",
            "L2": "problem",
            "L3": "hypothesis",
            "L4": "mechanism",
        }
        results = {}
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            results[layer] = worker.step(layer, 0, step, {"task_id": "vcc25"})

        for layer, field in fields.items():
            self.assertTrue(results[layer].action["decision"][field])
        coding = results["L5"].action["decision"]["coding_request"]
        self.assertTrue(coding["allowed_paths"])
        self.assertTrue(coding["hypothesis"])
        self.assertTrue(coding["activation_diagnostics"])
        self.assertTrue(coding["train_command"])
        self.assertTrue(coding["validation_command"])
        self.assertTrue(coding["expected_artifacts"])

    def test_every_action_contains_knowledge_provenance(self):
        worker = self.make_worker()
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            result = worker.step(layer, 0, step, {"task_id": "vcc25"})
            provenance = result.action["knowledge"]
            self.assertTrue(provenance["skill_id"].startswith("vcc25.lingshu."))
            self.assertTrue(provenance["card_ids"])
            self.assertEqual(len(provenance["content_hash"]), 64)
            self.assertIn("evaluation_authority", result.action)

    def test_prior_layer_decisions_flow_forward(self):
        backend = RecordingDecisionBackend()
        worker = self.make_worker(backend)
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            worker.step(layer, 0, step, {"task_id": "vcc25"})

        self.assertEqual([len(request.prior_decisions) for request in backend.requests], [0, 1, 2, 3, 4])
        self.assertEqual(backend.requests[-1].prior_decisions[0]["decision_type"], "direction")
        self.assertEqual(backend.prompts, [request.prompt for request in backend.requests])

    def test_planning_layers_never_fabricate_numeric_scores(self):
        worker = self.make_worker()
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            result = worker.step(layer, 0, step, {"task_id": "vcc25"})
            self.assertIsNone(result.score)
            self.assertNotIn("pearson_delta", result.metrics)

    def test_malformed_backend_output_fails_the_step(self):
        worker = self.make_worker(_MalformedBackend())
        with self.assertRaises(LayerDecisionError):
            worker.step("L1", 0, 1, {"task_id": "vcc25"})

    def test_l5_decision_is_forwarded_to_execution_worker(self):
        execution = Vcc25Worker({"worker": {"mode": "offline", "trial_defaults": {"require_knowledge_selection": True}}})
        worker = KnowledgeDrivenVcc25Worker(
            KnowledgeBridge(MANIFEST), RecordingDecisionBackend(), execution_worker=execution
        )
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            result = worker.step(layer, 0, step, {"task_id": "vcc25"})
        self.assertTrue(result.action["knowledge_selected"])
        self.assertEqual(result.action["variant"], "autonomous_research_candidate")
        self.assertTrue(result.action["knowledge"]["card_ids"])

    def test_codex_backend_is_ephemeral_read_only_and_parses_last_json_object(self):
        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))
            payload = {
                "layer": "L1",
                "decision_type": "direction",
                "direction": "compare train-only priors",
                "rationale": "tests a bounded source of signal",
            }
            Path(command[command.index("--output-last-message") + 1]).write_text(
                json.dumps(payload), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "progress\n", "")

        recording = RecordingDecisionBackend()
        worker = self.make_worker(recording)
        worker.step("L1", 0, 1, {"task_id": "vcc25"})
        request = recording.requests[0]
        backend = CodexJsonDecisionBackend(SCHEMA, runner=fake_runner)
        decision = backend.decide(request)

        self.assertEqual(decision["decision_type"], "direction")
        command, kwargs = calls[0]
        self.assertEqual(
            command,
            [
                "codex",
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--skip-git-repo-check",
                "--model",
                "gpt-5.6-sol",
                "--config",
                "model_reasoning_effort=low",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(L1_SCHEMA),
                "--output-last-message",
                command[command.index("--output-last-message") + 1],
                "-",
            ],
        )
        schema = json.loads(L1_SCHEMA.read_text(encoding="utf-8"))
        self.assertNotIn("oneOf", schema)
        self.assertEqual(schema["properties"]["layer"]["const"], "L1")
        self.assertEqual(schema["properties"]["layer"]["type"], "string")
        self.assertEqual(schema["properties"]["decision_type"]["type"], "string")
        self.assertEqual(kwargs["input"], request.prompt)
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])

    def test_structured_backend_is_provider_agnostic_alias(self):
        self.assertIs(StructuredDecisionBackend, CodexJsonDecisionBackend)


if __name__ == "__main__":
    unittest.main()
