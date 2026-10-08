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
    build_layer_prompt,
    validate_layer_decision,
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

    def test_l5_includes_multiple_model_candidates_for_selection(self):
        worker = self.make_worker()
        result = worker.step("L5", 0, 1, {"task_id": "vcc25"})
        content = result.action["knowledge"]["content"]
        model_lines = [line for line in content.splitlines() if line.startswith("card_id: kb:model:")]
        self.assertGreaterEqual(len(model_lines), 2)

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

    # -- autonomous model selection tests ------------------------------------

    def test_l5_prompt_contains_candidate_profiles(self):
        """L5 prompt must include structured candidate_profiles for comparison."""
        worker = self.make_worker()
        worker.step("L1", 0, 1, {"task_id": "vcc25"})
        worker.step("L2", 0, 2, {"task_id": "vcc25"})
        worker.step("L3", 0, 3, {"task_id": "vcc25"})
        worker.step("L4", 0, 4, {"task_id": "vcc25"})
        result = worker.step("L5", 0, 5, {"task_id": "vcc25"})
        prompt_text = result.action.get("knowledge", {}).get("content", "")
        # The prompt is stored in backend.requests; verify it has candidate_profiles
        backend = worker.backend
        l5_request = backend.requests[-1]
        prompt_data = json.loads(l5_request.prompt)
        self.assertIn("candidate_profiles", prompt_data)
        self.assertGreaterEqual(len(prompt_data["candidate_profiles"]), 2)
        first = prompt_data["candidate_profiles"][0]
        self.assertIn("card_id", first)
        self.assertIn("execution_readiness", first)
        self.assertIn("adapter_id", first)
        self.assertIn("compatibility", first)

    def test_l5_decision_populates_selected_model_and_rejected(self):
        """RecordingDecisionBackend must populate selected_model_id and rejected_alternatives."""
        worker = self.make_worker()
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            result = worker.step(layer, 0, step, {"task_id": "vcc25"})
        coding = result.action["decision"]["coding_request"]
        self.assertTrue(coding["selected_model_id"].startswith("kb:model:"))
        self.assertTrue(coding["selected_repository_id"].startswith("kb:repo:"))
        self.assertTrue(coding["selection_reason"])
        self.assertIsInstance(coding["rejected_alternatives"], list)
        # Since we inject >=2 models, at least one should be rejected
        self.assertGreaterEqual(len(coding["rejected_alternatives"]), 1)
        for alt in coding["rejected_alternatives"]:
            self.assertTrue(alt.startswith("kb:model:"))

    def test_l5_selected_model_is_highest_ranked(self):
        """The selected model must match the first model card in the knowledge content."""
        worker = self.make_worker()
        for step, layer in enumerate(("L1", "L2", "L3", "L4"), start=1):
            worker.step(layer, 0, step, {"task_id": "vcc25"})
        result = worker.step("L5", 0, 5, {"task_id": "vcc25"})
        content = result.action["knowledge"]["content"]
        model_lines = [line for line in content.splitlines() if line.startswith("card_id: kb:model:")]
        first_model_id = model_lines[0][len("card_id: "):].strip()
        coding = result.action["decision"]["coding_request"]
        self.assertEqual(coding["selected_model_id"], first_model_id)

    def test_l5_schema_requires_selection_fields(self):
        """The L5 JSON schema must require selected_model_id and rejected_alternatives."""
        schema_path = ROOT / "hier_loop" / "schemas" / "layer_decision.L5.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        required = schema["properties"]["coding_request"]["required"]
        self.assertIn("selected_model_id", required)
        self.assertIn("selected_repository_id", required)
        self.assertIn("selection_reason", required)
        self.assertIn("rejected_alternatives", required)

    def test_validate_layer_decision_rejects_missing_selection_fields(self):
        """validate_layer_decision must reject L5 decisions without selected_model_id."""
        worker = self.make_worker()
        for step, layer in enumerate(("L1", "L2", "L3", "L4"), start=1):
            worker.step(layer, 0, step, {"task_id": "vcc25"})
        result = worker.step("L5", 0, 5, {"task_id": "vcc25"})
        coding = dict(result.action["decision"]["coding_request"])
        del coding["selected_model_id"]
        del coding["rejected_alternatives"]
        bad_decision = {
            "layer": "L5",
            "decision_type": "coding_request",
            "coding_request": coding,
        }
        with self.assertRaises(LayerDecisionError):
            validate_layer_decision("L5", bad_decision)

    def test_l5_decision_is_forwarded_with_selected_model_id(self):
        """The execution worker must receive the selected_model_id in the forwarded decision."""
        execution = Vcc25Worker({"worker": {"mode": "offline", "trial_defaults": {"require_knowledge_selection": True}}})
        worker = KnowledgeDrivenVcc25Worker(
            KnowledgeBridge(MANIFEST), RecordingDecisionBackend(), execution_worker=execution
        )
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            result = worker.step(layer, 0, step, {"task_id": "vcc25"})
        forwarded = result.action["knowledge_decision"]["coding_request"]
        self.assertTrue(forwarded["selected_model_id"].startswith("kb:model:"))
        self.assertTrue(forwarded["selected_repository_id"].startswith("kb:repo:"))
        self.assertGreaterEqual(len(forwarded["rejected_alternatives"]), 1)

    def test_l5_rejected_alternatives_grew_with_expanded_candidates(self):
        """With expanded L5 candidates, rejected_alternatives should have >=3 entries."""
        worker = self.make_worker()
        for step, layer in enumerate(("L1", "L2", "L3", "L4", "L5"), start=1):
            result = worker.step(layer, 0, step, {"task_id": "vcc25"})
        coding = result.action["decision"]["coding_request"]
        self.assertGreaterEqual(len(coding["rejected_alternatives"]), 3)

    def test_l5_candidate_profiles_count_matches_rendered_models(self):
        """candidate_profiles in the prompt should have >=4 entries with expanded budget."""
        worker = self.make_worker()
        for step, layer in enumerate(("L1", "L2", "L3", "L4"), start=1):
            worker.step(layer, 0, step, {"task_id": "vcc25"})
        worker.step("L5", 0, 5, {"task_id": "vcc25"})
        l5_request = worker.backend.requests[-1]
        prompt_data = json.loads(l5_request.prompt)
        self.assertGreaterEqual(len(prompt_data["candidate_profiles"]), 4)


if __name__ == "__main__":
    unittest.main()
