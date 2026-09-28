from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omni_ar.initialization import RoughIdeaEngine
from omni_ar.initialization.llm_questions import (
    _decode_json_with_local_repairs, fallback_question_plan, generate_question_plan,
    generic_capabilities, normalize_question_plan, option_catalog, resolve_task,
)


REPOSITORY = Path(__file__).resolve().parents[2]


class QuestionFallbackTests(unittest.TestCase):
    def test_invalid_task_resolution_uses_explicit_registered_task(self) -> None:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def generate_json(self, **_kwargs):
                return SimpleNamespace(text="no json", input_tokens=1, output_tokens=1)

        def decode(_text):
            raise ValueError("response contains no valid JSON object or array")

        with patch(
            "omni_ar.initialization.llm_questions._heuresis_api",
            return_value=(Client, decode),
        ):
            result = resolve_task(
                Path("."), "improve the bound task",
                [{"id": "vcc25"}], task_hint="vcc25", model="test-model",
            )
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["task_id"], "vcc25")
        self.assertEqual(result["generator"]["provider"], "explicit_task_fallback")

    def test_invalid_task_resolution_without_hint_remains_closed(self) -> None:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def generate_json(self, **_kwargs):
                return SimpleNamespace(text="no json", input_tokens=1, output_tokens=1)

        with patch(
            "omni_ar.initialization.llm_questions._heuresis_api",
            return_value=(Client, lambda _text: (_ for _ in ()).throw(ValueError("no json"))),
        ):
            with self.assertRaisesRegex(Exception, "valid task resolution"):
                resolve_task(
                    Path("."), "unknown task", [{"id": "vcc25"}], model="test-model",
                )

    def test_invalid_boyue_text_uses_reviewed_fallback(self) -> None:
        class Client:
            def __init__(self, **_kwargs):
                pass

            def generate_json(self, **_kwargs):
                return SimpleNamespace(text="I could not produce JSON", input_tokens=1, output_tokens=1)

        def decode(_text):
            raise ValueError("response contains no valid JSON object or array")

        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            schema = Path(__file__).resolve().parents[1] / "schemas/generated_questions.schema.json"
            (repository / "omni_ar/schemas").mkdir(parents=True)
            (repository / "omni_ar/schemas/generated_questions.schema.json").write_bytes(schema.read_bytes())
            with patch(
                "omni_ar.initialization.llm_questions._heuresis_api",
                return_value=(Client, decode),
            ):
                plan = generate_question_plan(
                    repository, "improve an unfamiliar task",
                    {"type": "other", "modality": "other", "description": "test"},
                    generic_capabilities(), model="test-model", question_count=5,
                )
        self.assertEqual(plan["generator"]["provider"], "reviewed_fallback")
        self.assertEqual(len(plan["questions"]), 5)
        self.assertIn("research_goal", {item["id"] for item in plan["questions"]})
        self.assertIn("requested_scopes", {item["id"] for item in plan["questions"]})
        by_id = {item["id"]: item for item in plan["questions"]}
        self.assertEqual(by_id["research_goal"]["default"], "improve")
        self.assertEqual(by_id["exploration_level"]["default"], "balanced")
        attempts = plan["generator"]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["raw_response"], "I could not produce JSON")
        self.assertEqual(attempts[0]["status"], "invalid_response")
        self.assertEqual(plan["generator"]["input_tokens"], 2)
        self.assertEqual(plan["generator"]["output_tokens"], 2)

    def test_local_json_repair_is_audited(self) -> None:
        value, repairs = _decode_json_with_local_repairs(
            "```json\n{\"ok\": true,}\n```", json.loads,
        )
        self.assertEqual(value, {"ok": True})
        self.assertEqual(repairs[-1]["status"], "succeeded")
        self.assertIn("remove_trailing_commas", repairs[-1]["step"])

    def test_free_text_question_omission_is_repaired_and_audited(self) -> None:
        repairs: list[dict] = []
        normalized = normalize_question_plan({
            "schema_version": "omni-ar-generated-questions/v1",
            "summary": "test",
            "questions": [{
                "id": "research_details", "type": "free_text",
                "title": "补充要求", "why": "补充约束", "default": "",
            }],
        }, repairs)
        self.assertEqual(normalized["questions"][0]["options"], [])
        self.assertEqual(repairs[-1]["step"], "add_empty_options_to_free_text")

    def test_real_qa_bundle_writes_transcript_sources_and_constraint_trace(self) -> None:
        engine = RoughIdeaEngine(REPOSITORY)
        task = REPOSITORY / "tasks/vcc25/task_spec.yaml"
        _path, _spec, capabilities = engine.task_context(task)
        plan = fallback_question_plan(
            option_catalog(capabilities), 5, model="test", reason="test",
        )
        plan["questions"][0]["source"] = "llm"

        def resolver(*_args, **_kwargs):
            return {
                "schema_version": "omni-ar-task-resolution/v1", "status": "selected",
                "task_id": "vcc25", "confidence": 1.0, "reason": "test",
                "question": "", "task_profile": None,
                "generator": {"provider": "boyue", "model": "test"},
            }

        # Rough idea, five planned questions and Adapter protocol binding.
        prompts = iter(["improve VCC", "", "", "", "", "", ""])
        answers = engine.ask(
            task, input_fn=lambda _prompt: next(prompts), output_fn=lambda _line: None,
            question_generator=lambda *_args, **_kwargs: plan, task_resolver=resolver,
        )
        rough = engine.build(task, answers, initialization_id="qa-audit-test", confirmed=True)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = engine.write_bundle(Path(temporary), answers, rough)
            events = [json.loads(line) for line in bundle["qa_transcript"].read_text().splitlines()]
            generation = json.loads(bundle["question_generation"].read_text())
        sources = {item["source"] for item in events}
        self.assertTrue({"llm", "task_adapter", "fixed_safety_check"}.issubset(sources))
        self.assertEqual(
            rough["provenance"]["constraint_sources"]["evaluation.protocol"]["source"],
            "task_adapter",
        )
        self.assertIn("attempts", generation)

    def test_previous_answers_trigger_acceptance_followup(self) -> None:
        engine = RoughIdeaEngine(REPOSITORY)
        task = REPOSITORY / "tasks/vcc25/task_spec.yaml"
        _path, _spec, capabilities = engine.task_context(task)
        plan = fallback_question_plan(
            option_catalog(capabilities), 5, model="test", reason="test",
        )

        def resolver(*_args, **_kwargs):
            return {
                "schema_version": "omni-ar-task-resolution/v1", "status": "selected",
                "task_id": "vcc25", "confidence": 1.0, "reason": "test",
                "question": "", "task_profile": None,
                "generator": {"provider": "boyue", "model": "test"},
            }

        # Rough idea, five planned questions, dynamic success criteria, protocol binding.
        prompts = iter(["improve VCC", "", "", "", "", "", "", ""])
        with patch(
            "omni_ar.initialization.llm_questions.generate_question_plan",
            return_value=plan,
        ):
            answers = engine.ask(
                task, input_fn=lambda _prompt: next(prompts), output_fn=lambda _line: None,
                task_resolver=resolver,
            )
        dynamic = answers["_question_plan"]["dynamic_questions"]
        self.assertEqual(dynamic[0]["id"], "success_criteria")
        self.assertIn("multi_seed_stability", answers["success_criteria"])


if __name__ == "__main__":
    unittest.main()
