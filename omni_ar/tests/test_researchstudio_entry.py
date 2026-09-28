from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ResearchStudio-main/scripts/rough_idea_entry.py"
)
SPEC = importlib.util.spec_from_file_location("rough_idea_entry", SCRIPT)
assert SPEC and SPEC.loader
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)

AUTHOR_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "ResearchStudio-main/scripts/ideaspark_boyue_author.py"
)
AUTHOR_SPEC = importlib.util.spec_from_file_location("ideaspark_boyue_author", AUTHOR_SCRIPT)
assert AUTHOR_SPEC and AUTHOR_SPEC.loader
AUTHOR = importlib.util.module_from_spec(AUTHOR_SPEC)
AUTHOR_SPEC.loader.exec_module(AUTHOR)
IDEASPARK_RUN = (
    Path(__file__).resolve().parents[2]
    / "ResearchStudio-main/ResearchStudio-Idea/skills/idea_spark/scripts/run.py"
)


class ResearchStudioPhase0ReuseTests(unittest.TestCase):
    def test_identical_confirmed_input_reuses_complete_phase0(self) -> None:
        rough = {
            "task": {"name": "vcc25"},
            "dataset": {"ref": "vcc25@2", "version": 2, "usage_mode": "full"},
            "evaluation": {"protocol": "heldout_guide", "primary_metric": {"name": "pearson"}},
            "user_notes": {
                "rough_idea": "improve vcc",
                "clarifications": {"research_details": "fixed budget"},
            },
        }
        literature = {
            "query": "single cell perturbation",
            "queries": ["single cell perturbation"],
            "task_context": "vcc",
        }
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            old = repository / "research_initializations/old"
            old.mkdir(parents=True)
            (old / "rough_idea.yaml").write_text(
                yaml.safe_dump(rough, sort_keys=False), encoding="utf-8"
            )
            source = old / "autoresearch/planning/researchstudio/ideaspark_live/phase0"
            source.mkdir(parents=True)
            for name in ENTRY.PHASE0_REQUIRED:
                (source / name).write_text("real\n" if name == ".lit_grounding_mode" else "{}\n")
            current = repository / "research_initializations/current/rough_idea.yaml"
            current.parent.mkdir(parents=True)
            current_rough = {
                **rough,
                "user_notes": {
                    "rough_idea": "improve vcc",
                    "forbidden_directions": "fixed budget",
                },
            }
            current.write_text(yaml.safe_dump(current_rough, sort_keys=False), encoding="utf-8")
            target = repository / "target/phase0"

            reused = ENTRY.reuse_matching_phase0(
                repository, current, current_rough, literature, target,
            )

            self.assertEqual(reused, source)
            self.assertTrue((target / "lit_results.json").is_file())
            self.assertTrue((target / ".omni_ar_phase0_reuse.json").is_file())

    def test_changed_user_idea_does_not_reuse_phase0(self) -> None:
        rough = {
            "task": {"name": "vcc25"}, "dataset": {"ref": "vcc25@2"},
            "evaluation": {"protocol": "heldout_guide"},
            "user_notes": {"rough_idea": "new idea"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            current = repository / "research_initializations/current/rough_idea.yaml"
            current.parent.mkdir(parents=True)
            current.write_text(yaml.safe_dump(rough), encoding="utf-8")
            reused = ENTRY.reuse_matching_phase0(
                repository, current, rough,
                {"query": "q", "queries": ["q"], "task_context": "vcc"},
                repository / "target/phase0",
            )
            self.assertIsNone(reused)


class ResearchStudioJSONRetryTests(unittest.TestCase):
    def test_phase2_prepare_matches_provider_prefixed_paper_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "phase1").mkdir()
            (root / "phase0").mkdir()
            (root / "phase1/phase1_output.json").write_text(json.dumps({
                "closest_adjacent": [{"paper_id": "abc123"}],
            }), encoding="utf-8")
            (root / "phase0/lit_results.json").write_text(json.dumps([{
                "paper_id": "semanticscholar:abc123", "title": "A paper",
            }]), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(IDEASPARK_RUN), "phase2_prepare", "--dir", str(root)],
                text=True, capture_output=True, check=False,
            )
            matched = json.loads((root / "phase2_generate/closest_abstracts.json").read_text())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(matched[0]["paper_id"], "semanticscholar:abc123")
        self.assertNotIn("WARN", result.stderr)

    def test_non_object_json_is_retried(self) -> None:
        responses = iter(["[]", '{"status": "ok"}'])

        class Client:
            def generate_json(self, **_kwargs):
                return SimpleNamespace(text=next(responses))

        with patch.object(AUTHOR, "client", return_value=(Client(), "test")), patch.dict(
            os.environ, {"IDEASPARK_BOYUE_JSON_RETRIES": "2"}, clear=False,
        ):
            value = AUTHOR.call_json("prompt", max_completion_tokens=512)
        self.assertEqual(value, {"status": "ok"})

    def test_empty_api_attempt_is_retried_with_fresh_client(self) -> None:
        calls = iter([RuntimeError("empty assistant content"), '{"status": "ok"}'])

        class Client:
            def generate_json(self, **_kwargs):
                value = next(calls)
                if isinstance(value, Exception):
                    raise value
                return SimpleNamespace(text=value)

        with patch.object(AUTHOR, "client", return_value=(Client(), "test")), patch.dict(
            os.environ, {"IDEASPARK_BOYUE_JSON_RETRIES": "2"}, clear=False,
        ):
            value = AUTHOR.call_json("prompt", max_completion_tokens=512)
        self.assertEqual(value, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
