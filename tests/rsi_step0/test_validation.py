"""Unit tests for :mod:`rsi_step0.validation`.

Only depends on :mod:`rsi_step0.contracts` and :mod:`rsi_step0.validation`.
"""

import copy
import tempfile
import unittest
from unittest import mock

from rsi_step0 import contracts as C
from rsi_step0 import validation as V


def _valid_action():
    return {
        "action_id": "act-0001",
        "run_id": "run-vcc25_h1-0001",
        "round": 1,
        "step": 0,
        "operator": "draft",
        "mode": "thinking",
        "target": "proposal/vcc25_h1/0001",
        "decision": {"operator": "draft", "params": {"temperature": 0.7}},
        "budget": {"max_seconds": 300, "max_gpu_seconds": 120, "max_cost": 0.5},
        "provenance": {"parent_action_id": "root", "model": "deepseek", "seed": 42},
    }


class ValidateActionTests(unittest.TestCase):
    def test_valid_action_has_no_errors(self):
        self.assertEqual(V.validate_action(_valid_action()), [])

    def test_invalid_operator_and_mode(self):
        action = _valid_action()
        action["operator"] = "fly"
        action["mode"] = "dreaming"
        errors = V.validate_action(action)
        self.assertTrue(any("invalid operator 'fly'" in e for e in errors))
        self.assertTrue(any("invalid mode 'dreaming'" in e for e in errors))

    def test_missing_required_fields(self):
        action = _valid_action()
        del action["action_id"]
        del action["target"]
        errors = V.validate_action(action)
        self.assertIn("missing required field: action_id", errors)
        self.assertIn("missing required field: target", errors)

    def test_incomplete_budget_and_provenance(self):
        action = _valid_action()
        del action["budget"]["max_gpu_seconds"]
        del action["provenance"]["seed"]
        errors = V.validate_action(action)
        self.assertIn("budget missing required field: max_gpu_seconds", errors)
        self.assertIn("provenance missing required field: seed", errors)

    def test_wrong_types_flagged(self):
        action = _valid_action()
        action["round"] = "1"
        action["budget"]["max_cost"] = "cheap"
        action["provenance"]["seed"] = "42"
        errors = V.validate_action(action)
        self.assertTrue(any("field round must be an int" in e for e in errors))
        self.assertTrue(any("budget.max_cost must be a number" in e for e in errors))
        self.assertTrue(any("provenance.seed must be an int" in e for e in errors))

    def test_non_dict_action(self):
        errors = V.validate_action("not-a-dict")
        self.assertEqual(len(errors), 1)


class ValidateSchemaTests(unittest.TestCase):
    def _schema(self):
        return {
            "type": "object",
            "required": ["schema_version", "label"],
            "properties": {
                "schema_version": {
                    "type": "string",
                    "enum": [C.SCHEMA_VERSION],
                },
                "label": {"type": "string", "enum": list(C.LABELS)},
                "round": {"type": "integer"},
            },
        }

    def test_builtin_fallback_detects_missing_required_and_bad_enum(self):
        # Detects invalid records regardless of whether jsonschema or the
        # builtin structural checker is used (jsonschema presence differs
        # between local sandbox and the remote runtime).
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            import json

            json.dump(self._schema(), fh)
            schema_path = fh.name
        try:
            errors = V.validate_schema({"schema_version": C.SCHEMA_VERSION}, schema_path)
            self.assertTrue(errors, "expected missing-required errors")

            errors = V.validate_schema(
                {"schema_version": C.SCHEMA_VERSION, "label": "banana"},
                schema_path,
            )
            self.assertTrue(errors, "expected bad-enum errors")
            self.assertTrue(any("label" in e for e in errors))
        finally:
            import os

            os.unlink(schema_path)


class LoadYamlTests(unittest.TestCase):
    _TEXT = (
        "run:\n"
        "  mode: replay\n"
        "  seed: 42\n"
        "  rounds: 1\n"
        "reward:\n"
        "  metric: pearson_delta\n"
        "  baseline: 0.30614\n"
        "  lambda_c: 1.0e-6\n"
        "  use_cost_aware_reward: false\n"
        "operators: [draft, improve, crossover]\n"
    )

    def _write(self, text):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(text)
        tmp.close()
        return tmp.name

    def test_fallback_parser_when_pyyaml_missing(self):
        path = self._write(self._TEXT)
        try:
            with mock.patch.dict("sys.modules", {"yaml": None}):
                data = V.load_yaml(path)
            self.assertEqual(data["run"], {"mode": "replay", "seed": 42, "rounds": 1})
            self.assertEqual(data["reward"]["baseline"], 0.30614)
            self.assertEqual(data["reward"]["lambda_c"], 1.0e-6)
            self.assertIs(data["reward"]["use_cost_aware_reward"], False)
            self.assertEqual(data["operators"], ["draft", "improve", "crossover"])
            self.assertEqual(data["run"]["mode"], "replay")
        finally:
            import os

            os.unlink(path)

    def test_prefers_yaml_when_importable(self):
        class FakeYaml:
            @staticmethod
            def safe_load(stream):
                # If PyYAML is preferred, this is called and wins over the fallback.
                return {"run": {"mode": "from-yaml"}}

        path = self._write(self._TEXT)
        try:
            with mock.patch.dict("sys.modules", {"yaml": FakeYaml}):
                data = V.load_yaml(path)
            self.assertEqual(data["run"]["mode"], "from-yaml")
        finally:
            import os

            os.unlink(path)


class ValidateIsolationTests(unittest.TestCase):
    def test_no_overlap_is_ok(self):
        pi = [{"dedupe_key": "r1|1|p1|v0"}, {"dedupe_key": "r1|2|p2|v0"}]
        ft = [{"dedupe_key": "r1|3|p3|v0"}, {"dedupe_key": "r1|4|p4|v0"}]
        self.assertEqual(V.validate_isolation(pi, ft), [])

    def test_overlap_detected(self):
        pi = [{"dedupe_key": "r1|1|p1|v0"}, {"dedupe_key": "r1|2|p2|v0"}]
        ft = [{"dedupe_key": "r1|2|p2|v0"}, {"dedupe_key": "r1|3|p3|v0"}]
        errors = V.validate_isolation(pi, ft)
        self.assertTrue(any("r1|2|p2|v0" in e for e in errors))

    def test_non_list_inputs(self):
        self.assertTrue(V.validate_isolation([], None))


class ValidateReplayTests(unittest.TestCase):
    def _events(self):
        return [
            {"round": 1, "step": 0, "operator": "draft", "target": "t/0"},
            {"round": 1, "step": 1, "operator": "improve", "target": "t/1"},
        ]

    def test_stable_when_attempts_are_identical(self):
        a = self._events()
        b = copy.deepcopy(a)
        report = V.validate_replay([a, b])
        self.assertTrue(report["stable"])
        self.assertTrue(report["deterministic"])
        self.assertEqual(report["hash"], C.hash_json(a))

    def test_unstable_when_attempts_differ(self):
        a = self._events()
        b = copy.deepcopy(a)
        b[1]["target"] = "t/DIFFERENT"
        report = V.validate_replay([a, b])
        self.assertFalse(report["stable"])
        self.assertTrue(report["deterministic"])

    def test_flat_sequence_conflict_detected(self):
        a = self._events()
        b = copy.deepcopy(a)
        b[0]["target"] = "t/conflict"
        report = V.validate_replay(a + b)
        self.assertFalse(report["stable"])

    def test_non_list_input(self):
        report = V.validate_replay("not-a-list")
        self.assertFalse(report["stable"])


if __name__ == "__main__":
    unittest.main()
