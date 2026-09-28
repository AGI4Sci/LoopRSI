"""Unit tests for rsi_step0.router (make_action, GuidedRouter, TrainedRouter)."""

import os
import tempfile
import unittest

import rsi_step0.contracts as C
from rsi_step0.router import GuidedRouter, TrainedRouter, make_action


class MakeActionTest(unittest.TestCase):
    def test_defaults_and_legality(self):
        action = make_action(
            run_id="run1",
            round_no=1,
            step=2,
            operator="draft",
            mode="execution",
            target="task-42",
            decision={"hypothesis": "h"},
        )
        self.assertEqual(action["run_id"], "run1")
        self.assertEqual(action["round"], 1)
        self.assertEqual(action["step"], 2)
        self.assertEqual(action["operator"], "draft")
        self.assertEqual(action["mode"], "execution")
        self.assertEqual(action["target"], "task-42")
        self.assertEqual(action["decision"], {"hypothesis": "h"})
        self.assertIsInstance(action["action_id"], str)
        self.assertTrue(len(action["action_id"]) > 0)
        self.assertEqual(
            action["budget"],
            {"max_seconds": 3600, "max_gpu_seconds": 3600, "max_cost": 1.0},
        )
        self.assertEqual(
            action["provenance"],
            {"parent_action_id": None, "model": "deterministic", "seed": 0},
        )
        self.assertTrue(C.is_valid_operator(action["operator"]))
        self.assertTrue(C.is_valid_mode(action["mode"]))
        for key in (
            "action_id",
            "run_id",
            "round",
            "step",
            "operator",
            "mode",
            "target",
            "decision",
            "budget",
            "provenance",
        ):
            self.assertIn(key, action)

    def test_invalid_operator_raises(self):
        with self.assertRaises(ValueError):
            make_action("r", 1, 2, "nope", "execution", "t", {})


class GuidedRouterTest(unittest.TestCase):
    def _context(self):
        return {
            "run_id": "run1",
            "round": 3,
            "task_id": "task-77",
            "history": [
                {"action_id": "a1", "target": "task-77"},
                {"action_id": "a2", "target": "task-77"},
            ],
            "last_outcome": None,
            "available_actions": [{"action_id": "a1"}, {"action_id": "a2"}],
            "resource_state": {},
        }

    def test_decide_only_w2_operators_and_payload(self):
        router = GuidedRouter(seed=0)
        required = {
            "draft": {
                "hypothesis",
                "change_scope",
                "expected_effect",
                "acceptance_criteria",
            },
            "improve": {"target_action_id", "diagnosis", "patch"},
            "crossover": {"parent_action_ids", "recombination_strategy"},
        }
        seen = set()
        for _ in range(500):
            action = router.decide(self._context())
            self.assertEqual(action["mode"], "execution")
            self.assertEqual(action["target"], "task-77")
            self.assertTrue(C.is_valid_operator(action["operator"]))
            self.assertIn(action["operator"], C.W2_TRAIN_OPERATORS)
            seen.add(action["operator"])
            self.assertTrue(
                required[action["operator"]].issubset(set(action["decision"].keys())),
                f"missing keys for {action['operator']}: "
                f"{required[action['operator']]} vs {set(action['decision'].keys())}",
            )
        self.assertEqual(seen, set(C.W2_TRAIN_OPERATORS))

    def test_save_load_roundtrip(self):
        router = GuidedRouter(
            seed=7,
            operator_probs={"draft": 0.5, "improve": 0.25, "crossover": 0.25},
        )
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            router.save(tmp)
            loaded = GuidedRouter()
            loaded.load(tmp)
            self.assertEqual(loaded.seed, 7)
            self.assertAlmostEqual(loaded.operator_probs["draft"], 0.5, places=6)
            self.assertAlmostEqual(loaded.operator_probs["improve"], 0.25, places=6)
            self.assertAlmostEqual(loaded.operator_probs["crossover"], 0.25, places=6)
        finally:
            os.unlink(tmp)

    def test_score(self):
        router = GuidedRouter(seed=0)
        candidates = [
            make_action("r", 1, 1, "draft", "execution", "t", {"x": 1}),
            make_action("r", 1, 2, "improve", "execution", "t", {"y": 2}),
        ]
        scores = router.score(self._context(), candidates)
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLess(score, 1.0)
        self.assertNotEqual(scores[0], scores[1])


class TrainedRouterTest(unittest.TestCase):
    def _context(self):
        return {
            "run_id": "run1",
            "round": 1,
            "task_id": "task-9",
            "history": [],
            "last_outcome": None,
            "available_actions": [],
            "resource_state": {},
        }

    def test_sampling_distribution(self):
        policy = {"draft": 3.0, "improve": 0.5, "crossover": 0.1}
        router = TrainedRouter(policy=policy, seed=123)
        counts = {op: 0 for op in policy}
        for _ in range(3000):
            counts[router.decide(self._context())["operator"]] += 1
        self.assertEqual(set(counts), set(policy))
        for op in policy:
            self.assertGreater(counts[op], 0)
        self.assertGreater(counts["draft"], counts["improve"])
        self.assertGreater(counts["improve"], counts["crossover"])

    def test_save_load_roundtrip(self):
        policy = {"draft": 2.0, "improve": 1.0}
        router = TrainedRouter(policy=policy, seed=5)
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            router.save(tmp)
            loaded = TrainedRouter()
            loaded.load(tmp)
            self.assertEqual(loaded.seed, 5)
            self.assertEqual(loaded.policy, policy)
            action = loaded.decide(self._context())
            self.assertIn(action["operator"], policy)
        finally:
            os.unlink(tmp)

    def test_score_returns_weights(self):
        router = TrainedRouter(policy={"draft": 2.0, "improve": 1.0})
        candidates = [
            make_action("r", 1, 1, "draft", "execution", "t", {"x": 1}),
            make_action("r", 1, 2, "improve", "execution", "t", {"y": 2}),
            make_action("r", 1, 3, "crossover", "execution", "t", {"z": 3}),
        ]
        self.assertEqual(router.score(self._context(), candidates), [2.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
