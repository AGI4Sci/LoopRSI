"""Unit tests for rsi_step0.scheduler (advise, guard, deterministic fallback)."""

import unittest

import rsi_step0.contracts as C
from rsi_step0.scheduler import (
    LearnedScheduler,
    deterministic_advise,
    guard_budget,
)


def is_valid_schedule(action) -> bool:
    return (
        isinstance(action, dict)
        and C.is_valid_schedule_decision(action["decision"])
        and isinstance(action["reason"], str)
        and isinstance(action["budget_delta"], (int, float))
        and isinstance(action["next_task_id"], str)
    )


class DeterministicAdviseTest(unittest.TestCase):
    def test_stop_when_budget_exhausted(self):
        context = {"run_id": "r", "round": 1, "task_id": "t", "last_outcome": None}
        for budget in (
            {"max_seconds": 0, "max_cost": 1.0},
            {"max_seconds": 3600, "max_cost": 0},
        ):
            action = deterministic_advise(context, budget)
            self.assertEqual(action["decision"], "stop")
            self.assertTrue(is_valid_schedule(action))

    def test_converge_when_outcome_above_threshold(self):
        context = {
            "run_id": "r",
            "round": 1,
            "task_id": "t",
            "last_outcome": {"normalized_outcome": 0.8},
        }
        budget = {"max_seconds": 3600, "max_cost": 1.0}
        action = deterministic_advise(context, budget)
        self.assertEqual(action["decision"], "converge")
        self.assertTrue(is_valid_schedule(action))

    def test_continue_otherwise(self):
        context = {
            "run_id": "r",
            "round": 1,
            "task_id": "t",
            "last_outcome": {"normalized_outcome": 0.3},
        }
        budget = {"max_seconds": 3600, "max_cost": 1.0}
        self.assertEqual(deterministic_advise(context, budget)["decision"], "continue")

        no_outcome = {"run_id": "r", "round": 1, "task_id": "t", "last_outcome": None}
        self.assertEqual(deterministic_advise(no_outcome, budget)["decision"], "continue")


class GuardBudgetTest(unittest.TestCase):
    def test_forces_stop_when_exhausted(self):
        proposed = {
            "decision": "continue",
            "reason": "r",
            "budget_delta": 5.0,
            "next_task_id": "t",
        }
        action = guard_budget(proposed, {"max_seconds": 0, "max_cost": 1.0})
        self.assertEqual(action["decision"], "stop")
        self.assertTrue(is_valid_schedule(action))
        self.assertEqual(action["budget_delta"], 0.0)

    def test_invalid_decision_coerced_to_continue(self):
        proposed = {
            "decision": "nonsense",
            "reason": "r",
            "budget_delta": 1.0,
            "next_task_id": "t",
        }
        action = guard_budget(proposed, {"max_seconds": 3600, "max_cost": 1.0})
        self.assertEqual(action["decision"], "continue")
        self.assertTrue(is_valid_schedule(action))

    def test_budget_delta_clamped(self):
        proposed = {
            "decision": "continue",
            "reason": "r",
            "budget_delta": 99_999,
            "next_task_id": "t",
        }
        action = guard_budget(proposed, {"max_seconds": 3600, "max_cost": 1.0})
        self.assertLessEqual(action["budget_delta"], 3600.0)
        self.assertTrue(is_valid_schedule(action))


class LearnedSchedulerTest(unittest.TestCase):
    def _context(self):
        return {
            "run_id": "r",
            "round": 1,
            "task_id": "t1",
            "last_outcome": {"normalized_outcome": 0.5},
            "resource_state": {"max_seconds": 3600, "max_cost": 1.0},
        }

    def test_fallback_without_policy(self):
        scheduler = LearnedScheduler(policy=None, seed=0)
        action = scheduler.advise(self._context())
        self.assertEqual(action["decision"], "continue")
        self.assertTrue(is_valid_schedule(action))

    def test_fallback_exhausted(self):
        scheduler = LearnedScheduler(policy=None, seed=0)
        context = self._context()
        context["resource_state"] = {"max_seconds": 0, "max_cost": 1.0}
        action = scheduler.advise(context)
        self.assertEqual(action["decision"], "stop")
        self.assertTrue(is_valid_schedule(action))

    def test_policy_suggestion_guarded(self):
        scheduler = LearnedScheduler(
            policy={
                "decision": "continue",
                "reason": "learned",
                "budget_delta": 5.0,
                "next_task_id": "t1",
            },
            seed=0,
        )
        action = scheduler.advise(self._context())
        self.assertEqual(action["decision"], "continue")
        self.assertTrue(is_valid_schedule(action))

    def test_policy_stop_respected(self):
        scheduler = LearnedScheduler(
            policy={"decision": "stop", "reason": "done", "budget_delta": 1.0, "next_task_id": "t1"},
            seed=0,
        )
        action = scheduler.advise(self._context())
        self.assertEqual(action["decision"], "stop")
        self.assertTrue(is_valid_schedule(action))


if __name__ == "__main__":
    unittest.main()
