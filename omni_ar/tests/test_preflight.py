from __future__ import annotations

import unittest

from omni_ar.preflight import gpu_budget_allowed


class PreflightBudgetTests(unittest.TestCase):
    def test_cpu_only_task_needs_no_gpu(self) -> None:
        self.assertTrue(gpu_budget_allowed(0, 0))

    def test_gpu_task_requires_enough_authorized_gpus(self) -> None:
        self.assertFalse(gpu_budget_allowed(1, 0))
        self.assertFalse(gpu_budget_allowed(2, 1))
        self.assertTrue(gpu_budget_allowed(1, 1))
        self.assertTrue(gpu_budget_allowed(2, 2))


if __name__ == "__main__":
    unittest.main()
