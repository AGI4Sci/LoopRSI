import unittest
import rsi_step0.reward as R


class TestReward(unittest.TestCase):
    def test_normalize_outcome(self):
        self.assertAlmostEqual(R.normalize_outcome(0.30614), 0.0)
        self.assertAlmostEqual(R.normalize_outcome(1.0), 1.0)
        self.assertAlmostEqual(R.normalize_outcome(0.421, 0.30614), (0.421 - 0.30614) / (1.0 - 0.30614))
        self.assertEqual(R.normalize_outcome(0.1), 0.0)

    def test_compute_cost(self):
        cost, time_ = R.compute_cost({"runtime_seconds": 120.0, "gpu_count": 2})
        self.assertEqual(cost, 240.0)
        self.assertEqual(time_, 120.0)
        # missing gpu defaults to 1
        cost2, _ = R.compute_cost({"runtime_seconds": 50.0})
        self.assertEqual(cost2, 50.0)
        # missing fields -> 0
        c, t = R.compute_cost({})
        self.assertEqual(c, 0.0)
        self.assertEqual(t, 0.0)

    def test_cost_aware_reward_formula(self):
        r = R.cost_aware_reward(0.421, {"runtime_seconds": 120.0, "gpu_count": 1},
                                baseline=0.30614, lambda_c=1e-6, lambda_t=1e-6)
        norm = (0.421 - 0.30614) / (1.0 - 0.30614)
        self.assertAlmostEqual(r["normalized_outcome"], norm)
        self.assertAlmostEqual(r["cost_compute"], 120.0)
        self.assertAlmostEqual(r["time_cost"], 120.0)
        self.assertAlmostEqual(r["total"], norm - 1e-6 * 120.0 - 1e-6 * 120.0)

    def test_operator_shaping(self):
        self.assertAlmostEqual(R.operator_shaping(0.5, "draft"), 0.5)
        # improve above parent gets +0.5*delta
        self.assertAlmostEqual(R.operator_shaping(0.6, "improve", [0.5]), 0.6 + 0.5 * 0.1)
        # no parent -> parent base treated as 0, bonus 0.5*base
        self.assertAlmostEqual(R.operator_shaping(0.6, "improve", None), 0.6 + 0.5 * 0.6)
        # crossover with two parents uses max parent
        self.assertAlmostEqual(R.operator_shaping(0.7, "crossover", [0.5, 0.6]), 0.7 + 0.5 * 0.1)
        # base == 0 -> 0
        self.assertEqual(R.operator_shaping(0.0, "improve", [0.5]), 0.0)

    def test_group_advantage_boundaries(self):
        self.assertEqual(R.group_advantage([1.0]), [0.0])
        self.assertEqual(R.group_advantage([0.5, 0.5, 0.5]), [0.0, 0.0, 0.0])
        adv = R.group_advantage([0.1, 0.5, 0.9])
        self.assertEqual(len(adv), 3)
        # higher reward => higher advantage
        self.assertGreater(adv[2], adv[0])

    def test_classify_outcome(self):
        self.assertEqual(R.classify_outcome("ok", 0.5), "positive")
        self.assertEqual(R.classify_outcome("ok", 0.0), "excluded")
        self.assertEqual(R.classify_outcome("failed", 0.0), "negative")
        self.assertEqual(R.classify_outcome("failed", 0.0, environmental_failure=True), "excluded")
        self.assertEqual(R.classify_outcome("ok", 0.5, within_budget=False), "negative")
        self.assertEqual(R.classify_outcome("timeout", 0.0), "excluded")


if __name__ == "__main__":
    unittest.main()
