import unittest
import rsi_step0.contracts as C


class TestContracts(unittest.TestCase):
    def test_enums(self):
        self.assertIn("draft", C.OPERATORS)
        self.assertIn("schedule", C.OPERATORS)
        self.assertIn("execution", C.MODES)
        self.assertIn("positive", C.LABELS)
        self.assertEqual(C.W2_TRAIN_OPERATORS, ("draft", "improve", "crossover"))

    def test_validators(self):
        self.assertTrue(C.is_valid_operator("draft"))
        self.assertFalse(C.is_valid_operator("nope"))
        self.assertTrue(C.is_valid_mode("execution"))
        self.assertTrue(C.is_valid_label("positive"))
        self.assertTrue(C.is_valid_status("ok"))
        self.assertTrue(C.is_valid_schedule_decision("stop"))
        self.assertFalse(C.is_valid_schedule_decision("frobnitz"))

    def test_dedupe_key(self):
        k1 = C.make_dedupe_key("r1", 1, "p1", "v1")
        k2 = C.make_dedupe_key("r1", 1, "p1", "v1")
        k3 = C.make_dedupe_key("r1", 2, "p1", "v1")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)

    def test_hash_json_stable(self):
        self.assertEqual(C.hash_json({"b": 1, "a": 2}), C.hash_json({"a": 2, "b": 1}))
        self.assertNotEqual(C.hash_json({"a": 1}), C.hash_json({"a": 2}))


if __name__ == "__main__":
    unittest.main()
