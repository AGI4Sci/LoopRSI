import tempfile
import unittest
from pathlib import Path

from adapters.vcc25.probe_primeflow_environment import build_report


class PrimeFlowProbeTests(unittest.TestCase):
    def test_report_checks_only_train_validation_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.h5ad"
            validation = root / "validation.h5ad"
            train.touch()
            validation.touch()
            report = build_report(train, validation)

        self.assertTrue(report["assets_visible"]["train"])
        self.assertTrue(report["assets_visible"]["validation"])
        self.assertFalse(report["final_test_expression_read"])
        self.assertIn("python_version", report)
        self.assertIn("torch_version", report)
        self.assertEqual(
            set(report["runtime_modules"]),
            {"torch", "anndata", "perturbench", "primeflow"},
        )


if __name__ == "__main__":
    unittest.main()
