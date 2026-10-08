import unittest
from pathlib import Path

from adapters.vcc25.run_presage_h1_smoke import (
    build_scope,
    validate_source_path,
)


class PresageH1SmokeTests(unittest.TestCase):
    def test_only_official_training_expression_is_allowed(self):
        root = Path("/assets/official_2025")
        validate_source_path(root, root / "train" / "adata_Training.h5ad")

        for forbidden in (
            root / "validation" / "adata_Validation.h5ad",
            root / "test" / "adata_Test.h5ad",
            root / "final_test" / "expression.h5ad",
        ):
            with self.assertRaisesRegex(ValueError, "training expression"):
                validate_source_path(root, forbidden)

    def test_scope_is_one_train_and_one_internal_validation_batch(self):
        scope = build_scope(
            train_cells=48,
            validation_cells=16,
            genes=18080,
            conditions=("ACAT2", "ACVR1B", "AKT2", "ARID1A"),
        )

        self.assertEqual(scope["train_batches"], 1)
        self.assertEqual(scope["validation_batches"], 1)
        self.assertEqual(scope["genes"], 18080)
        self.assertEqual(
            scope["validation_source"],
            "deterministic_internal_split_of_official_training",
        )
        self.assertFalse(scope["official_validation_expression_read"])
        self.assertFalse(scope["final_test_expression_read"])


if __name__ == "__main__":
    unittest.main()
