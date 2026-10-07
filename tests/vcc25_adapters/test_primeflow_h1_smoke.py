import unittest
from pathlib import Path

from adapters.vcc25.run_primeflow_h1_smoke import build_train_command, validate_source_paths


class PrimeFlowH1SmokeTests(unittest.TestCase):
    def test_only_official_train_and_validation_are_allowed(self):
        root = Path("/assets/official_2025")
        validate_source_paths(
            root,
            root / "train" / "adata_Training.h5ad",
            root / "validation" / "adata_Validation.h5ad",
        )
        with self.assertRaisesRegex(ValueError, "final-test"):
            validate_source_paths(
                root,
                root / "train" / "adata_Training.h5ad",
                root / "test" / "adata_Test.h5ad",
            )

    def test_training_command_is_single_gpu_and_bounded(self):
        command = build_train_command(Path("/tmp/smoke.h5ad"), Path("/tmp/split.csv"), Path("/tmp/genes.csv"), Path("/tmp/out"))
        self.assertEqual(command[:3], ["python3", "-m", "primeflow.modelcore.train"])
        self.assertIn("trainer.devices=1", command)
        self.assertIn("trainer.max_epochs=1", command)
        self.assertIn("+trainer.limit_train_batches=1", command)
        self.assertIn("+trainer.limit_val_batches=1", command)
        self.assertIn("test=False", command)
        self.assertIn("model.dynamics_model.gene_embedding_parquet_filepath=null", command)
        self.assertIn("model/lr_scheduler=cosine_annealing_warm_restarts", command)
        self.assertIn("~model.lr_scheduler.conf.warmup_epochs", command)
        self.assertIn("~model.lr_scheduler.conf.warmup_start_lr", command)
        self.assertIn("~model.lr_scheduler.conf.eta_min", command)
        self.assertFalse(any("adata_Test" in item or "final_test" in item for item in command))


if __name__ == "__main__":
    unittest.main()
