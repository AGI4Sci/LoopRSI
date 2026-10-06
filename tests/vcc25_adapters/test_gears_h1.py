import argparse
import json
import tempfile
import unittest
from pathlib import Path

from adapters.vcc25 import get_adapter
from adapters.vcc25.gears_h1 import _prediction_perturbation, preflight
from domain_knowledge import KnowledgeQuery, KnowledgeStore


ROOT = Path(__file__).resolve().parents[2]


class GearsH1Tests(unittest.TestCase):
    def test_request_targets_real_gears_runner(self):
        store = KnowledgeStore.from_directory(ROOT / "knowledge" / "vcc25")
        model = next(card for card in store.query(KnowledgeQuery(
            task_id="vcc25", asset_types=("model",), max_results=20,
        )) if card.id == "kb:model:gears")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {role: str(root / role) for role in (
                "source_repository", "processed_dataset", "split_json",
                "gene2go", "gene_names",
            )}
            adapter = get_adapter("vcc25.gears")
            request = adapter.prepare("train", model, {
                "working_directory": str(root),
                "output_location": str(root / "output"),
                "inputs": inputs, "resources": {"gpu_count": 1, "max_minutes": 60},
            })
            self.assertIn("gears_h1.py", request.command[1])
            self.assertNotIn("reproduction_entry", " ".join(request.command))
            self.assertEqual(request.expected_artifacts[1], str((root / "output" / "validation_predictions.npz").resolve()))
            with self.assertRaisesRegex(ValueError, "action"):
                adapter.prepare("predict", model, {
                    "working_directory": str(root),
                    "output_location": str(root / "output"),
                    "inputs": inputs, "resources": {"gpu_count": 1, "max_minutes": 60},
                })

    def test_preflight_reports_missing_assets_without_expression_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            split.write_text(json.dumps({"train": ["A"], "val": ["B"]}))
            args = argparse.Namespace(
                source_repository=str(root / "source"),
                processed_dataset=str(root / "dataset"),
                split_json=str(split), gene2go=str(root / "gene2go.pkl"),
                gene_names=str(root / "genes.txt"), output=str(root / "output"),
            )
            result = preflight(args)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("dataset", result["missing_assets"])
            self.assertFalse(result["test_expression_read"])
            self.assertFalse(result["gpu_used"])
            split.write_text(json.dumps({"train": ["A"], "val": ["A"]}))
            with self.assertRaisesRegex(ValueError, "overlap"):
                preflight(args)

    def test_final_expression_path_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                source_repository=str(root / "source"),
                processed_dataset=str(root / "test_expression"),
                split_json=str(root / "split.json"),
                gene2go=str(root / "gene2go.pkl"),
                gene_names=str(root / "genes.txt"), output=str(root / "output"),
            )
            with self.assertRaisesRegex(ValueError, "final-test"):
                preflight(args)

    def test_gears_condition_conversion(self):
        self.assertEqual(_prediction_perturbation("CBL+ctrl"), ["CBL"])
        self.assertEqual(_prediction_perturbation("CBL+CNN1"), ["CBL", "CNN1"])
        self.assertEqual(_prediction_perturbation("ctrl"), [])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _prediction_perturbation("A+B+C")


if __name__ == "__main__":
    unittest.main()
