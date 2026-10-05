import argparse
import json
import tempfile
import unittest
from pathlib import Path

from adapters.vcc25 import get_adapter
from adapters.vcc25.scgenept_h1 import preflight
from domain_knowledge import KnowledgeQuery, KnowledgeStore


ROOT = Path(__file__).resolve().parents[2]


class ScGenePTH1Tests(unittest.TestCase):
    def test_adapter_targets_h1_runner(self):
        store = KnowledgeStore.from_directory(ROOT / "knowledge" / "vcc25")
        model = next(card for card in store.query(KnowledgeQuery(
            task_id="vcc25", asset_types=("model",), max_results=20,
        )) if card.id == "kb:model:scgenept-go-all")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = (
                "source_repository", "processed_dataset", "split_json", "gene_names",
                "checkpoint", "vocab", "go_all_embedding",
            )
            request = get_adapter("vcc25.scgenept").prepare("train", model, {
                "working_directory": str(root),
                "output_location": str(root / "output"),
                "inputs": {role: str(root / role) for role in roles},
                "resources": {"gpu_count": 1, "max_minutes": 60},
            })
            self.assertIn("scgenept_h1.py", request.command[1])
            self.assertNotIn("reproduction_entry", " ".join(request.command))
            self.assertEqual(len(request.expected_artifacts), 2)

    def test_preflight_requires_fixed_disjoint_split_and_rejects_test_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            genes = root / "genes.txt"
            split.write_text(json.dumps({
                "train": [f"T{i}" for i in range(150)],
                "val": [f"V{i}" for i in range(50)],
            }))
            genes.write_text("\n".join(f"G{i}" for i in range(18080)) + "\n")
            args = argparse.Namespace(
                source_repository=str(root / "source"),
                processed_dataset=str(root / "dataset"),
                split_json=str(split),
                gene_names=str(genes),
                checkpoint=str(root / "checkpoint"),
                vocab=str(root / "vocab"),
                go_all_embedding=str(root / "go_all"),
                output=str(root / "output"),
            )
            result = preflight(args)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("dataset", result["missing_assets"])
            self.assertFalse(result["test_expression_read"])
            split.write_text(json.dumps({
                "train": [f"T{i}" for i in range(150)],
                "val": ["T0", *(f"V{i}" for i in range(49))],
            }))
            with self.assertRaisesRegex(ValueError, "overlap"):
                preflight(args)
            args.processed_dataset = str(root / "final_test_expression")
            with self.assertRaisesRegex(ValueError, "final-test"):
                preflight(args)


if __name__ == "__main__":
    unittest.main()
