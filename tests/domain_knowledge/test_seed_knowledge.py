import json
import unittest
from pathlib import Path

from domain_knowledge import KnowledgeQuery, KnowledgeStore


ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = ROOT / "knowledge" / "vcc25"


class SeedKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.store = KnowledgeStore.from_directory(KNOWLEDGE_ROOT)
        self.cards = self.store.query(KnowledgeQuery(task_id="vcc25", max_results=100))
        self.by_id = {card.id: card for card in self.cards}

    def test_seed_has_exact_expected_inventory_and_unique_ids(self):
        counts = {
            kind: sum(card.asset_type == kind for card in self.cards)
            for kind in ("paper", "repository", "model")
        }
        self.assertEqual(counts, {"paper": 9, "repository": 9, "model": 9})
        self.assertEqual(len(self.by_id), 27)
        self.assertFalse(any(card.asset_type == "benchmark" for card in self.cards))
        self.assertNotIn(self.store.evaluation_contract.id, self.by_id)

    def test_every_card_is_provisionally_approved_and_has_https_provenance(self):
        for card in self.cards:
            with self.subTest(card=card.id):
                self.assertEqual(card.review_status, "approved")
                self.assertEqual(card.review_basis, "provisional_initial_release")
                self.assertTrue(
                    any(source["url"].startswith("https://") for source in card.sources),
                    f"{card.id} lacks an authoritative HTTPS source",
                )
                self.assertTrue(card.relations)

    def test_expected_papers_repositories_and_models_exist(self):
        paper_slugs = {
            "lingshu-cell", "state", "gears", "linear-baseline", "perturbench", "scgenept",
            "primeflow", "presage", "sclambda"
        }
        self.assertEqual(
            {card.id.removeprefix("kb:paper:") for card in self.cards if card.asset_type == "paper"},
            paper_slugs,
        )
        self.assertEqual(
            {card.id.removeprefix("kb:repo:") for card in self.cards if card.asset_type == "repository"},
            paper_slugs,
        )
        self.assertEqual(
            {card.id for card in self.cards if card.asset_type == "model"},
            {
                "kb:model:lingshu-vcc-85m",
                "kb:model:state-st-hvg-replogle",
                "kb:model:perturbench-latent-additive",
                "kb:model:scgenept-go-all",
                "kb:model:gears",
                "kb:model:linear-pseudobulk",
                "kb:model:primeflow",
                "kb:model:presage",
                "kb:model:sclambda",
            },
        )

    def test_model_readiness_and_execution_metadata_are_explicit(self):
        expected = {
            "kb:model:lingshu-vcc-85m": "adapter_required",
            "kb:model:state-st-hvg-replogle": "finetune_required",
            "kb:model:perturbench-latent-additive": "train_required",
            "kb:model:scgenept-go-all": "finetune_required",
            "kb:model:gears": "train_required",
            "kb:model:linear-pseudobulk": "train_required",
            "kb:model:primeflow": "train_required",
            "kb:model:presage": "reference_only",
            "kb:model:sclambda": "reference_only",
        }
        for card_id, readiness in expected.items():
            card = self.by_id[card_id]
            with self.subTest(card=card_id):
                self.assertEqual(card.execution_readiness, readiness)
                self.assertIsInstance(card.get("artifacts"), tuple)
                self.assertTrue(card.get("licenses").get("code"))
                self.assertTrue(card.get("licenses").get("weights"))
                self.assertTrue(card.get("io_contract").get("input_format"))
                self.assertIn("adapter_id", card.get("execution_recipe"))
                for artifact in card.get("artifacts"):
                    self.assertIn(artifact["integrity"]["status"], {"not_published", "not_applicable"})

    def test_schema_documents_are_valid_json_and_cover_all_contracts(self):
        schema_dir = KNOWLEDGE_ROOT / "schemas"
        expected = {
            "card.schema.json",
            "paper.schema.json",
            "repository.schema.json",
            "model.schema.json",
            "evaluation-contract.schema.json",
        }
        self.assertEqual({path.name for path in schema_dir.glob("*.json")}, expected)
        for path in schema_dir.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_no_weight_binary_is_committed_inside_knowledge_tree(self):
        forbidden = {".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".h5ad", ".npz"}
        offenders = [path for path in KNOWLEDGE_ROOT.rglob("*") if path.suffix.lower() in forbidden]
        self.assertEqual(offenders, [])

    def test_local_baseline_claim_is_not_a_sortable_score(self):
        lingshu = self.by_id["kb:model:lingshu-vcc-85m"]
        self.assertNotIn("benchmark_score", lingshu.data)
        notes = lingshu.get("evidence_notes")
        self.assertTrue(any("0.30614" in note and "unresolved" in note for note in notes))

    def test_executable_candidates_record_real_license_constraints(self):
        expected = {
            "kb:model:lingshu-vcc-85m": ("MIT", "model host card"),
            "kb:model:state-st-hvg-replogle": (
                "CC BY-NC-SA 4.0",
                "State Model Non-Commercial License",
            ),
            "kb:model:perturbench-latent-additive": ("BSD-3-Clause", "not_applicable"),
        }
        for card_id, (code_license, weight_license) in expected.items():
            licenses = self.by_id[card_id].get("licenses")
            with self.subTest(card=card_id):
                self.assertIn(code_license, licenses["code"])
                self.assertIn(weight_license, licenses["weights"])

    def test_second_edition_roles_are_explicit(self):
        self.assertEqual(
            self.by_id["kb:model:perturbench-latent-additive"].get("research_role"),
            "dependency_infrastructure",
        )
        for slug in ("primeflow", "presage", "sclambda"):
            with self.subTest(model=slug):
                self.assertEqual(self.by_id[f"kb:model:{slug}"].review_basis, "provisional_initial_release")
                self.assertEqual(self.by_id[f"kb:model:{slug}"].get("smoke_evidence"), ())


if __name__ == "__main__":
    unittest.main()
