import json
import tempfile
import unittest
from pathlib import Path

from domain_knowledge import (
    KnowledgeQuery,
    KnowledgeStore,
    KnowledgeValidationError,
)


def _paper(card_id="kb:paper:alpha", **updates):
    value = {
        "id": card_id,
        "schema_version": "1.0",
        "asset_type": "paper",
        "title": "Alpha perturbation paper",
        "summary_plain": "A genetic perturbation method.",
        "tags": ["perturbation", "baseline"],
        "layers": ["L1", "L2", "L3"],
        "sources": [{"url": "https://example.org/paper", "kind": "paper"}],
        "relations": [],
        "review_status": "approved",
        "review_basis": "provisional_initial_release",
        "mechanism": "Predict the response to a gene perturbation.",
        "limitations": ["Needs matched controls."],
    }
    value.update(updates)
    return value


def _repository(card_id="kb:repo:alpha", **updates):
    value = {
        "id": card_id,
        "schema_version": "1.0",
        "asset_type": "repository",
        "title": "Alpha repository",
        "summary_plain": "Reference implementation.",
        "tags": ["perturbation", "python"],
        "layers": ["L5"],
        "sources": [{"url": "https://github.com/example/alpha", "kind": "repository"}],
        "relations": ["kb:paper:alpha"],
        "review_status": "approved",
        "review_basis": "provisional_initial_release",
        "official_url": "https://github.com/example/alpha",
        "revision": "main",
        "licenses": {"code": "Apache-2.0"},
        "entrypoints": ["train", "predict"],
    }
    value.update(updates)
    return value


def _model(card_id="kb:model:alpha", **updates):
    value = {
        "id": card_id,
        "schema_version": "1.0",
        "asset_type": "model",
        "title": "Alpha model",
        "summary_plain": "A trainable perturbation model.",
        "tags": ["perturbation", "fast"],
        "layers": ["L4", "L5"],
        "sources": [{"url": "https://example.org/model", "kind": "model"}],
        "relations": ["kb:paper:alpha", "kb:repo:alpha"],
        "review_status": "approved",
        "review_basis": "provisional_initial_release",
        "usage_mode": "trainable_recipe",
        "execution_readiness": "train_required",
        "artifacts": [
            {
                "role": "config",
                "uri": "https://example.org/config.json",
                "format": "json",
                "access": "public",
                "integrity": {"status": "not_published"},
            }
        ],
        "licenses": {"code": "Apache-2.0", "weights": "not_applicable"},
        "io_contract": {"input": "h5ad", "output": "expression"},
        "resource_profile": {"gpu_count": 1, "max_minutes": 30},
        "vcc25_compatibility": {"level": "adapter_needed", "gaps": ["gene order"]},
        "execution_recipe": {"adapter_id": "vcc25.alpha", "actions": ["train", "predict"]},
        "smoke_evidence": [],
    }
    value.update(updates)
    return value


def _contract(**updates):
    value = {
        "id": "vcc25-eval-v1",
        "schema_version": "1.0",
        "task_id": "vcc25",
        "gene_count": 18080,
        "gene_order_source": "official-training-contract",
        "metrics": ["PCC"],
        "selection_policy": "training_or_proxy_only",
        "leakage_policy": "final_test_isolated",
    }
    value.update(updates)
    return value


class KnowledgeTree:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for name in ("papers", "repositories", "models"):
            (self.root / name).mkdir()
        self.write("papers", "alpha.json", _paper())
        self.write("repositories", "alpha.json", _repository())
        self.write("models", "alpha.json", _model())
        self.write_root("evaluation_contract.json", _contract())

    def write(self, directory, name, value):
        (self.root / directory / name).write_text(json.dumps(value), encoding="utf-8")

    def write_root(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def close(self):
        self._tmp.cleanup()


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self):
        self.tree = KnowledgeTree()

    def tearDown(self):
        self.tree.close()

    def test_loads_three_searchable_types_but_not_evaluation_contract(self):
        store = KnowledgeStore.from_directory(self.tree.root)
        cards = store.query(KnowledgeQuery(task_id="vcc25", max_results=10))
        self.assertEqual({card.asset_type for card in cards}, {"paper", "repository", "model"})
        self.assertNotIn("vcc25-eval-v1", {card.id for card in cards})
        self.assertEqual(store.evaluation_contract.id, "vcc25-eval-v1")
        self.assertEqual(store.validate(), ())

    def test_approved_provisional_card_is_searchable(self):
        result = KnowledgeStore.from_directory(self.tree.root).query(
            KnowledgeQuery(task_id="vcc25", text="baseline", approved_only=True)
        )
        self.assertEqual([card.id for card in result], ["kb:paper:alpha"])
        self.assertEqual(result[0].review_basis, "provisional_initial_release")

    def test_filters_type_layer_and_readiness_and_uses_stable_id_tie_break(self):
        self.tree.write("models", "beta.json", _model("kb:model:beta"))
        store = KnowledgeStore.from_directory(self.tree.root)
        result = store.query(
            KnowledgeQuery(
                task_id="vcc25",
                layer="L5",
                text="perturbation",
                asset_types=("model",),
                readiness=("train_required",),
                max_results=10,
            )
        )
        self.assertEqual([card.id for card in result], ["kb:model:alpha", "kb:model:beta"])

    def test_unknown_layer_is_rejected(self):
        store = KnowledgeStore.from_directory(self.tree.root)
        with self.assertRaisesRegex(KnowledgeValidationError, "layer"):
            store.query(KnowledgeQuery(task_id="vcc25", layer="L6"))

    def test_duplicate_id_reports_both_files(self):
        self.tree.write("papers", "duplicate.json", _paper())
        with self.assertRaisesRegex(KnowledgeValidationError, r"duplicate.*alpha\.json.*duplicate\.json"):
            KnowledgeStore.from_directory(self.tree.root)

    def test_missing_shared_field_has_file_and_field_context(self):
        bad = _paper()
        del bad["summary_plain"]
        self.tree.write("papers", "alpha.json", bad)
        with self.assertRaisesRegex(KnowledgeValidationError, r"papers/alpha\.json.*summary_plain"):
            KnowledgeStore.from_directory(self.tree.root)

    def test_missing_model_field_is_rejected(self):
        bad = _model()
        del bad["execution_recipe"]
        self.tree.write("models", "alpha.json", bad)
        with self.assertRaisesRegex(KnowledgeValidationError, r"models/alpha\.json.*execution_recipe"):
            KnowledgeStore.from_directory(self.tree.root)

    def test_malformed_artifacts_are_rejected(self):
        self.tree.write("models", "alpha.json", _model(artifacts={"role": "weights"}))
        with self.assertRaisesRegex(KnowledgeValidationError, r"models/alpha\.json.*artifacts"):
            KnowledgeStore.from_directory(self.tree.root)

    def test_broken_relation_is_rejected(self):
        self.tree.write("models", "alpha.json", _model(relations=["kb:paper:missing"]))
        with self.assertRaisesRegex(KnowledgeValidationError, r"models/alpha\.json.*kb:paper:missing"):
            KnowledgeStore.from_directory(self.tree.root)

    def test_contract_rejects_mismatched_result_ids(self):
        contract = KnowledgeStore.from_directory(self.tree.root).evaluation_contract
        contract.assert_comparable(["vcc25-eval-v1", "vcc25-eval-v1"])
        with self.assertRaisesRegex(KnowledgeValidationError, "not comparable"):
            contract.assert_comparable(["vcc25-eval-v1", "another-contract"])

    def test_restricted_identifiers_are_rejected_recursively(self):
        self.tree.write(
            "models",
            "alpha.json",
            _model(io_contract={"nested": [{"payload": "/safe/real_de/answers.h5ad"}]}),
        )
        with self.assertRaisesRegex(KnowledgeValidationError, r"models/alpha\.json.*real_de"):
            KnowledgeStore.from_directory(self.tree.root)

    def test_ready_requires_smoke_evidence(self):
        self.tree.write("models", "alpha.json", _model(execution_readiness="ready"))
        with self.assertRaisesRegex(KnowledgeValidationError, r"models/alpha\.json.*smoke_evidence"):
            KnowledgeStore.from_directory(self.tree.root)

    def test_unknown_card_type_is_rejected(self):
        self.tree.write("papers", "alpha.json", _paper(asset_type="benchmark"))
        with self.assertRaisesRegex(KnowledgeValidationError, r"papers/alpha\.json.*asset_type"):
            KnowledgeStore.from_directory(self.tree.root)


if __name__ == "__main__":
    unittest.main()
