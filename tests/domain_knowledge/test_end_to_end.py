import contextlib
import io
import json
import unittest
from pathlib import Path

from domain_knowledge.__main__ import main


ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT = ROOT / "knowledge" / "vcc25"


def run_cli(arguments):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(["--root", str(KNOWLEDGE_ROOT), *arguments])
    payload = json.loads(stdout.getvalue()) if stdout.getvalue() else None
    return code, payload, stderr.getvalue()


class KnowledgeCliEndToEndTests(unittest.TestCase):
    def test_validate_reports_fixed_contract_separately(self):
        code, payload, error = run_cli(["validate"])
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["card_count"], 27)
        self.assertEqual(payload["evaluation_contract"]["id"], "vcc25-eval-v1")
        self.assertNotIn("benchmark", payload)

    def test_l1_and_l5_queries_return_stable_provisional_ids_and_hash(self):
        l1_args = ["query", "--layer", "L1", "--token-budget", "900"]
        code, l1, error = run_cli(l1_args)
        self.assertEqual((code, error), (0, ""))
        self.assertTrue(l1["card_ids"])
        self.assertTrue(all(card_id.startswith("kb:paper:") for card_id in l1["card_ids"]))
        self.assertIn("provisional_initial_release", l1["content"])
        self.assertEqual(run_cli(l1_args)[1], l1)

        code, l5, error = run_cli(
            [
                "query", "--layer", "L5", "--asset-type", "model",
                "--readiness", "adapter_required", "finetune_required",
                "--token-budget", "900",
            ]
        )
        self.assertEqual((code, error), (0, ""))
        self.assertIn("kb:model:lingshu-vcc-85m", l5["card_ids"])
        self.assertEqual(len(l5["content_hash"]), 64)
        self.assertNotIn("benchmark", l5)

    def test_adapter_check_reports_assets_without_execution(self):
        code, payload, error = run_cli(
            ["check-adapter", "--model-id", "kb:model:lingshu-vcc-85m"]
        )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(payload["adapter_id"], "vcc25.lingshu")
        self.assertEqual(
            payload["missing"],
            ["checkpoint", "gene_embeddings", "reference_dataset"],
        )
        self.assertFalse(payload["evidence"]["execution_performed"])

    def test_restricted_query_exits_nonzero(self):
        code, payload, error = run_cli(
            ["query", "--layer", "L5", "--text", "/private/competition_test/data"]
        )
        self.assertNotEqual(code, 0)
        self.assertIsNone(payload)
        self.assertIn("competition_test", error)


if __name__ == "__main__":
    unittest.main()
