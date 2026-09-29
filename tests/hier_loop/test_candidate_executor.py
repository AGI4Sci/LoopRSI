import json
import sys
import tempfile
import unittest
from pathlib import Path

from hier_loop.candidate_executor import CandidateBoundaryError, CandidateExecutor


def request(**overrides):
    value = {
        "allowed_paths": ["crpm", "scripts", "tests"],
        "hypothesis": "a train-only target prior improves validation PCC",
        "activation_diagnostics": ["prior coverage", "prediction delta norm"],
        "train_command": [
            "python3",
            "tasks/vcc25/implementation/official_h1_autonomous_research_loop.py",
            "--output-dir",
            "{smoke_output_dir}",
        ],
        "validation_command": [
            "python3",
            "tasks/vcc25/implementation/official_h1_autonomous_research_loop.py",
            "--output-dir",
            "{validation_output_dir}",
        ],
        "expected_artifacts": ["lineage_summary.json"],
    }
    value.update(overrides)
    return value


class CandidateExecutorTests(unittest.TestCase):
    def make_workspace(self, root):
        workspace = Path(root) / "experiment"
        implementation = workspace / "tasks/vcc25/implementation"
        for name in ("crpm", "scripts", "tests", "reports"):
            (implementation / name).mkdir(parents=True, exist_ok=True)
        return workspace

    def test_candidate_paths_remain_below_resolved_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(temporary)
            prepared = CandidateExecutor().prepare(request(), workspace)
            resolved_workspace = workspace.resolve()
            self.assertTrue(prepared.allowed_paths)
            self.assertTrue(
                all(path == resolved_workspace or resolved_workspace in path.parents for path in prepared.allowed_paths)
            )
            self.assertTrue(prepared.request_path.is_file())

    def test_parent_traversal_and_symlink_escape_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(temporary)
            with self.assertRaises(CandidateBoundaryError):
                CandidateExecutor().prepare(request(allowed_paths=["../outside"]), workspace)

            outside = Path(temporary) / "outside"
            outside.mkdir()
            (workspace / "tasks/vcc25/implementation/scripts/link").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(CandidateBoundaryError):
                CandidateExecutor().prepare(request(allowed_paths=["scripts/link"]), workspace)

    def test_only_declared_implementation_paths_are_writable(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(temporary)
            for forbidden in ("reports", "tasks/vcc25/official_h1_adapter.py", ".git"):
                with self.subTest(forbidden=forbidden):
                    with self.assertRaises(CandidateBoundaryError):
                        CandidateExecutor().prepare(request(allowed_paths=[forbidden]), workspace)

    def test_final_expression_and_canonical_run_trial_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(temporary)
            unsafe_commands = (
                ["python3", "candidate.py", "--reference-h5ad", "/data/test/adata_Test.h5ad"],
                ["python3", "-m", "tasks.vcc25.official_h1_adapter", "run_trial"],
                ["python3", "candidate.py", "--real-de", "/data/real_de.csv"],
            )
            for command in unsafe_commands:
                with self.subTest(command=command):
                    with self.assertRaises(CandidateBoundaryError):
                        CandidateExecutor().prepare(request(validation_command=command), workspace)

    def test_validation_only_command_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(temporary)
            prepared = CandidateExecutor().prepare(request(), workspace)
            self.assertTrue(
                any(token.endswith("official_h1_autonomous_research_loop.py") for token in prepared.validation_command)
            )
            implementation_request = json.loads(
                prepared.implementation_request_path.read_text(encoding="utf-8")
            )
            parameters = implementation_request["trial_proposal"]["parameters"]
            self.assertEqual(parameters, {"candidate_variant": "autonomous_research_candidate"})
            self.assertNotIn("evaluation_authority", parameters)

    def test_preflight_checks_tools_and_assets_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = self.make_workspace(temporary)
            asset = Path(temporary) / "weights.bin"
            asset.write_bytes(b"asset")
            prepared = CandidateExecutor().prepare(request(), workspace)
            result = CandidateExecutor().run_preflight(
                prepared,
                required_tools=(sys.executable,),
                required_assets=(asset,),
            )
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["codex_invoked"])
            self.assertFalse(result["gpu_invoked"])
            saved = json.loads((workspace / "artifacts/candidate_preflight.json").read_text())
            self.assertEqual(saved["status"], "ok")


if __name__ == "__main__":
    unittest.main()
