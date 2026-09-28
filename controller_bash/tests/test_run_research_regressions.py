from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "controller_bash/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_research import load_regression_suite  # noqa: E402
from task_contract import load_task_spec  # noqa: E402


class UnifiedResearchEntryRegressionTest(unittest.TestCase):
    def test_generic_entry_contains_no_task_or_method_names(self) -> None:
        source = (SCRIPTS / "run_research.py").read_text(encoding="utf-8").lower()
        forbidden = ("cifar", "vcc25", "bade", "afr", "crpm", "pseudobulk", "single_cell")
        for term in forbidden:
            self.assertNotIn(term, source)

    def test_both_suites_use_the_same_schema_and_disable_search(self) -> None:
        for task_dir in (ROOT / "tasks/cifar", ROOT / "tasks/vcc25"):
            spec = load_task_spec(task_dir / "task_spec.yaml")
            suite = load_regression_suite(
                task_dir / "regression_suite.yaml", spec["task"]["name"]
            )
            self.assertEqual(suite["schema_version"], "omni-ar-regression/v1")
            self.assertFalse(suite["policy"]["allow_search"])
            self.assertTrue(suite["policy"]["preserve_best"])

    def test_same_entry_runs_both_tasks_and_replays_history(self) -> None:
        summaries = []
        with TemporaryDirectory() as temp:
            for name in ("cifar", "vcc25"):
                output = Path(temp) / name
                proc = subprocess.run(
                    [
                        str(ROOT / "run_research"), "--task",
                        str(ROOT / f"tasks/{name}/task_spec.yaml"),
                        "--execution-mode", "skip", "--output-dir", str(output),
                    ],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
                summary = json.loads((output / "regression_summary.json").read_text())
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(summary["controller"]["num_trials"], 2)
                self.assertTrue(summary["compiled_case_coverage"]["passed"])
                self.assertTrue(summary["historical_replay"]["passed"])
                self.assertTrue(summary["generic_prompt_and_core_unchanged"])
                summaries.append(summary)
        self.assertEqual(
            summaries[0]["controller"]["entrypoint"], summaries[1]["controller"]["entrypoint"]
        )
        self.assertEqual(summaries[0]["controller_fingerprint"], summaries[1]["controller_fingerprint"])


if __name__ == "__main__":
    unittest.main()
