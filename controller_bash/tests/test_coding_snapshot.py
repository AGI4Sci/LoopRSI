from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "controller_bash/scripts"))

from run_coding_agent import (  # noqa: E402
    create_worktree, ensure_bubblewrap, file_hashes, materialize_dataset_binding,
    materialize_source_snapshot, run_direct_codex,
)


class CodingSnapshotTests(unittest.TestCase):
    def test_file_hashes_ignore_git_worktree_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").write_text("gitdir: /tmp/example\n", encoding="utf-8")
            (root / "model.py").write_text("VALUE = 1\n", encoding="utf-8")

            self.assertEqual(set(file_hashes(root)), {"model.py"})

    def test_dirty_source_is_snapshotted_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, worktree, output = root / "repository", root / "worktree", root / "output"
            repository.mkdir()
            output.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repository, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
            (repository / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "model.py"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
            (repository / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
            (repository / "new_module.py").write_text("ACTIVE = True\n", encoding="utf-8")
            (repository / ".env").write_text("SECRET=hidden\n", encoding="utf-8")

            create_worktree(repository, worktree)
            manifest = materialize_source_snapshot(repository, worktree, output)

            self.assertEqual((worktree / "model.py").read_text(), "VALUE = 2\n")
            self.assertEqual((worktree / "new_module.py").read_text(), "ACTIVE = True\n")
            self.assertFalse((worktree / ".env").exists())
            self.assertEqual(manifest["untracked_sources"], ["new_module.py"])

    def test_dataset_payload_is_hardlinked_into_candidate_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository, worktree, output = root / "repository", root / "worktree", root / "output"
            pack = repository / "datasets/demo/data"
            pack.mkdir(parents=True)
            worktree.mkdir()
            output.mkdir()
            (repository / "datasets/registry.yaml").write_text(
                "schema_version: omni-ar-dataset-registry/v1\ndatasets:\n- ref: demo@1\n  path: demo\n",
                encoding="utf-8",
            )
            (repository / "datasets/demo/dataset.yaml").write_text(
                "schema_version: omni-ar-dataset/v1\ndataset: {id: demo, version: '1', name: demo, modality: text}\n"
                "format: {kind: text_csv}\ndefault_artifact: raw\nartifacts:\n  raw: {path: data/raw.tsv, role: raw}\n"
                "splits: {}\npreparation: {profiles: {}}\nlineage: {source: {kind: test}, transformations: []}\n",
                encoding="utf-8",
            )
            source = pack / "raw.tsv"
            source.write_text("text\tlabel\nhello\t1\n", encoding="utf-8")
            record = materialize_dataset_binding(
                repository, worktree, {"data": {"dataset_ref": "demo@1"}}, output,
            )
            candidate = worktree / "datasets/demo/data/raw.tsv"
            self.assertTrue(candidate.is_file())
            self.assertEqual(source.stat().st_ino, candidate.stat().st_ino)
            self.assertEqual(record["mode"], "hardlink")
            self.assertEqual(record["linked_files"][0]["path"], "data/raw.tsv")

    def test_codex_bundled_bubblewrap_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "codex"
            agent = root / "bin/codex.js"
            bundled = root / "node_modules/@openai/codex-linux-x64/vendor/x/codex-resources/bwrap"
            agent.parent.mkdir(parents=True)
            bundled.parent.mkdir(parents=True)
            agent.write_text("", encoding="utf-8")
            bundled.write_text("", encoding="utf-8")
            bundled.chmod(0o755)

            def which(name: str):
                return None if name == "bwrap" else str(agent)

            with patch("run_coding_agent.shutil.which", side_effect=which), patch.dict(
                "os.environ", {"PATH": "/usr/bin"}, clear=False,
            ):
                selected = ensure_bubblewrap("codex")
            self.assertEqual(selected, str(bundled.resolve()))

    def test_direct_codex_is_ephemeral_scoped_and_strips_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, agent_dir = root / "source", root / "agent"
            source.mkdir()
            (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            calls = []

            def fake_runner(command, **kwargs):
                calls.append((command, kwargs))
                return subprocess.CompletedProcess(command, 0, "done\n", "")

            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "must-not-leak",
                    "OMNI_AR_CODEX_BASE_URL": "https://example.invalid/v1",
                },
                clear=False,
            ):
                run = run_direct_codex(
                    "codex",
                    "gpt-test",
                    source,
                    agent_dir,
                    "edit only model.py",
                    timeout=30,
                    runner=fake_runner,
                )

            self.assertEqual(run.exit_code, 0)
            self.assertEqual((agent_dir / "solution/model.py").read_text(), "VALUE = 1\n")
            command, kwargs = calls[0]
            self.assertIn("--ephemeral", command)
            self.assertIn("--ignore-user-config", command)
            self.assertIn("workspace-write", command)
            self.assertEqual(command[command.index("--cd") + 1], str(agent_dir / "solution"))
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            self.assertEqual(kwargs["input"], "edit only model.py")


if __name__ == "__main__":
    unittest.main()
