import unittest
from pathlib import Path

from adapters.vcc25.probe_primeflow_offline_install import (
    activate_paths,
    build_install_command,
    build_resolved_install_command,
    resolve_assets,
)


class PrimeFlowOfflineInstallTests(unittest.TestCase):
    def test_install_command_uses_wheelhouse_without_replacing_torch(self):
        command = build_install_command(Path("/wheelhouse"), Path("/tmp/deps"))
        self.assertIn("--no-index", command)
        self.assertIn("--no-deps", command)
        self.assertIn("numpy==2.2.6", command)
        self.assertIn("pandas==2.3.3", command)
        self.assertIn("h5py==3.14.0", command)
        self.assertIn("lightning-utilities", command)
        self.assertIn("sqlparse", command)
        self.assertIn("httpx2==2.13.1", command)
        self.assertIn("httpcore2==2.13.1", command)
        self.assertIn("huggingface-hub==1.33.0", command)
        self.assertIn("colorlog", command)
        self.assertIn("torchcde", command)
        self.assertIn("torchdiffeq", command)
        self.assertIn("torchsde", command)
        self.assertIn("trampoline", command)
        self.assertNotIn("torch", command)

    def test_assets_are_resolved_inside_packaged_job_folder(self):
        wheelhouse, sources = resolve_assets(Path("/workdir/rjob/src"))
        self.assertEqual(wheelhouse, Path("/workdir/rjob/src/wheelhouse"))
        self.assertEqual(sources["primeflow"].parent, Path("/workdir/rjob/src/sources"))

    def test_mlflow_dependencies_are_resolved_only_from_wheelhouse(self):
        command = build_resolved_install_command(Path("/wheelhouse"), Path("/tmp/deps"))
        self.assertIn("--no-index", command)
        self.assertNotIn("--no-deps", command)
        self.assertIn("mlflow-skinny==3.16.1", command)
        self.assertIn("scanpy==1.12.4", command)
        self.assertIn("jax==0.6.1", command)
        self.assertIn("jaxlib==0.6.1", command)
        self.assertIn("ott-jax", command)

    def test_isolated_dependencies_precede_system_packages(self):
        paths = ["/system"]
        activate_paths(Path("/tmp/deps"), [Path("/src/primeflow")], paths)
        self.assertEqual(paths[:2], ["/src/primeflow", "/tmp/deps"])


if __name__ == "__main__":
    unittest.main()
