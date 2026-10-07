import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapters.vcc25 import get_adapter
from adapters.vcc25.lingshu import LingshuAdapter
from adapters.vcc25.perturbench import PerturBenchAdapter
from adapters.vcc25.primeflow import PrimeFlowAdapter
from adapters.vcc25.state import StateAdapter
from domain_knowledge import KnowledgeQuery, KnowledgeStore


ROOT = Path(__file__).resolve().parents[2]
STORE = KnowledgeStore.from_directory(ROOT / "knowledge" / "vcc25")
MODELS = {
    card.id: card
    for card in STORE.query(
        KnowledgeQuery(task_id="vcc25", asset_types=("model",), max_results=20)
    )
}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.work = self.root / "work"
        self.output = self.root / "output"
        self.repo.mkdir()
        self.work.mkdir()
        self.output.mkdir()
        self.input_path = self.root / "training.h5"
        self.input_path.touch()
        self.script = self.repo / "inference.py"
        self.script.touch()

    def tearDown(self):
        self.tmp.cleanup()

    def _context(self, **updates):
        value = {
            "working_directory": str(self.work),
            "output_location": str(self.output),
            "inputs": {"training_data": str(self.input_path)},
            "environment": {"CUDA_VISIBLE_DEVICES": "0"},
            "resources": {"gpu_count": 1, "max_minutes": 60},
        }
        value.update(updates)
        return value

    def _lingshu_inputs(self):
        return {
            "checkpoint": str(self.input_path),
            "gene_embeddings": str(self.input_path),
            "reference_dataset": str(self.input_path),
            "condition_csv": str(self.input_path),
        }

    def test_registry_is_exact_and_rejects_unknown_adapter(self):
        self.assertIsInstance(get_adapter("vcc25.lingshu"), LingshuAdapter)
        self.assertIsInstance(get_adapter("vcc25.state"), StateAdapter)
        self.assertIsInstance(get_adapter("vcc25.perturbench"), PerturBenchAdapter)
        self.assertIsInstance(get_adapter("vcc25.primeflow"), PrimeFlowAdapter)
        with self.assertRaisesRegex(ValueError, "unknown adapter"):
            get_adapter("vcc25.shell")

    def test_asset_checks_report_exact_missing_roles_without_changing_readiness(self):
        cases = [
            (
                LingshuAdapter(repository_dir=self.repo),
                MODELS["kb:model:lingshu-vcc-85m"],
                {"checkpoint": str(self.input_path)},
                ("gene_embeddings", "reference_dataset"),
            ),
            (
                StateAdapter(),
                MODELS["kb:model:state-st-hvg-replogle"],
                {},
                ("checkpoint",),
            ),
            (
                PerturBenchAdapter(),
                MODELS["kb:model:perturbench-latent-additive"],
                {},
                ("config",),
            ),
        ]
        with patch.object(subprocess, "run", side_effect=AssertionError("must not execute")):
            for adapter, model, locations, missing in cases:
                with self.subTest(adapter=adapter.adapter_id):
                    result = adapter.check_assets(model, locations)
                    self.assertEqual(result.status, "missing")
                    self.assertEqual(result.missing, missing)
                    self.assertEqual(result.evidence["model_readiness"], model.execution_readiness)
                    self.assertNotEqual(model.execution_readiness, "ready")

    def test_each_adapter_builds_a_typed_tuple_request(self):
        cases = [
            (
                LingshuAdapter(repository_dir=self.repo, executable="python3"),
                "kb:model:lingshu-vcc-85m",
                "predict",
                "python3",
                self._context(inputs=self._lingshu_inputs()),
            ),
            (
                StateAdapter(executable="state"),
                "kb:model:state-st-hvg-replogle",
                "train",
                "state",
                self._context(inputs={"toml_config": str(self.input_path)}),
            ),
            (
                PerturBenchAdapter(train_executable="train", predict_executable="predict"),
                "kb:model:perturbench-latent-additive",
                "predict",
                "predict",
                self._context(inputs={"config": str(self.input_path)}),
            ),
        ]
        for adapter, model_id, action, executable, context in cases:
            with self.subTest(adapter=adapter.adapter_id):
                request = adapter.prepare(action, MODELS[model_id], context)
                self.assertIsInstance(request.command, tuple)
                self.assertEqual(request.command[0], executable)
                self.assertEqual(request.working_directory, str(self.work.resolve()))
                self.assertEqual(request.output_location, str(self.output.resolve()))
                self.assertEqual(
                    request.required_inputs,
                    tuple(
                        str(Path(context["inputs"][role]).resolve())
                        for role in sorted(context["inputs"])
                    ),
                )
                self.assertEqual(request.runtime["gpu_count"], 1)
                self.assertEqual(request.runtime["max_minutes"], 60)
                self.assertNotIn("sh", request.command[:1])
                self.assertNotIn("bash", request.command[:1])

    def test_commands_match_the_documented_upstream_entrypoints(self):
        lingshu = LingshuAdapter(repository_dir=self.repo).prepare(
            "predict",
            MODELS["kb:model:lingshu-vcc-85m"],
            self._context(inputs=self._lingshu_inputs()),
        )
        self.assertEqual(lingshu.command[:4], ("python3", "-m", "workflows.infer.main", "--config-name"))
        self.assertNotIn("inference.py", lingshu.command)
        self.assertIn("checkpoint.model_path=" + str(self.input_path.resolve()), lingshu.command)
        self.assertIn("output.workdir=" + str(self.output.resolve()), lingshu.command)

        state_train = StateAdapter().prepare(
            "train",
            MODELS["kb:model:state-st-hvg-replogle"],
            self._context(inputs={"toml_config": str(self.input_path)}),
        )
        self.assertEqual(state_train.command[:3], ("state", "tx", "train"))
        self.assertIn(
            "data.kwargs.toml_config_path=" + str(self.input_path.resolve()),
            state_train.command,
        )
        self.assertIn("output_dir=" + str(self.output.resolve()), state_train.command)

        state_predict = StateAdapter().prepare(
            "predict",
            MODELS["kb:model:state-st-hvg-replogle"],
            self._context(
                inputs={
                    "model_dir": str(self.work),
                    "checkpoint": str(self.input_path),
                    "adata": str(self.input_path),
                }
            ),
        )
        self.assertEqual(state_predict.command[:3], ("state", "tx", "infer"))
        self.assertIn("--model-dir", state_predict.command)
        self.assertIn("--checkpoint", state_predict.command)
        self.assertIn("--adata", state_predict.command)

        perturbench = PerturBenchAdapter().prepare(
            "train",
            MODELS["kb:model:perturbench-latent-additive"],
            self._context(inputs={"config": str(self.input_path)}),
        )
        self.assertEqual(perturbench.command[0], "train")
        self.assertIn("model=latent_additive", perturbench.command)
        self.assertIn("hydra.run.dir=" + str(self.output.resolve()), perturbench.command)

        primeflow = PrimeFlowAdapter().prepare(
            "train",
            MODELS["kb:model:primeflow"],
            self._context(
                inputs={
                    "training_data": str(self.input_path),
                    "split_csv": str(self.root / "split.csv"),
                    "gene_features": str(self.root / "genes.csv"),
                },
                resources={"gpu_count": 1, "max_minutes": 30},
            ),
        )
        self.assertEqual(primeflow.command[0], "primeflow.train")
        self.assertIn("trainer.devices=1", primeflow.command)
        self.assertIn("trainer.num_nodes=1", primeflow.command)
        self.assertIn("trainer.max_epochs=1", primeflow.command)
        self.assertIn("trainer.limit_train_batches=2", primeflow.command)
        self.assertIn("trainer.limit_val_batches=2", primeflow.command)
        self.assertIn("hydra.run.dir=" + str(self.output.resolve()), primeflow.command)
        self.assertFalse(any("test" in item.lower() for item in primeflow.required_inputs))

    def test_primeflow_accepts_only_bounded_training_inputs(self):
        adapter = PrimeFlowAdapter()
        model = MODELS["kb:model:primeflow"]
        context = self._context(
            inputs={
                "training_data": str(self.input_path),
                "split_csv": str(self.root / "split.csv"),
                "gene_features": str(self.root / "genes.csv"),
            },
            resources={"gpu_count": 1, "max_minutes": 30},
        )
        with self.assertRaisesRegex(ValueError, "action"):
            adapter.prepare("predict", model, context)
        with self.assertRaisesRegex(ValueError, "final-test"):
            adapter.prepare(
                "train",
                model,
                self._context(
                    inputs={
                        "training_data": str(self.root / "final_test" / "adata_Test.h5ad"),
                        "split_csv": str(self.root / "split.csv"),
                        "gene_features": str(self.root / "genes.csv"),
                    },
                    resources={"gpu_count": 1, "max_minutes": 30},
                ),
            )

    def test_adapter_accepts_only_its_declared_actions_and_matching_model(self):
        adapter = LingshuAdapter(repository_dir=self.repo)
        with self.assertRaisesRegex(ValueError, "action"):
            adapter.prepare("train", MODELS["kb:model:lingshu-vcc-85m"], self._context())
        with self.assertRaisesRegex(ValueError, "adapter"):
            adapter.prepare("predict", MODELS["kb:model:state-st-hvg-replogle"], self._context())

    def test_adapter_rejects_an_executable_outside_its_allowlist(self):
        adapter = LingshuAdapter(repository_dir=self.repo, executable="bash")
        with self.assertRaisesRegex(ValueError, "executable"):
            adapter.prepare(
                "predict",
                MODELS["kb:model:lingshu-vcc-85m"],
                self._context(inputs=self._lingshu_inputs()),
            )

    def test_prepare_rejects_path_traversal_and_non_absolute_boundaries(self):
        adapter = StateAdapter()
        model = MODELS["kb:model:state-st-hvg-replogle"]
        with self.assertRaisesRegex(ValueError, "traversal"):
            adapter.prepare("train", model, self._context(output_location=str(self.root / "x" / ".." / "out")))
        with self.assertRaisesRegex(ValueError, "absolute"):
            adapter.prepare("train", model, self._context(working_directory="relative/work"))

    def test_prepare_rejects_undeclared_environment_and_restricted_identifiers(self):
        adapter = StateAdapter()
        model = MODELS["kb:model:state-st-hvg-replogle"]
        with self.assertRaisesRegex(ValueError, "environment"):
            adapter.prepare("train", model, self._context(environment={"LD_PRELOAD": "/tmp/x"}))
        with self.assertRaisesRegex(ValueError, "competition_test"):
            adapter.prepare(
                "train",
                model,
                self._context(inputs={"training_data": str(self.root / "competition_test" / "x")}),
            )

    def test_prepare_rejects_resource_requests_above_budget(self):
        adapter = PerturBenchAdapter()
        model = MODELS["kb:model:perturbench-latent-additive"]
        with self.assertRaisesRegex(ValueError, "gpu_count"):
            adapter.prepare("train", model, self._context(resources={"gpu_count": 2, "max_minutes": 60}))
        with self.assertRaisesRegex(ValueError, "max_minutes"):
            adapter.prepare("train", model, self._context(resources={"gpu_count": 1, "max_minutes": 999}))


if __name__ == "__main__":
    unittest.main()
