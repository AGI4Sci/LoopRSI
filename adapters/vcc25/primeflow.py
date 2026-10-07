"""Bounded PRiMeFlow training adapter for H1 smoke validation."""

from pathlib import Path
from typing import Mapping, Tuple

from .base import VCC25ModelAdapter


class PrimeFlowAdapter(VCC25ModelAdapter):
    """Build a single-GPU, one-epoch PRiMeFlow training request."""

    adapter_id = "vcc25.primeflow"
    allowed_actions = ("train",)
    allowed_executables = frozenset({"primeflow.train"})

    def _command(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Tuple[str, ...]:
        self._require_input_roles(inputs, ("training_data", "split_csv", "gene_features"))
        for path in inputs.values():
            normalized = str(Path(path)).lower().replace("-", "_")
            if "final_test" in normalized or "adata_test" in normalized:
                raise ValueError("final-test inputs are forbidden for bounded PRiMeFlow training")
        return (
            "primeflow.train",
            "experiment=primeflow/vcc/flow_matching_gaussian_source_annbacked_unet",
            f"data.data.filename={inputs['training_data']}",
            f"data.splitter.split_path={inputs['split_csv']}",
            f"data.data_iter_factory.feature_filter_path={inputs['gene_features']}",
            "trainer.devices=1",
            "trainer.num_nodes=1",
            "trainer.min_epochs=1",
            "trainer.max_epochs=1",
            "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=2",
            "test=False",
            "data.loader.num_workers=2",
            f"hydra.run.dir={output}",
            "+finetune=True",
        )
