"""GEARS training request for a prepared H1 train/validation dataset."""

from pathlib import Path
from typing import Mapping, Tuple

from .base import VCC25ModelAdapter


class GEARSAdapter(VCC25ModelAdapter):
    adapter_id = "vcc25.gears"
    allowed_actions = ("train",)
    allowed_executables = frozenset({"python3"})

    def _command(
        self, action: str, inputs: Mapping[str, str], output: str,
    ) -> Tuple[str, ...]:
        self._require_input_roles(inputs, (
            "source_repository", "processed_dataset", "split_json",
            "gene2go", "gene_names",
        ))
        runner = Path(__file__).with_name("gears_h1.py").resolve()
        return (
            "python3", str(runner),
            "--source-repository", inputs["source_repository"],
            "--processed-dataset", inputs["processed_dataset"],
            "--split-json", inputs["split_json"],
            "--gene2go", inputs["gene2go"],
            "--gene-names", inputs["gene_names"],
            "--output", output,
        )

    @staticmethod
    def _expected_artifacts(action: str, output: str) -> Tuple[str, ...]:
        return (str(Path(output) / "model"), str(Path(output) / "validation_predictions.npz"))
