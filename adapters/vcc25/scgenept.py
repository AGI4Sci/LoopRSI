"""scGenePT GO-All H1 training request."""

from pathlib import Path
from typing import Mapping, Tuple

from .base import VCC25ModelAdapter


class ScGenePTAdapter(VCC25ModelAdapter):
    adapter_id = "vcc25.scgenept"
    allowed_actions = ("train",)
    allowed_executables = frozenset({"python3"})

    def _command(
        self, action: str, inputs: Mapping[str, str], output: str,
    ) -> Tuple[str, ...]:
        roles = (
            "source_repository", "processed_dataset", "split_json", "gene_names",
            "checkpoint", "vocab", "go_all_embedding",
        )
        self._require_input_roles(inputs, roles)
        command = ["python3", str(Path(__file__).with_name("scgenept_h1.py").resolve())]
        for role in roles:
            command.extend((f"--{role.replace('_', '-')}", inputs[role]))
        return (*command, "--output", output)

    @staticmethod
    def _expected_artifacts(action: str, output: str) -> Tuple[str, ...]:
        return (str(Path(output) / "model.pt"), str(Path(output) / "validation_predictions.npz"))
