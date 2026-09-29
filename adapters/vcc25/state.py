"""Safe request builder for STATE's ``state tx`` CLI."""

from typing import Mapping, Tuple

from .base import VCC25ModelAdapter


class StateAdapter(VCC25ModelAdapter):
    adapter_id = "vcc25.state"
    allowed_actions = ("train", "predict")
    allowed_executables = frozenset({"state"})

    def __init__(self, executable: str = "state") -> None:
        self.executable = executable

    def _command(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Tuple[str, ...]:
        if action == "train":
            self._require_input_roles(inputs, ("toml_config",))
            return (
                self.executable,
                "tx",
                "train",
                f"data.kwargs.toml_config_path={inputs['toml_config']}",
                f"output_dir={output}",
                "name=vcc25_candidate",
            )
        self._require_input_roles(inputs, ("model_dir", "checkpoint", "adata"))
        return (
            self.executable,
            "tx",
            "infer",
            "--model-dir",
            inputs["model_dir"],
            "--checkpoint",
            inputs["checkpoint"],
            "--adata",
            inputs["adata"],
            "--output",
            f"{output}/predictions.h5ad",
        )
