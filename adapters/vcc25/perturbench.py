"""Safe request builder for PerturBench train and predict entrypoints."""

from typing import Mapping, Tuple

from .base import VCC25ModelAdapter


class PerturBenchAdapter(VCC25ModelAdapter):
    adapter_id = "vcc25.perturbench"
    allowed_actions = ("train", "predict")
    allowed_executables = frozenset({"train", "predict"})

    def __init__(
        self,
        train_executable: str = "train",
        predict_executable: str = "predict",
    ) -> None:
        self.executables = {"train": train_executable, "predict": predict_executable}

    def _command(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Tuple[str, ...]:
        self._require_input_roles(inputs, ("config",))
        overrides = tuple(f"{role}={path}" for role, path in inputs.items())
        model_override = ("model=latent_additive",) if action == "train" else ()
        return (
            self.executables[action],
            *model_override,
            *overrides,
            f"hydra.run.dir={output}",
        )
