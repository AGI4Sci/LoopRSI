"""Uniform adapter for paper implementations without a native VCC25 wrapper."""

from typing import Mapping, Tuple

from .base import VCC25ModelAdapter


class ExternalPaperAdapter(VCC25ModelAdapter):
    """Build a safe request for an allowlisted upstream reproduction runner.

    The runner performs preflight first and must not be treated as a successful
    reproduction until it emits a validated EvidenceCard.
    """

    allowed_actions = ("prepare", "train", "predict", "convert_output")
    allowed_executables = frozenset({"python3"})

    def __init__(self, method_key: str, adapter_id: str) -> None:
        self.method_key = method_key
        self.adapter_id = adapter_id

    def _command(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Tuple[str, ...]:
        self._require_input_roles(inputs, ("source_repository",))
        return (
            "python3",
            "-m",
            "adapters.vcc25.reproduction_entry",
            "--method",
            self.method_key,
            "--action",
            action,
            "--source-repository",
            inputs["source_repository"],
            "--output",
            output,
        )
