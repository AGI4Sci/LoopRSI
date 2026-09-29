"""Safe request builder for the declared Lingshu inference path."""

from pathlib import Path
from typing import Mapping, Optional, Tuple

from .base import VCC25ModelAdapter


class LingshuAdapter(VCC25ModelAdapter):
    adapter_id = "vcc25.lingshu"
    allowed_actions = ("predict", "convert_output")
    allowed_executables = frozenset({"python3"})

    def __init__(
        self,
        repository_dir: Optional[Path] = None,
        executable: str = "python3",
    ) -> None:
        root = repository_dir if repository_dir is not None else Path.cwd()
        self.repository_dir = Path(root)
        self.executable = executable

    def _command(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Tuple[str, ...]:
        self._require_input_roles(
            inputs,
            ("checkpoint", "gene_embeddings", "reference_dataset", "condition_csv"),
        )
        return (
            self.executable,
            "-m",
            "workflows.infer.main",
            "--config-name",
            "infer/legacy_vcc",
            f"checkpoint.model_path={inputs['checkpoint']}",
            f"output.workdir={output}",
            f"task.condition_csv={inputs['condition_csv']}",
            f"model.config.gene_embedding_file={inputs['gene_embeddings']}",
            "sampler.steps=3",
            "sampler.cfg_scale=2.0",
            f"sampler.clue_csv={inputs['reference_dataset']}",
        )

    def _internal_environment(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Mapping[str, str]:
        repository = self._safe_absolute_path(str(self.repository_dir), "repository_dir")
        return {"PYTHONPATH": repository}
