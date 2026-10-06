"""Shared validation for declaration-only model adapters."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from ai4ai.plugin_protocols import CandidateExecutionRequest
from domain_knowledge import KnowledgeCard
from domain_knowledge.store import assert_safe_knowledge


@dataclass(frozen=True)
class AssetCheck:
    status: str
    missing: Tuple[str, ...]
    evidence: Mapping[str, Any]


class VCC25ModelAdapter:
    """Validate an allowlisted recipe and build a typed request; never execute it."""

    adapter_id = ""
    allowed_actions: Tuple[str, ...] = ()
    allowed_executables = frozenset()
    allowed_environment = frozenset(
        {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "TOKENIZERS_PARALLELISM"}
    )

    def check_assets(
        self,
        model: KnowledgeCard,
        locations: Mapping[str, str],
    ) -> AssetCheck:
        self._validate_model(model)
        assert_safe_knowledge(locations, f"{self.adapter_id}.asset_locations")
        required_roles = tuple(artifact["role"] for artifact in model.get("artifacts", ()))
        found = {}
        missing = []
        for role in required_roles:
            location = locations.get(role)
            exists = bool(location) and Path(str(location)).expanduser().exists()
            found[role] = {"location": str(location) if location else None, "exists": exists}
            if not exists:
                missing.append(role)
        return AssetCheck(
            status="complete" if not missing else "missing",
            missing=tuple(missing),
            evidence={
                "adapter_id": self.adapter_id,
                "model_id": model.id,
                "model_readiness": model.execution_readiness,
                "assets": found,
                "execution_performed": False,
            },
        )

    def prepare(
        self,
        action: str,
        model: KnowledgeCard,
        context: Mapping[str, Any],
    ) -> CandidateExecutionRequest:
        self._validate_model(model)
        if action not in self.allowed_actions or action not in model.get("execution_recipe", {}).get(
            "actions", ()
        ):
            raise ValueError(f"action {action!r} is not declared for {self.adapter_id}")
        assert_safe_knowledge(context, f"{self.adapter_id}.context")
        working = self._safe_absolute_path(context.get("working_directory"), "working_directory")
        output = self._safe_absolute_path(context.get("output_location"), "output_location")
        inputs = context.get("inputs")
        if not isinstance(inputs, Mapping) or not inputs:
            raise ValueError("inputs must be a non-empty mapping")
        input_paths = []
        normalized_inputs = {}
        for role in sorted(inputs):
            path = self._safe_absolute_path(inputs[role], f"inputs.{role}")
            normalized_inputs[str(role)] = path
            input_paths.append(path)
        environment = context.get("environment", {})
        if not isinstance(environment, Mapping):
            raise ValueError("environment must be a mapping")
        undeclared = set(environment) - self.allowed_environment
        if undeclared:
            raise ValueError(f"undeclared environment keys: {sorted(undeclared)!r}")
        resources = self._validate_resources(model, context.get("resources", {}))
        command = self._command(action, normalized_inputs, output)
        if not isinstance(command, tuple) or not command:
            raise ValueError("adapter command must be a non-empty tuple")
        if Path(command[0]).name not in self.allowed_executables:
            raise ValueError(f"executable {command[0]!r} is not allowlisted for {self.adapter_id}")
        request_environment = {str(key): str(value) for key, value in environment.items()}
        request_environment.update(self._internal_environment(action, normalized_inputs, output))
        return CandidateExecutionRequest(
            candidate_id=f"{model.id}:{action}",
            command=command,
            working_directory=working,
            output_location=output,
            environment=request_environment,
            required_inputs=tuple(input_paths),
            expected_artifacts=self._expected_artifacts(action, output),
            runtime={"adapter_id": self.adapter_id, "action": action, **resources},
        )

    def _validate_model(self, model: KnowledgeCard) -> None:
        recipe = model.get("execution_recipe", {})
        if model.asset_type != "model" or recipe.get("adapter_id") != self.adapter_id:
            raise ValueError(
                f"model {model.id!r} does not declare adapter {self.adapter_id!r}"
            )

    @staticmethod
    def _safe_absolute_path(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be an absolute path")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"{field} contains unsafe characters")
        path = Path(value).expanduser()
        if ".." in path.parts:
            raise ValueError(f"{field} contains path traversal")
        if not path.is_absolute():
            raise ValueError(f"{field} must be absolute")
        return str(path.resolve())

    @staticmethod
    def _validate_resources(
        model: KnowledgeCard,
        requested: Any,
    ) -> Mapping[str, int]:
        if not isinstance(requested, Mapping):
            raise ValueError("resources must be a mapping")
        gpu_count = requested.get("gpu_count", 0)
        max_minutes = requested.get("max_minutes", 0)
        if not isinstance(gpu_count, int) or gpu_count < 0 or gpu_count > 1:
            raise ValueError("gpu_count exceeds the VCC25 adapter budget of 1")
        profile = model.get("resource_profile", {})
        model_gpu_limit = int(profile.get("gpu_count", 1))
        if gpu_count > model_gpu_limit:
            raise ValueError(f"gpu_count exceeds model limit {model_gpu_limit}")
        model_time_limit = min(int(profile.get("max_minutes", 180)), 180)
        if not isinstance(max_minutes, int) or max_minutes <= 0 or max_minutes > model_time_limit:
            raise ValueError(f"max_minutes must be between 1 and {model_time_limit}")
        return {"gpu_count": gpu_count, "max_minutes": max_minutes}

    def _command(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Tuple[str, ...]:
        raise NotImplementedError

    def _internal_environment(
        self,
        action: str,
        inputs: Mapping[str, str],
        output: str,
    ) -> Mapping[str, str]:
        return {}

    @staticmethod
    def _require_input_roles(inputs: Mapping[str, str], roles: Sequence[str]) -> None:
        missing = tuple(role for role in roles if role not in inputs)
        if missing:
            raise ValueError(f"missing required input roles: {missing!r}")

    @staticmethod
    def _input_arguments(inputs: Mapping[str, str]) -> Tuple[str, ...]:
        arguments = []
        for role, path in inputs.items():
            arguments.extend((f"--{role.replace('_', '-')}", path))
        return tuple(arguments)

    @staticmethod
    def _expected_artifacts(action: str, output: str) -> Tuple[str, ...]:
        name = "model" if action == "train" else "predictions"
        return (str(Path(output) / name),)
