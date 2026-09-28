"""Function-call friendly facade shared by users, Heuresis, and CLI clients."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


class LoopServiceError(ValueError):
    pass


class ResearchLoop:
    """Compose dataset and task adapters without exposing physical paths."""

    protocol_version = "omni-ar-loop/v1"

    def __init__(self, repository: str | Path | None = None):
        self.repository = Path(repository or Path(__file__).resolve().parents[1]).resolve()
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))

    def _envelope(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": self.protocol_version,
            "status": "ok",
            "operation": operation,
            "payload": payload,
        }

    def _registry(self):
        from datasets.registry import DatasetRegistry

        return DatasetRegistry(self.repository / "datasets")

    def datasets(self) -> dict[str, Any]:
        registry = self._registry()
        return self._envelope("datasets", {
            "datasets": registry.entries(),
            "channels": registry.document.get("channels", {}),
        })

    def dataset_loading(
        self,
        dataset_ref: str,
        *,
        mode: str = "default",
        inspect_limit: int = 5,
        deep_validate: bool = False,
        prepare: bool = False,
    ) -> dict[str, Any]:
        """Select a predeclared usage mode and return a safe logical binding."""
        registry = self._registry()
        adapter = registry.load_adapter(dataset_ref)
        modes = adapter.usage_modes()
        selected = modes.get(mode)
        if selected is None:
            raise LoopServiceError(
                f"unknown usage mode {mode!r}; choose one of {sorted(modes)}"
            )
        validation = adapter.dispatch("validate", deep=deep_validate)
        if not validation["payload"].get("valid", False):
            raise LoopServiceError(f"dataset validation failed for {dataset_ref}")
        preparation = None
        profile = selected.get("preparation_profile")
        if prepare and profile:
            preparation = adapter.dispatch("prepare", profile=profile)
        binding = registry.binding(
            dataset_ref,
            artifact=selected.get("artifact"),
            split=selected.get("split"),
        )
        ready = selected.get("status") != "declared_not_materialized"
        payload = {
            "dataset_ref": dataset_ref,
            "mode": mode,
            "mode_spec": selected,
            "ready_for_training": ready,
            "capabilities": adapter.dispatch("capabilities")["payload"],
            "inspection": adapter.dispatch("inspect", limit=min(max(inspect_limit, 1), 20))["payload"],
            "validation": validation["payload"],
            "preparation": preparation["payload"] if preparation else None,
            "binding": self._logical_binding(binding),
        }
        self._redact_paths(payload)
        return self._envelope("dataset_loading", payload)

    def dataset_tool(self, dataset_ref: str, action: str, **arguments: Any) -> dict[str, Any]:
        """Execute one bounded Dataset Adapter operation for an LLM tool call."""
        from controller_bash.scripts.dataset_tools import execute_tool_call

        call = {
            "schema_version": "omni-ar-tool-call/v1",
            "tool": "dataset_adapter",
            "call_id": str(arguments.pop("call_id", "loop-dataset-call")),
            "arguments": {"action": action, "dataset_ref": dataset_ref, **arguments},
        }
        return self._envelope(
            "dataset_tool",
            execute_tool_call(call, dataset_ref=dataset_ref, repository=self.repository),
        )

    def design_code(
        self, task_spec: str | Path, proposal: dict[str, Any], *, dataset_mode: str | None = None,
    ) -> dict[str, Any]:
        """Compile a command-free proposal through the configured Task Adapter."""
        from controller_bash.scripts.task_contract import load_task_adapter, load_task_spec

        spec_path = self._scoped(task_spec)
        spec = load_task_spec(spec_path)
        adapter = load_task_adapter(spec, spec_path)
        defaults = dict(spec["adapter"]["trial_defaults"])
        mode_spec = None
        dataset_ref = (spec.get("data") or {}).get("dataset_ref")
        if dataset_mode:
            if not dataset_ref:
                raise LoopServiceError("dataset_mode requires task_spec.data.dataset_ref")
            dataset_adapter = self._registry().load_adapter(str(dataset_ref))
            mode_spec = dataset_adapter.usage_modes().get(dataset_mode)
            if mode_spec is None:
                raise LoopServiceError(f"unknown dataset mode {dataset_mode!r}")
            if mode_spec.get("status") == "declared_not_materialized":
                raise LoopServiceError(f"dataset mode {dataset_mode!r} is not ready for training")
            defaults.update(mode_spec.get("task_parameters") or {})
        proposal_body = dict(proposal)
        proposal_id = str(proposal_body.pop("proposal_id", "experiment_000"))
        trial = adapter.proposal_to_trial(
            proposal_body,
            defaults,
            proposal_id,
        )
        return self._envelope("design_code", {
            "task": spec["task"]["name"],
            "dataset_ref": dataset_ref,
            "dataset_mode": dataset_mode,
            "dataset_mode_spec": mode_spec,
            "trial": trial,
            "execution_authorized": False,
            "next": "Call run_trial(..., execute=True) explicitly after resource review.",
        })

    def run_trial(
        self,
        task_spec: str | Path,
        proposal: dict[str, Any],
        *,
        dataset_mode: str | None = None,
        execute: bool = False,
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compile by default; execute locally only after explicit authorization."""
        designed = self.design_code(task_spec, proposal, dataset_mode=dataset_mode)
        if not execute:
            return self._envelope("run_trial", {
                **designed["payload"],
                "status": "planned",
            })
        if output is None:
            raise LoopServiceError("output is required when execute=True")
        from controller_bash.scripts.task_contract import load_task_adapter, load_task_spec

        spec_path = self._scoped(task_spec)
        spec = load_task_spec(spec_path)
        adapter = load_task_adapter(spec, spec_path)
        request = dict(designed["payload"]["trial"]["cli_overrides"])
        request["output"] = str(self._scoped(output))
        result = adapter.dispatch("run_trial", request)
        return self._envelope("run_trial", {"status": "executed", "result": result})

    def evaluate(self, task_spec: str | Path, request: dict[str, Any]) -> dict[str, Any]:
        return self._task_action(task_spec, "evaluate", request)

    def summarize_results(self, task_spec: str | Path, request: dict[str, Any]) -> dict[str, Any]:
        return self._task_action(task_spec, "summarize_results", request)

    def _task_action(self, task_spec: str | Path, action: str, request: dict[str, Any]) -> dict[str, Any]:
        from controller_bash.scripts.task_contract import load_task_adapter, load_task_spec

        spec_path = self._scoped(task_spec)
        spec = load_task_spec(spec_path)
        result = load_task_adapter(spec, spec_path).dispatch(action, dict(request))
        return self._envelope(action, result)

    def _scoped(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repository / candidate
        candidate = candidate.resolve()
        if candidate != self.repository and self.repository not in candidate.parents:
            raise LoopServiceError(f"path escapes repository: {path}")
        return candidate

    @staticmethod
    def _logical_binding(binding: dict[str, Any]) -> dict[str, Any]:
        logical = json.loads(json.dumps(binding))
        logical.pop("pack_root", None)
        for artifact in logical.get("resolved", {}).get("artifacts", []):
            artifact.pop("path", None)
        return logical

    @staticmethod
    def _redact_paths(value: Any) -> None:
        if isinstance(value, dict):
            for key in list(value):
                if key in {"path", "pack_root", "output", "command"}:
                    value.pop(key)
                else:
                    ResearchLoop._redact_paths(value[key])
        elif isinstance(value, list):
            for item in value:
                ResearchLoop._redact_paths(item)


loop = ResearchLoop()
