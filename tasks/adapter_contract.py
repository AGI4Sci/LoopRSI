from __future__ import annotations

import argparse
import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping


ACTIONS = (
    "prepare_data",
    "validate_data",
    "run_baseline",
    "run_trial",
    "evaluate",
    "summarize_results",
)

EXPERIMENT_PROPOSAL_REQUIRED_FIELDS = {
    "hypothesis",
    "change_scope",
    "parameters",
    "expected_effect",
    "acceptance_criteria",
    "resource_request",
}
EXPERIMENT_PROPOSAL_FIELDS = {
    *EXPERIMENT_PROPOSAL_REQUIRED_FIELDS,
    "mechanism_alignment",
}


class AdapterError(RuntimeError):
    pass


class TaskAdapter(ABC):
    task_name: str
    trial_parameters: frozenset[str] | None = None
    method_capabilities: frozenset[str] = frozenset()
    method_capability_details: Mapping[str, Mapping[str, Any]] = {}
    parameter_capabilities: Mapping[str, Mapping[str, Any]] = {}

    @abstractmethod
    def prepare_data(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def validate_data(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def run_baseline(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def run_trial(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def summarize_results(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def dispatch(self, action: str, request: dict[str, Any]) -> dict[str, Any]:
        if action not in ACTIONS:
            raise AdapterError(f"unsupported adapter action: {action}")
        method: Callable[[dict[str, Any]], dict[str, Any]] = getattr(self, action)
        result = method(request)
        if not isinstance(result, dict):
            raise AdapterError(f"{action} must return a mapping")
        if action == "run_trial":
            # Preserve the exact values received by the Adapter.  The
            # controller uses this task-agnostic receipt to prove that a
            # generated candidate and its combination controls were activated
            # with the parameters that were proposed.
            result.setdefault(
                "executed_parameters",
                {key: value for key, value in request.items() if key != "output"},
            )
        result.setdefault("status", "ok")
        result.setdefault("task_name", self.task_name)
        result.setdefault("adapter_action", action)
        return result

    def initialization_capabilities(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Describe task choices exposed by the pre-research initializer.

        The fallback is derived only from the task spec. Tasks may override it
        with reviewed protocols and baselines without teaching the controller
        domain-specific concepts.
        """
        search = spec.get("search") or {}
        generic_operations = [
            "representation", "model", "objective", "regularization",
            "sampling", "optimization", "inference", "evaluation",
        ]
        defaults = dict((spec.get("adapter") or {}).get("trial_defaults") or {})
        methods = set(self.method_capabilities)
        default_method = defaults.get("method")
        if isinstance(default_method, str) and default_method:
            methods.add(default_method)
        parameter_details: dict[str, dict[str, Any]] = {}
        if self.trial_parameters is not None:
            for name in sorted(self.trial_parameters):
                value = defaults.get(name)
                if isinstance(value, bool):
                    kind = "boolean"
                elif isinstance(value, int):
                    kind = "integer"
                elif isinstance(value, float):
                    kind = "number"
                elif isinstance(value, str):
                    kind = "string"
                elif value is None:
                    kind = "scalar"
                else:
                    kind = type(value).__name__
                parameter_details[name] = {"type": kind}
        parameter_details.update({
            str(name): dict(detail)
            for name, detail in self.parameter_capabilities.items()
        })
        method_details = {
            name: {"available": True, "source": "task_adapter"}
            for name in sorted(methods)
        }
        for name, detail in self.method_capability_details.items():
            method_details.setdefault(str(name), {}).update(dict(detail))
        return {
            "metrics": spec.get("metrics") or {},
            "protocols": ["default"],
            "default_protocol": "default",
            "baseline": None,
            "protected_items": list(search.get("protected_paths") or []),
            "operation_capabilities": generic_operations,
            "resources": spec.get("resources") or {},
            "trial_parameters": (
                sorted(self.trial_parameters)
                if self.trial_parameters is not None else None
            ),
            "trial_defaults": defaults,
            "method_capabilities": sorted(methods),
            "method_capability_details": method_details,
            "parameter_capabilities": parameter_details,
        }

    def proposal_to_trial(
        self,
        proposal: dict[str, Any],
        defaults: dict[str, Any],
        name: str,
    ) -> dict[str, Any]:
        """Compile a command-free experiment proposal into a run_trial request."""
        missing = sorted(EXPERIMENT_PROPOSAL_REQUIRED_FIELDS.difference(proposal))
        extra = sorted(set(proposal).difference(EXPERIMENT_PROPOSAL_FIELDS))
        if missing or extra:
            raise AdapterError(f"invalid experiment proposal fields; missing={missing}, extra={extra}")
        parameters = proposal.get("parameters")
        if not isinstance(parameters, dict):
            raise AdapterError("experiment proposal parameters must be an object")
        invalid_names = sorted(
            str(key) for key in parameters
            if not str(key).replace("_", "").isalnum() or str(key).startswith("_")
        )
        if invalid_names:
            raise AdapterError(f"parameter names must use snake_case: {invalid_names}")
        compiled = dict(defaults)
        compiled.update(parameters)
        if self.trial_parameters is not None:
            unsupported = sorted(set(compiled).difference(self.trial_parameters))
            if unsupported:
                raise AdapterError(f"unsupported {self.task_name} trial parameters: {unsupported}")
        if self.method_capabilities:
            allowed_methods = set(self.method_capabilities)
            default_method = defaults.get("method")
            if isinstance(default_method, str) and default_method:
                allowed_methods.add(default_method)
            method = compiled.get("method")
            if method not in allowed_methods:
                raise AdapterError(
                    f"unsupported {self.task_name} method {method!r}; "
                    f"choose one of {sorted(allowed_methods)}"
                )
        for parameter, detail in self.parameter_capabilities.items():
            if parameter not in compiled:
                continue
            value = compiled[parameter]
            expected = detail.get("type")
            valid_type = (
                expected in {None, "scalar"}
                or (expected == "boolean" and isinstance(value, bool))
                or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
                or (expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
                or (expected == "string" and isinstance(value, str))
            )
            if not valid_type:
                raise AdapterError(
                    f"{self.task_name} parameter {parameter!r} must have type {expected}"
                )
            if "enum" in detail and value not in detail["enum"]:
                raise AdapterError(
                    f"{self.task_name} parameter {parameter!r} must be one of {detail['enum']}"
                )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in detail and value < detail["minimum"]:
                    raise AdapterError(
                        f"{self.task_name} parameter {parameter!r} must be >= {detail['minimum']}"
                    )
                if "maximum" in detail and value > detail["maximum"]:
                    raise AdapterError(
                        f"{self.task_name} parameter {parameter!r} must be <= {detail['maximum']}"
                    )
        return {
            "name": name,
            "entrypoint": "run_trial",
            "cli_overrides": compiled,
            "purpose": str(proposal["hypothesis"]),
            "expected_signal": proposal["expected_effect"],
            "change_scope": proposal["change_scope"],
            "acceptance_criteria": proposal["acceptance_criteria"],
            "resource_request": proposal["resource_request"],
            "structured_proposal": proposal,
        }


def merge_research_capabilities(
    task_capabilities: Mapping[str, Any],
    dataset_capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge data-only and task-only capabilities without domain logic."""
    payload = dataset_capabilities.get("payload", dataset_capabilities)
    if not isinstance(payload, Mapping):
        raise AdapterError("dataset capabilities payload must be a mapping")
    merged = dict(task_capabilities)
    merged.update({
        "data_actions": list(payload.get("actions") or []),
        "data_preparation_profiles": list(payload.get("profiles") or []),
        "data_usage_modes": list(payload.get("usage_modes") or []),
        "data_features": dict(payload.get("features") or {}),
        "capability_sources": {
            "data": "dataset_adapter",
            "task": "task_adapter",
        },
    })
    return merged


def repository_root(file: str | Path) -> Path:
    start = Path(file).resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "controller_bash").is_dir() and (parent / "tasks").is_dir():
            return parent
    raise AdapterError(f"cannot locate repository root from {file}")


def parse_scalar(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def unknown_args_to_request(items: list[str]) -> dict[str, Any]:
    request: dict[str, Any] = {}
    index = 0
    while index < len(items):
        token = items[index]
        if not token.startswith("--"):
            raise AdapterError(f"unexpected adapter argument: {token}")
        key = token[2:].replace("-", "_")
        if index + 1 < len(items) and not items[index + 1].startswith("--"):
            request[key] = parse_scalar(items[index + 1])
            index += 2
        else:
            request[key] = True
            index += 1
    return request


def run_command(command: list[str], cwd: Path, log_prefix: Path) -> subprocess.CompletedProcess[str]:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    log_prefix.with_suffix(".stdout.log").write_text(proc.stdout, encoding="utf-8", errors="replace")
    log_prefix.with_suffix(".stderr.log").write_text(proc.stderr, encoding="utf-8", errors="replace")
    if proc.returncode:
        stderr_tail = proc.stderr.strip()[-4000:]
        raise AdapterError(
            f"backend command failed with exit code {proc.returncode}; "
            f"see {log_prefix.with_suffix('.stderr.log')}; stderr tail: {stderr_tail or '<empty>'}"
        )
    return proc


def load_json_or_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AdapterError(f"result not found: {path}")
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise AdapterError(f"result is empty: {path}")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = json.loads(lines[-1])
    if not isinstance(obj, dict):
        raise AdapterError(f"result must contain a JSON object: {path}")
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def adapter_main(adapter: TaskAdapter) -> int:
    parser = argparse.ArgumentParser(description=f"{adapter.task_name} OMNI-AR task adapter")
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args, unknown = parser.parse_known_args()
    request: dict[str, Any] = {}
    if args.request:
        loaded = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise AdapterError("--request must contain a JSON object")
        request.update(loaded)
    request.update(unknown_args_to_request(unknown))
    request["output"] = str(args.output.resolve())
    try:
        result = adapter.dispatch(args.action, request)
    except (AdapterError, OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "status": "failed",
            "task_name": adapter.task_name,
            "adapter_action": args.action,
            "error": str(exc),
        }
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 2
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0
