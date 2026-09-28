"""Bounded Dataset Adapter tools exposed to the Heuresis planning loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


READ_ACTIONS = {
    "describe", "inspect", "list_splits", "list_modes", "load",
    "capabilities", "validate", "load_hierarchy", "validate_hierarchy",
    "hierarchy_statistics", "build_negative_control",
}
CONTROLLED_ACTIONS = {"prepare", "resolve"}
ALLOWED_ACTIONS = READ_ACTIONS | CONTROLLED_ACTIONS
MAX_INSPECT_ROWS = 20


def tool_schema(dataset_ref: str) -> dict[str, Any]:
    return {
        "name": "dataset_adapter",
        "description": "Inspect and prepare a registered dataset through its versioned adapter; never guess physical paths.",
        "input_schema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {"enum": sorted(ALLOWED_ACTIONS)},
                "dataset_ref": {"const": dataset_ref},
                "artifact": {"type": "string"},
                "split": {"type": "string"},
                "profile": {"type": "string"},
                "mode": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_INSPECT_ROWS},
                "deep": {"type": "boolean"},
                "hierarchy": {"type": "string"},
                "strategy": {"enum": ["shuffle_target_annotations", "degree_preserving_target_swap"]},
                "seed": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    }


def is_tool_call(value: object) -> bool:
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "omni-ar-tool-call/v1"
        and value.get("tool") == "dataset_adapter"
        and isinstance(value.get("arguments"), dict)
    )


def execute_tool_call(call: dict[str, Any], *, dataset_ref: str, repository: Path) -> dict[str, Any]:
    arguments = dict(call["arguments"])
    action = arguments.pop("action", None)
    requested_ref = arguments.pop("dataset_ref", dataset_ref)
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"dataset tool action is not allowed: {action!r}")
    if requested_ref != dataset_ref:
        raise ValueError("dataset tool cannot switch the task's bound dataset_ref")
    unknown = set(arguments).difference({
        "artifact", "split", "profile", "mode", "limit", "deep",
        "hierarchy", "strategy", "seed",
    })
    if unknown:
        raise ValueError(f"unsupported dataset tool arguments: {sorted(unknown)}")
    arguments = {key: value for key, value in arguments.items() if value is not None}
    if "limit" in arguments:
        arguments["limit"] = max(1, min(int(arguments["limit"]), MAX_INSPECT_ROWS))

    import sys

    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from datasets.registry import DatasetRegistry

    adapter = DatasetRegistry(repository / "datasets").load_adapter(dataset_ref)
    result = adapter.dispatch(action, **arguments)
    # Paths are controller details. The model receives logical artifact names and metadata only.
    redacted = json.loads(json.dumps(result))
    _redact_paths(redacted)
    return {
        "schema_version": "omni-ar-tool-result/v1",
        "tool": "dataset_adapter",
        "call_id": str(call.get("call_id", "dataset-call")),
        "arguments": {"action": action, "dataset_ref": dataset_ref, **arguments},
        "result": redacted,
    }


def _redact_paths(value: Any) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if key in {"path", "pack_root", "output", "command"}:
                value.pop(key)
            else:
                _redact_paths(value[key])
    elif isinstance(value, list):
        for item in value:
            _redact_paths(item)
