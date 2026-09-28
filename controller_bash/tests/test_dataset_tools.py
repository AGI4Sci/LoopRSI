from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "controller_bash/scripts/dataset_tools.py"
SPEC = importlib.util.spec_from_file_location("dataset_tools", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tool_call_is_bounded_and_path_redacted() -> None:
    call = {
        "schema_version": "omni-ar-tool-call/v1",
        "tool": "dataset_adapter",
        "call_id": "test-1",
        "arguments": {"action": "resolve", "dataset_ref": "vcc25@1", "artifact": "full_512g"},
    }
    result = MODULE.execute_tool_call(call, dataset_ref="vcc25@1", repository=ROOT)
    assert result["result"]["status"] == "ok"
    assert "path" not in str(result["result"])


def test_tool_cannot_switch_dataset() -> None:
    call = {
        "schema_version": "omni-ar-tool-call/v1",
        "tool": "dataset_adapter",
        "arguments": {"action": "describe", "dataset_ref": "cifar-10@1"},
    }
    try:
        MODULE.execute_tool_call(call, dataset_ref="vcc25@1", repository=ROOT)
    except ValueError as exc:
        assert "cannot switch" in str(exc)
    else:
        raise AssertionError("dataset switch must be rejected")


def test_tool_lists_predeclared_usage_modes() -> None:
    call = {
        "schema_version": "omni-ar-tool-call/v1",
        "tool": "dataset_adapter",
        "arguments": {"action": "list_modes", "dataset_ref": "vcc25@1"},
    }
    result = MODULE.execute_tool_call(call, dataset_ref="vcc25@1", repository=ROOT)
    assert "heldout_guide" in result["result"]["payload"]["usage_modes"]
