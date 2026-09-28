#!/usr/bin/env python3
"""Dispatch stability finalization through the task-plugin policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai4ai.plugin_manifest import load_task_plugin_manifest
from ai4ai.plugin_registry import load_entrypoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--task-plugin", type=Path, default=None)
    parser.add_argument("--consecutive-local", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    snapshot = json.loads((root / "reproducibility_snapshot.json").read_text(encoding="utf-8"))
    task_id = args.task or snapshot.get("task")
    if not task_id:
        raise SystemExit("stability finalization requires --task or snapshot.task")
    plugin_path = args.task_plugin or (Path(__file__).resolve().parents[2] / "tasks" / task_id / "task_plugin.yaml")
    manifest = load_task_plugin_manifest(plugin_path)
    entrypoint = manifest.section("policies").get("stability")
    if not entrypoint:
        raise SystemExit(f"task plugin {task_id!r} does not declare a stability policy")
    policy = load_entrypoint(str(entrypoint))
    finalize = getattr(policy, "finalize", None)
    if not callable(finalize):
        raise SystemExit(f"stability policy {entrypoint!r} does not expose finalize")
    return int(finalize(root, args.consecutive_local))


if __name__ == "__main__":
    raise SystemExit(main())
