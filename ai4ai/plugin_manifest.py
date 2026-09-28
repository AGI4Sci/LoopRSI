"""Manifest parsing and validation for task plugins."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

try:
    import yaml
except ModuleNotFoundError:  # Keep minimal controller runtimes dependency-free.
    yaml = None


class PluginManifestError(ValueError):
    pass


@dataclass(frozen=True)
class TaskPluginManifest:
    raw: Mapping[str, Any]
    path: Path

    @property
    def task_id(self) -> str:
        return str((self.raw.get("task") or {}).get("id") or (self.raw.get("task") or {}).get("name"))

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw.get(name) or {}
        if not isinstance(value, Mapping):
            raise PluginManifestError(f"manifest section {name!r} must be a mapping")
        return value

    def validate(self) -> None:
        if self.raw.get("schema_version") != "ai4ai-task-plugin/v1":
            raise PluginManifestError("unsupported task plugin manifest schema")
        if not self.task_id or self.task_id == "None":
            raise PluginManifestError("task.id is required")
        for name in ("skills", "dataset", "candidate", "evaluator", "policies", "runtime"):
            self.section(name)
        if "compatibility" in self.raw:
            self.section("compatibility")


def load_task_plugin_manifest(path: str | Path) -> TaskPluginManifest:
    manifest_path = Path(path).resolve()
    text = manifest_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) if yaml is not None else json.loads(text)
    if not isinstance(raw, Mapping):
        raise PluginManifestError("manifest must be a YAML mapping")
    manifest = TaskPluginManifest(raw, manifest_path)
    manifest.validate()
    return manifest
