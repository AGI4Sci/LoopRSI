"""Manifest-backed plugin registry; implementation loading is deliberately lazy."""
from __future__ import annotations

from pathlib import Path
from importlib import import_module
from typing import Any

from .plugin_manifest import TaskPluginManifest, load_task_plugin_manifest


class TaskPluginRegistry:
    def __init__(self) -> None:
        self._manifests: dict[str, TaskPluginManifest] = {}

    def register_manifest(self, manifest: TaskPluginManifest) -> None:
        if manifest.task_id in self._manifests:
            raise ValueError(f"duplicate task plugin: {manifest.task_id}")
        self._manifests[manifest.task_id] = manifest

    def load(self, path: str | Path) -> TaskPluginManifest:
        manifest = load_task_plugin_manifest(path)
        self.register_manifest(manifest)
        return manifest

    def get(self, task_id: str) -> TaskPluginManifest:
        return self._manifests[task_id]

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._manifests))


def load_entrypoint(entrypoint: str) -> Any:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid plugin entrypoint: {entrypoint!r}")
    module = import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"plugin entrypoint does not exist: {entrypoint}") from exc


def load_skill_bundle(manifest: TaskPluginManifest) -> list[Any]:
    entrypoints = manifest.section("skills").get("entrypoints") or []
    if not isinstance(entrypoints, list):
        raise ValueError("skills.entrypoints must be a list")
    skills = []
    for entrypoint in entrypoints:
        implementation = load_entrypoint(str(entrypoint))
        skills.append(implementation())
    return skills
