#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from task_contract import load_task_adapter, load_task_spec, resolved_workspace
from tasks.adapter_contract import merge_research_capabilities


def read_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        return f"[missing] {path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]


def read_json(path: Path, max_chars: int) -> object:
    if not path.exists():
        return {"missing": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars]
        return {"truncated_text": text}
    return json.loads(text)


def split_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item for item in raw.split(":") if item]


def expand(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            files.extend(Path(x) for x in matches)
        else:
            files.append(Path(pattern))
    return sorted(dict.fromkeys(files))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    env = os.environ
    max_chars = int(env.get("CONTEXT_MAX_CHARS_PER_FILE", "30000"))
    text_files = expand(split_list(env.get("CONTEXT_TEXT_FILES")))
    json_files = expand(split_list(env.get("CONTEXT_JSON_FILES")))
    code_files = expand(split_list(env.get("CONTEXT_CODE_GLOBS")))
    task_spec_path = Path(env["TASK_SPEC"]).resolve() if env.get("TASK_SPEC") else None
    task_spec = load_task_spec(task_spec_path) if task_spec_path else None

    context = {
        "round": args.round,
        "paths": {
            "root_dir": env.get("ROOT_DIR"),
            "project_root": env.get("PROJECT_ROOT"),
            "implementation_dir": env.get("IMPLEMENTATION_DIR"),
            "controller_dir": env.get("CONTROLLER_DIR"),
            "task_spec": str(task_spec_path) if task_spec_path else None,
        },
        "text_files": {str(path): read_text(path, max_chars) for path in text_files},
        "json_files": {str(path): read_json(path, max_chars) for path in json_files},
        "code_files": {str(path): read_text(path, max_chars) for path in code_files},
        "limits": {
            "max_trials_per_round": int(env.get("MAX_TRIALS_PER_ROUND", "12")),
            "max_code_changes_per_round": int(env.get("MAX_CODE_CHANGES_PER_ROUND", "8")),
            "rjob_gpu_limit": int(env.get("RJOB_GPU_LIMIT", "2")),
            "rjob_gpu_per_trial": int(env.get("RJOB_GPU_PER_TRIAL", "1")),
        },
    }
    if task_spec is not None and task_spec_path is not None:
        project_root, implementation_dir = resolved_workspace(task_spec, task_spec_path)
        context["task_spec"] = task_spec
        context["task_contract_paths"] = {
            "project_root": str(project_root),
            "implementation_dir": str(implementation_dir),
        }
        dataset_ref = str((task_spec.get("data") or {}).get("dataset_ref", ""))
        if dataset_ref:
            repository = task_spec_path.parents[2]
            if str(repository) not in os.sys.path:
                os.sys.path.insert(0, str(repository))
            from datasets.registry import DatasetRegistry

            registry = DatasetRegistry(repository / "datasets")
            dataset_adapter = registry.load_adapter(dataset_ref)
            task_adapter = load_task_adapter(task_spec, task_spec_path)
            capabilities = merge_research_capabilities(
                task_adapter.initialization_capabilities(task_spec),
                dataset_adapter.dispatch("capabilities"),
            )
            capabilities["operations"] = capabilities.get("operation_capabilities")
            skill_path = registry.pack_root(dataset_ref) / "SKILL.md"
            context["dataset"] = {
                "ref": dataset_ref,
                "description": dataset_adapter.dispatch("describe"),
                "capabilities": capabilities,
                "splits": dataset_adapter.dispatch("list_splits"),
                "skill": read_text(skill_path, max_chars),
                "rule": "Use Dataset Adapter capabilities; do not invent physical paths or modify raw data.",
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
