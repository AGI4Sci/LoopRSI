#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai4ai.contracts import ResearchState
from ai4ai.plugin_registry import load_skill_bundle
from ai4ai.plugin_manifest import load_task_plugin_manifest
from ai4ai.router import SkillRouter
from ai4ai.stage_hook import SkillStageHook
from skills.example.evidence import EvidenceSummarySkill


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def evidence_from(context: dict) -> tuple[dict, ...]:
    records: list[dict] = []
    for path, value in (context.get("json_files") or {}).items():
        if isinstance(value, dict):
            records.append({"id": path, **value})
    return tuple(records)


def skills_for_task(task_name: str) -> list[object]:
    manifest_path = ROOT / "tasks" / task_name / "task_plugin.yaml"
    if manifest_path.is_file():
        return load_skill_bundle(load_task_plugin_manifest(manifest_path))
    return [EvidenceSummarySkill()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8"))
    task_name = ((context.get("task_spec") or {}).get("task") or {}).get("name", "unknown")
    state = ResearchState(task_id=task_name, question=str(context.get("scientific_question") or f"Improve {task_name}"), evidence=evidence_from(context))
    skills = skills_for_task(task_name)
    router = SkillRouter(skills)
    hook = SkillStageHook(router)
    injections = []
    for skill in skills:
        state, qa = hook.inject("qa_to_idea_card", skill.manifest.skill_id, state)
        injections.append({"stage": qa.stage, **qa.result.to_dict()})
        state, proposal = hook.inject("idea_card_to_heuresis", skill.manifest.skill_id, state)
        injections.append({"stage": proposal.stage, **proposal.result.to_dict()})
    original_hash = digest(context)
    augmented = dict(context)
    augmented["skill_injections"] = injections
    augmented["skill_injection_provenance"] = {
        "original_context_sha256": original_hash,
        "augmented_context_sha256": digest(augmented),
        "mode": "additive",
    }
    args.context.write_text(json.dumps(augmented, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
