from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import json


@dataclass(frozen=True)
class SkillManifest:
    skill_id: str
    version: str
    kind: str
    capabilities: tuple[str, ...]
    input_schema: str
    output_schema: str
    entrypoint: str
    context_requirements: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.skill_id or "." not in self.skill_id:
            raise ValueError("skill_id must be namespaced, e.g. example.echo")
        if not self.capabilities:
            raise ValueError("skill must declare at least one capability")
        if not self.entrypoint:
            raise ValueError("skill entrypoint is required")
        if not self.input_schema.startswith("ai4ai/") or not self.output_schema.startswith("ai4ai/"):
            raise ValueError("skill schemas must use the ai4ai namespace")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"protocol_version": "ai4ai-skill/v1", "skill_id": self.skill_id, "version": self.version, "kind": self.kind, "capabilities": list(self.capabilities), "input_schema": self.input_schema, "output_schema": self.output_schema, "entrypoint": self.entrypoint, "context_requirements": list(self.context_requirements), "provenance": dict(self.provenance)}


@dataclass(frozen=True)
class ResearchState:
    task_id: str
    question: str
    evidence: tuple[Mapping[str, Any], ...] = ()
    open_hypotheses: tuple[Mapping[str, Any], ...] = ()
    rejected_hypotheses: tuple[Mapping[str, Any], ...] = ()
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillContext:
    research_state: ResearchState
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    skill_id: str
    status: str
    evidence: Mapping[str, Any]
    next_actions: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if self.status not in {"ok", "blocked", "rejected"}:
            raise ValueError("invalid skill result status")
        return {"protocol_version": "ai4ai-skill-result/v1", "skill_id": self.skill_id, "status": self.status, "evidence": dict(self.evidence), "next_actions": [dict(x) for x in self.next_actions], "provenance": dict(self.provenance)}


def validate_json_message(value: Any, expected_protocol: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("protocol_version") != expected_protocol:
        raise ValueError(f"expected protocol {expected_protocol}")
    return value


def encode_json_message(value: Mapping[str, Any], expected_protocol: str) -> str:
    validate_json_message(dict(value), expected_protocol)
    return json.dumps(value, sort_keys=True, ensure_ascii=False)

