"""Structured decision backends for knowledge-driven hierarchy layers."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .knowledge_bridge import KnowledgeInjection


class LayerDecisionError(ValueError):
    """Raised when a layer backend returns an invalid decision."""


@dataclass(frozen=True)
class LayerDecisionRequest:
    layer: str
    round: int
    step: int
    knowledge: KnowledgeInjection
    context: Mapping[str, Any]
    prior_decisions: tuple[Mapping[str, Any], ...]
    prompt: str


class LayerDecisionBackend(Protocol):
    def decide(self, request: LayerDecisionRequest) -> Mapping[str, Any]: ...


_TYPE_BY_LAYER = {
    "L1": "direction",
    "L2": "problem",
    "L3": "hypothesis",
    "L4": "mechanism",
    "L5": "coding_request",
}

_FIELDS_BY_TYPE = {
    "direction": {"direction", "rationale"},
    "problem": {"problem", "comparison_target"},
    "hypothesis": {"hypothesis", "falsification_criterion"},
    "mechanism": {"mechanism", "expected_effect"},
}

_CODING_FIELDS = {
    "allowed_paths",
    "hypothesis",
    "activation_diagnostics",
    "train_command",
    "validation_command",
    "expected_artifacts",
}


def validate_layer_decision(layer: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LayerDecisionError("layer decision must be an object")
    decision = json.loads(json.dumps(value, ensure_ascii=False))
    expected_type = _TYPE_BY_LAYER.get(layer)
    if decision.get("layer") != layer:
        raise LayerDecisionError(f"decision layer must be {layer}")
    if decision.get("decision_type") != expected_type:
        raise LayerDecisionError(f"decision_type for {layer} must be {expected_type}")
    if expected_type in _FIELDS_BY_TYPE:
        required = _FIELDS_BY_TYPE[expected_type]
        allowed = {"layer", "decision_type", *required}
        if set(decision) != allowed:
            raise LayerDecisionError(f"{layer} decision fields must be {sorted(allowed)}")
        for field in required:
            if not isinstance(decision.get(field), str) or not decision[field].strip():
                raise LayerDecisionError(f"{field} must be a non-empty string")
        return decision

    allowed = {"layer", "decision_type", "coding_request"}
    if set(decision) != allowed:
        raise LayerDecisionError(f"L5 decision fields must be {sorted(allowed)}")
    coding = decision.get("coding_request")
    if not isinstance(coding, Mapping) or set(coding) != _CODING_FIELDS:
        raise LayerDecisionError(f"coding_request fields must be {sorted(_CODING_FIELDS)}")
    for field in ("allowed_paths", "activation_diagnostics", "train_command", "validation_command", "expected_artifacts"):
        values = coding.get(field)
        if not isinstance(values, list) or not values or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise LayerDecisionError(f"coding_request.{field} must be a non-empty string list")
    if not isinstance(coding.get("hypothesis"), str) or not coding["hypothesis"].strip():
        raise LayerDecisionError("coding_request.hypothesis must be a non-empty string")
    return decision


def build_layer_prompt(
    layer: str,
    round_: int,
    step: int,
    knowledge: KnowledgeInjection,
    context: Mapping[str, Any],
    prior_decisions: tuple[Mapping[str, Any], ...],
) -> str:
    payload = {
        "instruction": (
            f"Return one JSON decision for {layer}. Use only the supplied knowledge and prior "
            "decisions. Do not invent evaluation scores or claim experimental results."
        ),
        "layer": layer,
        "round": round_,
        "step": step,
        "required_decision_type": _TYPE_BY_LAYER[layer],
        "knowledge": {
            "skill_id": knowledge.skill_id,
            "card_ids": list(knowledge.card_ids),
            "content": knowledge.content,
            "content_hash": knowledge.content_hash,
            "source": knowledge.source,
        },
        "prior_decisions": list(prior_decisions),
        "context": dict(context),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str)


class RecordingDecisionBackend:
    """Deterministic no-GPU backend that retains every complete request prompt."""

    def __init__(self) -> None:
        self.requests: list[LayerDecisionRequest] = []
        self.prompts: list[str] = []

    def decide(self, request: LayerDecisionRequest) -> Mapping[str, Any]:
        self.requests.append(request)
        self.prompts.append(request.prompt)
        decisions: dict[str, dict[str, Any]] = {
            "L1": {
                "layer": "L1",
                "decision_type": "direction",
                "direction": "compare train-only biological priors for unseen H1 perturbations",
                "rationale": "the knowledge cards identify prior choice as a bounded, testable lever",
            },
            "L2": {
                "layer": "L2",
                "decision_type": "problem",
                "problem": "improve unseen-target perturbation response prediction without test-expression access",
                "comparison_target": "same-contract Lingshu full-validation baseline",
            },
            "L3": {
                "layer": "L3",
                "decision_type": "hypothesis",
                "hypothesis": "a train-only target prior can improve validation PCC over the matched baseline",
                "falsification_criterion": "reject if the frozen-seed full-validation delta is not positive",
            },
            "L4": {
                "layer": "L4",
                "decision_type": "mechanism",
                "mechanism": "inject a bounded train-derived target prior into the residual prediction path",
                "expected_effect": "increase perturbation-specific signal while preserving gene order and split isolation",
            },
            "L5": {
                "layer": "L5",
                "decision_type": "coding_request",
                "coding_request": {
                    "allowed_paths": [
                        "crpm",
                        "scripts",
                        "tests",
                    ],
                    "hypothesis": "a train-only target prior improves same-contract validation PCC",
                    "activation_diagnostics": [
                        "prior_coverage",
                        "prediction_delta_norm",
                        "test_expression_not_read",
                    ],
                    "train_command": [
                        "python3",
                        "tasks/vcc25/implementation/official_h1_autonomous_research_loop.py",
                        "--output-dir",
                        "{smoke_output_dir}",
                        "--n-train-targets",
                        "8",
                        "--n-val-targets",
                        "4",
                    ],
                    "validation_command": [
                        "python3",
                        "tasks/vcc25/implementation/official_h1_autonomous_research_loop.py",
                        "--output-dir",
                        "{validation_output_dir}",
                    ],
                    "expected_artifacts": [
                        "proposal.json",
                        "lineage_summary.json",
                        "activation_diagnostics.json",
                    ],
                },
            },
        }
        return decisions[request.layer]


def _parse_final_json_object(stdout: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[Mapping[str, Any]] = []
    for index, character in enumerate(stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if stdout[index + end :].strip() or not isinstance(value, Mapping):
            continue
        candidates.append(value)
    if not candidates:
        raise LayerDecisionError("Codex did not return a final JSON object")
    return candidates[-1]


class CodexJsonDecisionBackend:
    """Read-only Codex backend for L1-L4/L5 structured planning decisions."""

    def __init__(
        self,
        schema_path: str | Path,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.schema_path = Path(schema_path).resolve()
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def _schema_for_layer(self, layer: str) -> Path:
        suffix = ".schema.json"
        if not self.schema_path.name.endswith(suffix):
            raise LayerDecisionError(f"Codex schema must end with {suffix}: {self.schema_path}")
        basename = self.schema_path.name[: -len(suffix)]
        layer_schema = self.schema_path.with_name(f"{basename}.{layer}{suffix}")
        if not layer_schema.is_file():
            raise LayerDecisionError(f"missing Codex output schema for {layer}: {layer_schema}")
        return layer_schema

    def decide(self, request: LayerDecisionRequest) -> Mapping[str, Any]:
        schema_path = self._schema_for_layer(request.layer)
        command = [
            "codex",
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(schema_path),
            "-",
        ]
        completed = self.runner(
            command,
            input=request.prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise LayerDecisionError(
                "Codex decision failed: " + str(completed.stderr or completed.stdout)[-2000:]
            )
        return _parse_final_json_object(completed.stdout)
