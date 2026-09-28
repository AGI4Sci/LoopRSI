#!/usr/bin/env python3
"""Structured autonomous-research state primitives.

This module is intentionally project-agnostic. Task-specific scientific metrics
remain in Task Specs and evaluator outputs; this layer only stores evidence,
lineage, failure memory, and generic decisions used to build the next planner
context.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


STATUS_OPEN = "open"
STATUS_REJECTED = "rejected"
STATUS_PROMOTED = "promoted"
STATUS_BLOCKED = "blocked"


@dataclass
class InformationSource:
    source_id: str
    source_type: str
    path_or_uri: str
    provenance: str
    test_time_legal: bool
    uses_expression: bool = False
    leakage_risk: str = "none"
    coverage: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class FailureRecord:
    hypothesis_id: str
    mechanism_family: str
    evidence: dict[str, Any]
    failure_mode: str
    decision: str
    confidence: str = "medium"
    rejected_for_leakage: bool = False
    rejected_for_contract: bool = False


@dataclass
class ExperimentNode:
    node_id: str
    candidate_id: str
    hypothesis_id: str
    mechanism_family: str
    implementation: str
    result: dict[str, Any]
    decision: str
    parent: str | None = None
    relations: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ResearchState:
    schema_version: str = "omni-ar-research-state/v1"
    current_task: str = ""
    current_scientific_question: str = ""
    current_best_candidate: str = ""
    current_best_evidence: dict[str, Any] = field(default_factory=dict)
    open_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    rejected_hypotheses: list[FailureRecord] = field(default_factory=list)
    known_bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    available_information_sources: list[InformationSource] = field(default_factory=list)
    active_blockers: list[dict[str, Any]] = field(default_factory=list)
    next_research_action: str = ""
    experiment_graph: list[ExperimentNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchState":
        state = cls(
            schema_version=str(payload.get("schema_version", "omni-ar-research-state/v1")),
            current_task=str(payload.get("current_task", "")),
            current_scientific_question=str(payload.get("current_scientific_question", "")),
            current_best_candidate=str(payload.get("current_best_candidate", "")),
            current_best_evidence=dict(payload.get("current_best_evidence") or {}),
            open_hypotheses=list(payload.get("open_hypotheses") or []),
            known_bottlenecks=list(payload.get("known_bottlenecks") or []),
            active_blockers=list(payload.get("active_blockers") or []),
            next_research_action=str(payload.get("next_research_action", "")),
        )
        state.rejected_hypotheses = [FailureRecord(**item) for item in payload.get("rejected_hypotheses", [])]
        state.available_information_sources = [InformationSource(**item) for item in payload.get("available_information_sources", [])]
        state.experiment_graph = [ExperimentNode(**item) for item in payload.get("experiment_graph", [])]
        return state


def load_state(path: Path) -> ResearchState:
    return ResearchState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_state(state: ResearchState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def canonical_mechanism_family(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    tokens = set(text.split("_"))
    if tokens.intersection({"coefficient", "coeff"}) or {"conditional", "vae"}.issubset(tokens):
        return "target_to_coefficient_prediction"
    aliases = [
        ({"response", "interpolation"}, "response_space_transfer"),
        ({"manifold", "projection"}, "response_space_transfer"),
        ({"sparse", "selection"}, "response_basis_selection"),
        ({"causal", "program", "sem"}, "structural_gene_program"),
        ({"ranking", "encoder"}, "direct_target_to_delta_encoder"),
    ]
    for required, family in aliases:
        if required.issubset(tokens):
            return family
    return text


def is_duplicate_hypothesis(state: ResearchState, mechanism_family: str, hypothesis: str = "") -> tuple[bool, str]:
    family = canonical_mechanism_family(mechanism_family or hypothesis)
    failed = {canonical_mechanism_family(item.mechanism_family) for item in state.rejected_hypotheses}
    if family in failed:
        return True, f"mechanism_family_already_rejected:{family}"
    text = hypothesis.lower()
    if "validation-visible" in text or "validation expression" in text or "test expression" in text:
        return True, "proposal_mentions_leakage_prone_data_access"
    return False, ""


def build_planner_context(state: ResearchState) -> dict[str, Any]:
    return {
        "schema_version": "omni-ar-planner-context/v1",
        "current_task": state.current_task,
        "current_scientific_question": state.current_scientific_question,
        "current_best_candidate": state.current_best_candidate,
        "current_best_evidence": state.current_best_evidence,
        "open_hypotheses": state.open_hypotheses,
        "rejected_hypotheses": [asdict(item) for item in state.rejected_hypotheses],
        "known_bottlenecks": state.known_bottlenecks,
        "available_information_sources": [asdict(item) for item in state.available_information_sources],
        "active_blockers": state.active_blockers,
        "next_research_action": state.next_research_action,
        "experiment_graph_summary": [asdict(item) for item in state.experiment_graph],
    }


def ready_information_sources(
    state: ResearchState,
    *,
    required_coverage_key: str,
    required_coverage_value: str,
    exclude_source_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    excluded = set(exclude_source_ids)
    ready = []
    for source in state.available_information_sources:
        if source.source_id in excluded:
            continue
        if not source.test_time_legal or source.uses_expression:
            continue
        if source.leakage_risk not in {"none", "low"}:
            continue
        if source.coverage.get(required_coverage_key) != required_coverage_value:
            continue
        ready.append(asdict(source))
    return ready


def continuation_decision_from_information(
    state: ResearchState,
    *,
    required_coverage_key: str,
    required_coverage_value: str,
    exclude_source_ids: Iterable[str] = (),
) -> dict[str, Any]:
    ready = ready_information_sources(
        state,
        required_coverage_key=required_coverage_key,
        required_coverage_value=required_coverage_value,
        exclude_source_ids=exclude_source_ids,
    )
    if ready:
        return {
            "status": "READY_FOR_PLANNER",
            "ready_information_sources": ready,
            "reason": "new_legal_information_source_available",
        }
    return {
        "status": "SEEK_INFORMATION",
        "ready_information_sources": [],
        "reason": "no_new_legal_information_source_satisfies_required_coverage",
    }


def promotion_decision(
    *,
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    primary_metric: str,
    direction: str,
    min_delta: float,
    stable: bool,
    contract_valid: bool,
    no_leakage: bool,
    scientific_evidence: bool,
) -> dict[str, Any]:
    cand = float(candidate_metrics[primary_metric])
    base = float(baseline_metrics[primary_metric])
    delta = cand - base if direction == "maximize" else base - cand
    passed = delta >= min_delta and stable and contract_valid and no_leakage and scientific_evidence
    return {
        "status": "PROMOTE" if passed else "DO_NOT_PROMOTE",
        "primary_metric": primary_metric,
        "direction": direction,
        "delta": delta,
        "min_delta": min_delta,
        "stable": stable,
        "contract_valid": contract_valid,
        "no_leakage": no_leakage,
        "scientific_evidence": scientific_evidence,
    }


def stop_policy_decision(
    state: ResearchState,
    *,
    failed_distinct_family_threshold: int = 4,
    plateau: bool = False,
    new_information_exhausted: bool = False,
    resource_blocker: bool = False,
) -> dict[str, Any]:
    failed_families = {canonical_mechanism_family(item.mechanism_family) for item in state.rejected_hypotheses}
    repeated_family = len(state.rejected_hypotheses) > len(failed_families)
    scientific_blocker = bool(state.known_bottlenecks)
    should_stop = (
        resource_blocker
        or new_information_exhausted
        or (scientific_blocker and len(failed_families) >= failed_distinct_family_threshold)
        or (plateau and repeated_family)
    )
    reasons = []
    if resource_blocker:
        reasons.append("resource_blocker")
    if new_information_exhausted:
        reasons.append("new_information_exhausted")
    if scientific_blocker and len(failed_families) >= failed_distinct_family_threshold:
        reasons.append("scientific_blocker_with_multiple_failed_families")
    if plateau and repeated_family:
        reasons.append("plateau_with_repeated_failed_family")
    return {
        "status": "STOP_AND_SEEK_INFORMATION" if should_stop else "CONTINUE_RESEARCH_LOOP",
        "failed_distinct_mechanism_families": sorted(failed_families),
        "repeated_failed_family_detected": repeated_family,
        "reasons": reasons,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--planner-context-out", type=Path)
    args = parser.parse_args()
    state = load_state(args.state)
    payload = build_planner_context(state)
    if args.planner_context_out:
        args.planner_context_out.parent.mkdir(parents=True, exist_ok=True)
        args.planner_context_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
