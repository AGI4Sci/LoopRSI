"""Task-plugin protocols shared by the generic research core."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    checks: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptFragment:
    skill_id: str
    stage: str
    content: str
    priority: int = 0
    token_budget: int | None = None
    activation_reason: str = ""
    source: str = ""
    content_hash: str = ""


@dataclass(frozen=True)
class DatasetInventory:
    task_id: str
    splits: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    status: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class CandidateExecutionRequest:
    candidate_id: str
    command: tuple[str, ...]
    working_directory: str
    output_location: str
    environment: Mapping[str, str] = field(default_factory=dict)
    required_inputs: tuple[str, ...] = ()
    expected_artifacts: tuple[str, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateExecutionResult:
    candidate_id: str
    return_code: int
    stdout: str
    stderr: str
    execution_metadata: Mapping[str, Any] = field(default_factory=dict)
    produced_artifacts: tuple[str, ...] = ()
    artifact_manifest: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatorRequest:
    task_id: str
    prediction_artifact: str
    contract_artifact: str
    output_location: str
    config: Mapping[str, Any] = field(default_factory=dict)
    required_metrics: tuple[str, ...] = ()
    existing_evaluator_artifact: str | None = None
    existing_cell_eval_artifact: str | None = None


@dataclass(frozen=True)
class EvaluatorResult:
    task_id: str
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metric_completeness: bool = False
    validation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ResearchSkill(Protocol):
    """Task/domain skill consumed by the generic router."""

    @property
    def skill_id(self) -> str: ...

    def can_activate(self, context: Mapping[str, Any]) -> bool: ...

    def inject(self, context: Mapping[str, Any]) -> PromptFragment | None: ...

    def validate_proposal(self, proposal: Mapping[str, Any], context: Mapping[str, Any]) -> ValidationResult: ...


class DatasetAdapter(Protocol):
    def inventory(self) -> DatasetInventory: ...

    def validate_split(self, split: str) -> ValidationResult: ...

    def provenance(self) -> Mapping[str, Any]: ...


class CandidateAdapter(Protocol):
    def validate_package(self, package: Any) -> ValidationResult: ...

    def prepare_execution(self, package: Any, context: Mapping[str, Any]) -> CandidateExecutionRequest: ...

    def execute(self, request: CandidateExecutionRequest) -> CandidateExecutionResult: ...

    def validate_artifacts(self, result: CandidateExecutionResult) -> ValidationResult: ...


class Evaluator(Protocol):
    def validate_input(self, request: EvaluatorRequest) -> ValidationResult: ...

    def evaluate(self, request: EvaluatorRequest) -> EvaluatorResult: ...

    def normalize(self, result: EvaluatorResult) -> EvaluatorResult: ...

    def archive_record(self, result: EvaluatorResult) -> Mapping[str, Any]: ...
