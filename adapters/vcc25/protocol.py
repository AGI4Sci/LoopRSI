"""Common lifecycle types for VCC25 paper reproduction adapters.

Adapters are deliberately declarative: they build allowlisted execution
requests, while the controller owns process execution and evidence writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


REPRODUCTION_STATUSES = ("reproduced", "partial", "blocked", "reference_only")
REPRODUCTION_ACTIONS = (
    "check_assets",
    "prepare",
    "train",
    "predict",
    "convert_output",
)


@dataclass(frozen=True)
class ReproductionContext:
    """Inputs shared by every paper adapter lifecycle."""

    paper_id: str
    model_id: str
    code_id: str
    working_directory: str
    output_location: str
    inputs: Mapping[str, str]
    resources: Mapping[str, int] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReproductionEvidence:
    """Controller-facing evidence record; no final-test data is accepted."""

    paper_id: str
    model_id: str
    code_id: str
    status: str
    code_revision: str | None
    model_revision: str | None
    commands: Sequence[Sequence[str]]
    data_scope: Mapping[str, Any]
    metrics: Mapping[str, float]
    limitations: Sequence[str]
    test_expression_read: bool = False

    def __post_init__(self) -> None:
        if self.status not in REPRODUCTION_STATUSES:
            raise ValueError(f"unsupported reproduction status: {self.status!r}")
        if self.test_expression_read:
            raise ValueError("paper reproduction evidence cannot read final test expression")


class VCC25ReproductionAdapter(Protocol):
    """Minimal interface every future paper adapter must expose."""

    adapter_id: str
    allowed_actions: Sequence[str]

    def check_assets(self, model: Mapping[str, Any], locations: Mapping[str, str]) -> Any: ...

    def prepare(self, action: str, model: Mapping[str, Any], context: Mapping[str, Any]) -> Any: ...

    def evidence_template(self, context: ReproductionContext, status: str, **kwargs: Any) -> ReproductionEvidence: ...
