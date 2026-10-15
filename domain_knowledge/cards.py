"""Immutable public types for the VCC25 knowledge store."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple


class KnowledgeValidationError(ValueError):
    """Raised when knowledge cannot safely satisfy the public contract."""


@dataclass(frozen=True)
class EvidenceRecord:
    """Structured validation/provenance record, not a searchable card."""

    id: str
    status: str
    authority: str
    data: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], fallback_id: str) -> "EvidenceRecord":
        return cls(
            id=str(value.get("id", fallback_id)),
            status=str(value.get("status", "unknown")),
            authority=str(value.get("authority", "unknown")),
            data=_freeze(dict(value)),
        )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class KnowledgeCard:
    id: str
    schema_version: str
    asset_type: str
    title: str
    summary_plain: str
    tags: Tuple[str, ...]
    layers: Tuple[str, ...]
    sources: Tuple[Mapping[str, Any], ...]
    relations: Tuple[str, ...]
    review_status: str
    review_basis: str
    data: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KnowledgeCard":
        return cls(
            id=value["id"],
            schema_version=value["schema_version"],
            asset_type=value["asset_type"],
            title=value["title"],
            summary_plain=value["summary_plain"],
            tags=tuple(value["tags"]),
            layers=tuple(value["layers"]),
            sources=tuple(_freeze(source) for source in value["sources"]),
            relations=tuple(value["relations"]),
            review_status=value["review_status"],
            review_basis=value["review_basis"],
            data=_freeze(dict(value)),
        )

    @property
    def execution_readiness(self) -> Optional[str]:
        value = self.data.get("execution_readiness")
        return value if isinstance(value, str) else None

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass(frozen=True)
class KnowledgeQuery:
    task_id: str
    layer: Optional[str] = None
    text: str = ""
    asset_types: Tuple[str, ...] = ()
    readiness: Tuple[str, ...] = ()
    max_results: int = 5
    approved_only: bool = True

    def for_layer(self, layer: str, **changes: Any) -> "KnowledgeQuery":
        """Return a copy with a layer-specific retrieval constraint."""
        if layer not in {"L1", "L2", "L3", "L4", "L5"}:
            raise KnowledgeValidationError(f"unknown layer {layer!r}")
        values = {
            "task_id": self.task_id, "layer": layer, "text": self.text,
            "asset_types": self.asset_types, "readiness": self.readiness,
            "max_results": self.max_results, "approved_only": self.approved_only,
        }
        values.update(changes)
        return KnowledgeQuery(**values)


@dataclass(frozen=True)
class VCC25EvaluationContract:
    id: str
    schema_version: str
    task_id: str
    gene_count: int
    gene_order_source: str
    metrics: Tuple[str, ...]
    selection_policy: str
    leakage_policy: str
    data: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VCC25EvaluationContract":
        return cls(
            id=value["id"],
            schema_version=value["schema_version"],
            task_id=value["task_id"],
            gene_count=value["gene_count"],
            gene_order_source=value["gene_order_source"],
            metrics=tuple(value["metrics"]),
            selection_policy=value["selection_policy"],
            leakage_policy=value["leakage_policy"],
            data=_freeze(dict(value)),
        )

    def assert_comparable(self, contract_ids: Sequence[str]) -> None:
        mismatches = tuple(value for value in contract_ids if value != self.id)
        if mismatches:
            raise KnowledgeValidationError(
                "results are not comparable: expected evaluation contract "
                f"{self.id!r}, received {mismatches!r}"
            )
