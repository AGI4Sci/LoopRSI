"""Validated, deterministic domain knowledge for the VCC25 research loop."""

from .cards import (
    EvidenceRecord,
    KnowledgeCard,
    KnowledgeQuery,
    KnowledgeValidationError,
    VCC25EvaluationContract,
)
from .store import KnowledgeStore
from .query import KnowledgeQueryService
from .evidence import EvidenceStore

__all__ = [
    "KnowledgeCard",
    "EvidenceRecord",
    "KnowledgeQuery",
    "KnowledgeStore",
    "KnowledgeQueryService",
    "EvidenceStore",
    "KnowledgeValidationError",
    "VCC25EvaluationContract",
]
