"""Validated, deterministic domain knowledge for the VCC25 research loop."""

from .cards import (
    KnowledgeCard,
    KnowledgeQuery,
    KnowledgeValidationError,
    VCC25EvaluationContract,
)
from .store import KnowledgeStore

__all__ = [
    "KnowledgeCard",
    "KnowledgeQuery",
    "KnowledgeStore",
    "KnowledgeValidationError",
    "VCC25EvaluationContract",
]
