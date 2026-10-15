"""Stable task-agnostic query facade over KnowledgeStore."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .cards import KnowledgeCard, KnowledgeQuery
from .store import KnowledgeStore


class KnowledgeQueryService:
    """Resolve a task's knowledge root and execute bounded multi-card queries.

    The service deliberately keeps the storage backend replaceable: callers
    depend on ``query`` and not on JSON paths or a particular index.
    """

    def __init__(self, roots: Mapping[str, str | Path] | None = None) -> None:
        self._roots = {str(task): Path(root) for task, root in (roots or {}).items()}
        self._stores: dict[str, KnowledgeStore] = {}

    def register(self, task_id: str, root: str | Path) -> None:
        self._roots[str(task_id)] = Path(root)
        self._stores.pop(str(task_id), None)

    def store(self, task_id: str) -> KnowledgeStore:
        task_id = str(task_id)
        if task_id not in self._roots:
            raise KeyError(f"knowledge root is not registered for task {task_id!r}")
        if task_id not in self._stores:
            self._stores[task_id] = KnowledgeStore.from_directory(self._roots[task_id])
        return self._stores[task_id]

    def query(self, query: KnowledgeQuery) -> tuple[KnowledgeCard, ...]:
        return self.store(query.task_id).query(query)

    def query_many(self, queries: Sequence[KnowledgeQuery]) -> tuple[KnowledgeCard, ...]:
        """Run multiple bounded queries and de-duplicate cards by stable ID."""
        result: list[KnowledgeCard] = []
        seen: set[str] = set()
        for query in queries:
            for card in self.query(query):
                if card.id not in seen:
                    result.append(card)
                    seen.add(card.id)
        return tuple(result)
