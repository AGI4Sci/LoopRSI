"""Loading and indexing of VCC25 evidence records.

Evidence is provenance for cards and experiments; it is intentionally not
returned by KnowledgeStore.query().
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cards import EvidenceRecord, KnowledgeValidationError


class EvidenceStore:
    def __init__(self, records: tuple[EvidenceRecord, ...]) -> None:
        self.records = records
        self._by_id = {record.id: record for record in records}

    @classmethod
    def from_directory(cls, root: Path) -> "EvidenceStore":
        directory = Path(root) / "evidence"
        if not directory.is_dir():
            return cls(())
        records = []
        for path in sorted(directory.glob("*.json")):
            try:
                value: Any = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise KnowledgeValidationError(f"{path}: invalid evidence JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise KnowledgeValidationError(f"{path}: evidence must be an object")
            records.append(EvidenceRecord.from_mapping(value, f"evidence:{path.stem}"))
        return cls(tuple(records))

    def get(self, record_id: str) -> EvidenceRecord | None:
        return self._by_id.get(record_id)
