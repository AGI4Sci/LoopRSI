"""JSON loading, validation, and deterministic lookup."""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .cards import (
    KnowledgeCard,
    KnowledgeQuery,
    KnowledgeValidationError,
    VCC25EvaluationContract,
)


CARD_DIRECTORIES = {
    "papers": ("paper", "kb:paper:"),
    "repositories": ("repository", "kb:repo:"),
    "models": ("model", "kb:model:"),
}
LAYERS = frozenset({"L1", "L2", "L3", "L4", "L5"})
ASSET_TYPES = frozenset(value[0] for value in CARD_DIRECTORIES.values())
READINESS = frozenset(
    {"ready", "adapter_required", "finetune_required", "train_required", "reference_only"}
)
SHARED_FIELDS = (
    "id",
    "schema_version",
    "asset_type",
    "title",
    "summary_plain",
    "tags",
    "layers",
    "sources",
    "relations",
    "review_status",
    "review_basis",
)
MODEL_FIELDS = (
    "usage_mode",
    "execution_readiness",
    "artifacts",
    "licenses",
    "io_contract",
    "resource_profile",
    "vcc25_compatibility",
    "execution_recipe",
    "smoke_evidence",
)
CONTRACT_FIELDS = (
    "id",
    "schema_version",
    "task_id",
    "gene_count",
    "gene_order_source",
    "metrics",
    "selection_policy",
    "leakage_policy",
)
RESTRICTED_IDENTIFIERS = (
    "real_de",
    "test_expression",
    "test_answers",
    "hidden_test",
    "vcc_test_validation_h5ad",
    "test_h5ad",
    "competition_test",
)
READINESS_ORDER = {
    "ready": 0,
    "adapter_required": 1,
    "finetune_required": 2,
    "train_required": 3,
    "reference_only": 4,
    None: 5,
}


def _error(path: Path, field: str, message: str) -> KnowledgeValidationError:
    return KnowledgeValidationError(f"{path.as_posix()}: field {field}: {message}")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeValidationError(f"{path.as_posix()}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(path, "<root>", "must be an object")
    return value


def _require_fields(value: Mapping[str, Any], fields: Iterable[str], path: Path) -> None:
    for field in fields:
        if field not in value:
            raise _error(path, field, "is required")


def _walk_restricted(value: Any, path: Path, field_path: str = "<root>") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_restricted(str(key), path, f"{field_path}.<key>")
            _walk_restricted(item, path, f"{field_path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_restricted(item, path, f"{field_path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for identifier in RESTRICTED_IDENTIFIERS:
            if identifier in lowered:
                raise _error(path, field_path, f"restricted identifier {identifier!r}")


def assert_safe_knowledge(value: Any, context: str = "input") -> None:
    """Reject restricted identifiers in cards, prompts, or adapter inputs."""
    _walk_restricted(value, Path(context))


def _validate_string_list(value: Any, path: Path, field: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise _error(path, field, "must be an array of non-empty strings")


def _validate_card(value: Mapping[str, Any], path: Path, expected_type: str, prefix: str) -> None:
    _require_fields(value, SHARED_FIELDS, path)
    if value["asset_type"] != expected_type or expected_type not in ASSET_TYPES:
        raise _error(path, "asset_type", f"must be {expected_type!r}")
    if not isinstance(value["id"], str) or not value["id"].startswith(prefix):
        raise _error(path, "id", f"must start with {prefix!r}")
    for field in ("schema_version", "title", "summary_plain", "review_status", "review_basis"):
        if not isinstance(value[field], str) or not value[field]:
            raise _error(path, field, "must be a non-empty string")
    _validate_string_list(value["tags"], path, "tags")
    _validate_string_list(value["layers"], path, "layers")
    if not value["layers"] or any(layer not in LAYERS for layer in value["layers"]):
        raise _error(path, "layers", "must contain only L1 through L5")
    _validate_string_list(value["relations"], path, "relations")
    if not isinstance(value["sources"], list) or not value["sources"]:
        raise _error(path, "sources", "must be a non-empty array")
    if not all(isinstance(source, dict) and isinstance(source.get("url"), str) for source in value["sources"]):
        raise _error(path, "sources", "each source must contain a URL")
    if value["review_status"] not in {"approved", "draft", "rejected"}:
        raise _error(path, "review_status", "has an unsupported value")
    if expected_type == "model":
        _validate_model(value, path)
    _walk_restricted(value, path)


def _validate_model(value: Mapping[str, Any], path: Path) -> None:
    _require_fields(value, MODEL_FIELDS, path)
    if value["execution_readiness"] not in READINESS:
        raise _error(path, "execution_readiness", "has an unsupported value")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list):
        raise _error(path, "artifacts", "must be an array")
    required = {"role", "uri", "format", "access", "integrity"}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise _error(path, f"artifacts[{index}]", f"must contain {sorted(required)!r}")
    for field in ("licenses", "io_contract", "resource_profile", "vcc25_compatibility", "execution_recipe"):
        if not isinstance(value[field], dict):
            raise _error(path, field, "must be an object")
    if not isinstance(value["smoke_evidence"], list):
        raise _error(path, "smoke_evidence", "must be an array")
    if value["execution_readiness"] == "ready" and not value["smoke_evidence"]:
        raise _error(path, "smoke_evidence", "is required when execution_readiness is ready")


def _validate_contract(value: Mapping[str, Any], path: Path) -> None:
    _require_fields(value, CONTRACT_FIELDS, path)
    if value["task_id"] != "vcc25":
        raise _error(path, "task_id", "must be 'vcc25'")
    if not isinstance(value["gene_count"], int) or value["gene_count"] <= 0:
        raise _error(path, "gene_count", "must be a positive integer")
    _validate_string_list(value["metrics"], path, "metrics")


def _tokens(text: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_+-]+|[\u4e00-\u9fff]+", text.lower()))


class KnowledgeStore:
    """An immutable view of a validated on-disk knowledge tree."""

    def __init__(
        self,
        cards: Sequence[KnowledgeCard],
        evaluation_contract: VCC25EvaluationContract,
    ) -> None:
        self._cards = tuple(cards)
        self.evaluation_contract = evaluation_contract

    @classmethod
    def from_directory(cls, root: Path) -> "KnowledgeStore":
        root = Path(root)
        contract_path = root / "evaluation_contract.json"
        contract_value = _load_json(contract_path)
        _validate_contract(contract_value, contract_path.relative_to(root))

        cards: List[KnowledgeCard] = []
        seen: Dict[str, Path] = {}
        raw_paths: Dict[str, Path] = {}
        for directory, (asset_type, prefix) in CARD_DIRECTORIES.items():
            card_dir = root / directory
            if not card_dir.is_dir():
                raise KnowledgeValidationError(f"{directory}: required card directory is missing")
            for path in sorted(card_dir.glob("*.json")):
                relative_path = path.relative_to(root)
                value = _load_json(path)
                _validate_card(value, relative_path, asset_type, prefix)
                card_id = value["id"]
                if card_id in seen:
                    raise KnowledgeValidationError(
                        f"duplicate id {card_id!r}: {seen[card_id].as_posix()} and "
                        f"{relative_path.as_posix()}"
                    )
                seen[card_id] = relative_path
                raw_paths[card_id] = relative_path
                cards.append(KnowledgeCard.from_mapping(value))

        known_ids = set(seen)
        for card in cards:
            for relation in card.relations:
                if relation not in known_ids:
                    raise _error(raw_paths[card.id], "relations", f"unknown card id {relation!r}")

        return cls(cards, VCC25EvaluationContract.from_mapping(contract_value))

    def validate(self) -> Tuple[str, ...]:
        return ()

    def query(self, query: KnowledgeQuery) -> Tuple[KnowledgeCard, ...]:
        if query.layer is not None and query.layer not in LAYERS:
            raise KnowledgeValidationError(f"unknown layer {query.layer!r}; expected L1 through L5")
        if query.max_results < 1:
            raise KnowledgeValidationError("max_results must be positive")
        unknown_types = set(query.asset_types) - ASSET_TYPES
        if unknown_types:
            raise KnowledgeValidationError(f"unknown asset types: {sorted(unknown_types)!r}")
        unknown_readiness = set(query.readiness) - READINESS
        if unknown_readiness:
            raise KnowledgeValidationError(f"unknown readiness values: {sorted(unknown_readiness)!r}")
        if query.task_id != self.evaluation_contract.task_id:
            return ()

        query_tokens = set(_tokens(query.text))
        ranked = []
        for card in self._cards:
            if query.approved_only and card.review_status != "approved":
                continue
            if query.asset_types and card.asset_type not in query.asset_types:
                continue
            if query.layer is not None and query.layer not in card.layers:
                continue
            if query.readiness and card.execution_readiness not in query.readiness:
                continue
            haystack = " ".join(
                (card.title, card.summary_plain, " ".join(card.tags))
            ).lower()
            overlap = sum(1 for token in query_tokens if token in haystack)
            if query_tokens and overlap == 0:
                continue
            layer_rank = 0 if query.layer is not None and query.layer in card.layers else 1
            ranked.append(
                (
                    layer_rank,
                    -overlap,
                    READINESS_ORDER[card.execution_readiness],
                    card.id,
                    card,
                )
            )
        ranked.sort(key=lambda item: item[:-1])
        return tuple(item[-1] for item in ranked[: query.max_results])
