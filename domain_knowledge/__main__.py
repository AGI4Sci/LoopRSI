"""Command-line inspection for the VCC25 domain knowledge tree."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from adapters.vcc25 import get_adapter

from .cards import KnowledgeQuery, KnowledgeValidationError
from .render import render_cards
from .store import KnowledgeStore, assert_safe_knowledge


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "knowledge" / "vcc25"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect validated VCC25 domain knowledge")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate all cards and the fixed contract")

    query = commands.add_parser("query", help="query searchable cards")
    query.add_argument("--layer", choices=("L1", "L2", "L3", "L4", "L5"))
    query.add_argument("--text", default="")
    query.add_argument(
        "--asset-type",
        action="append",
        choices=("paper", "repository", "model"),
        default=[],
    )
    query.add_argument("--readiness", nargs="+", default=[])
    query.add_argument("--max-results", type=int, default=10)
    query.add_argument("--token-budget", type=int, default=1200)

    adapter = commands.add_parser("check-adapter", help="inspect declared model assets only")
    adapter.add_argument("--model-id", required=True)
    adapter.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="map one artifact role to a local path",
    )
    return parser


def _json(value: Mapping[str, Any], stream: Any = None) -> None:
    if stream is None:
        stream = sys.stdout
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=stream)


def _asset_locations(values: Sequence[str]) -> Mapping[str, str]:
    locations = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role or not path:
            raise ValueError(f"invalid --asset {value!r}; expected ROLE=PATH")
        if role in locations:
            raise ValueError(f"duplicate --asset role {role!r}")
        locations[role] = path
    return locations


def _run(args: argparse.Namespace) -> Mapping[str, Any]:
    store = KnowledgeStore.from_directory(args.root)
    if args.command == "validate":
        warnings = store.validate()
        cards = store.query(KnowledgeQuery(task_id="vcc25", max_results=1000))
        contract = store.evaluation_contract
        return {
            "status": "ok",
            "warnings": list(warnings),
            "card_count": len(cards),
            "evaluation_contract": {
                "id": contract.id,
                "task_id": contract.task_id,
                "gene_count": contract.gene_count,
                "metrics": list(contract.metrics),
            },
        }
    if args.command == "query":
        assert_safe_knowledge(
            {"text": args.text, "asset_types": args.asset_type, "readiness": args.readiness},
            "cli_query",
        )
        cards = store.query(
            KnowledgeQuery(
                task_id="vcc25",
                layer=args.layer,
                text=args.text,
                asset_types=tuple(args.asset_type),
                readiness=tuple(args.readiness),
                max_results=args.max_results,
            )
        )
        rendered = render_cards(cards, args.token_budget)
        return {
            "evaluation_contract_id": store.evaluation_contract.id,
            "card_ids": list(rendered.card_ids),
            "estimated_tokens": rendered.estimated_tokens,
            "content_hash": rendered.content_hash,
            "content": rendered.content,
        }
    if args.command == "check-adapter":
        assert_safe_knowledge(
            {"model_id": args.model_id, "assets": args.asset},
            "cli_adapter_check",
        )
        models = store.query(
            KnowledgeQuery(task_id="vcc25", asset_types=("model",), max_results=100)
        )
        try:
            model = next(card for card in models if card.id == args.model_id)
        except StopIteration as exc:
            raise ValueError(f"unknown model id: {args.model_id!r}") from exc
        adapter_id = model.get("execution_recipe", {}).get("adapter_id")
        if not adapter_id:
            raise ValueError(f"model {model.id!r} has no implemented adapter")
        check = get_adapter(adapter_id).check_assets(model, _asset_locations(args.asset))
        return {
            "model_id": model.id,
            "adapter_id": adapter_id,
            "status": check.status,
            "missing": list(check.missing),
            "evidence": check.evidence,
        }
    raise ValueError(f"unknown command: {args.command!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _json(_run(args))
    except (KnowledgeValidationError, ValueError, OSError) as exc:
        _json({"status": "error", "error": str(exc)}, stream=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
