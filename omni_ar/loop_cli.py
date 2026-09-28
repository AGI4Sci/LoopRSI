from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loop import ResearchLoop


def main() -> int:
    parser = argparse.ArgumentParser(prog="loopctl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("datasets")
    loading = sub.add_parser("dataset-loading")
    loading.add_argument("--dataset", required=True)
    loading.add_argument("--mode", default="default")
    loading.add_argument("--inspect-limit", type=int, default=5)
    loading.add_argument("--deep", action="store_true")
    loading.add_argument("--prepare", action="store_true")
    design = sub.add_parser("design-code")
    design.add_argument("--task", type=Path, required=True)
    design.add_argument("--proposal", type=Path, required=True)
    design.add_argument("--dataset-mode")
    args = parser.parse_args()
    service = ResearchLoop()
    if args.command == "datasets":
        result = service.datasets()
    elif args.command == "dataset-loading":
        result = service.dataset_loading(
            args.dataset, mode=args.mode, inspect_limit=args.inspect_limit,
            deep_validate=args.deep, prepare=args.prepare,
        )
    else:
        proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
        result = service.design_code(args.task, proposal, dataset_mode=args.dataset_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
