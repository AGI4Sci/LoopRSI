#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--suggestion", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--trial-results", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    if args.ledger.exists():
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    else:
        ledger = {"version": 1, "rounds": []}
    entry = {
        "round": args.round,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "round_dir": str(args.round_dir),
        "context": str(args.context),
        "suggestion": str(args.suggestion),
        "validation": str(args.validation),
        "trial_results": str(args.trial_results),
        "summary_json": str(args.summary_json),
    }
    ledger["rounds"] = [x for x in ledger.get("rounds", []) if x.get("round") != args.round]
    ledger["rounds"].append(entry)
    ledger["rounds"].sort(key=lambda x: x["round"])
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

