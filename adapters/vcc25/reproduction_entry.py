"""Bounded upstream paper reproduction preflight entrypoint.

This command never reads final-test expression data. It reports a blocked
preflight when an upstream checkout is absent; a method-specific adapter can
later replace this runner without changing the controller contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source_repository)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        status = "blocked"
        reason = "upstream source checkout is not present in the independent experiment directory"
    else:
        status = "partial"
        reason = "source checkout detected; method-specific training/prediction adapter still required"
    record = {
        "schema_version": "vcc25.paper-reproduction-preflight/v1",
        "method": args.method,
        "action": args.action,
        "status": status,
        "reason": reason,
        "source_repository": str(source),
        "test_expression_read": False,
    }
    (output / "preflight.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
