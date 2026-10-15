"""Keep the top official GO edges needed by GEARS for each target."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=21)
    args = parser.parse_args()
    if args.top_k < 1 or args.source.resolve() == args.output.resolve():
        raise ValueError("invalid GO compaction arguments")

    import pandas as pd

    partial = []
    source_rows = 0
    for chunk in pd.read_csv(args.source, chunksize=500_000):
        if list(chunk.columns) != ["source", "target", "importance"]:
            raise ValueError("unexpected official GO columns")
        source_rows += len(chunk)
        ordered = chunk.sort_values("importance", ascending=False, kind="stable")
        partial.append(ordered[ordered.groupby("target", sort=False).cumcount() < args.top_k])
    candidates = pd.concat(partial, ignore_index=True)
    ordered = candidates.sort_values("importance", ascending=False, kind="stable")
    compact = ordered[ordered.groupby("target", sort=False).cumcount() < args.top_k]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(args.output, index=False)
    print(json.dumps({
        "source_rows": source_rows,
        "compact_rows": len(compact),
        "target_count": compact["target"].nunique(),
        "top_k": args.top_k,
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
