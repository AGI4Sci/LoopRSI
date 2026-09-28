#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(path for path in args.root.rglob("*.py") if "__pycache__" not in path.parts)
    for path in files:
        py_compile.compile(str(path), doraise=True)
    print(f"compiled {len(files)} python file(s) under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
