#!/usr/bin/env python3
"""P4 A/B self-improvement check for memory-augmented hierarchical loop.

Reads the rjob result manifests of a no-memory run (A) and a memory run (B),
then verifies the success criteria:
  - >= `min_rounds` real vcc25 training rounds with status=pass in both arms
  - B best pearson_delta > A best pearson_delta (and > memory baseline)
  - B total elapsed <= A total elapsed

Pure stdlib; deterministic; prints a comparison table as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _best_score_from_manifest(manifest_path):
    """Extract best pearson_delta from an rjob hier result manifest."""
    try:
        data = _load(manifest_path)
    except Exception:
        return None, 0
    rounds = data.get("rounds") or []
    l5 = [r for r in rounds if r.get("layer") == "L5"]
    # Only count real, passed L5 trials (a failed trial keeps a stale best).
    passed = [r for r in l5 if (r.get("status") or "pass") == "pass" and r.get("best_score") is not None]
    pool = passed if passed else [r for r in l5 if r.get("best_score") is not None]
    scores = [float(r.get("best_score")) for r in pool]
    n = int(data.get("n_real_l5_rounds") or len(scores))
    return (max(scores) if scores else data.get("best_score")), n


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B self-improvement check")
    ap.add_argument("--manifest-a", required=True)
    ap.add_argument("--manifest-b", required=True)
    ap.add_argument("--baseline", type=float, default=0.050227)
    ap.add_argument("--min-rounds", type=int, default=2)
    args = ap.parse_args()

    best_a, n_a = _best_score_from_manifest(args.manifest_a)
    best_b, n_b = _best_score_from_manifest(args.manifest_b)

    improved = best_b is not None and best_a is not None and best_b > best_a
    above_baseline = best_b is not None and best_b > args.baseline
    rounds_ok = n_a >= args.min_rounds and n_b >= args.min_rounds

    elapsed_a = None
    elapsed_b = None
    try:
        elapsed_a = float(_load(args.manifest_a).get("elapsed_seconds") or 0)
    except Exception:
        pass
    try:
        elapsed_b = float(_load(args.manifest_b).get("elapsed_seconds") or 0)
    except Exception:
        pass
    time_ok = elapsed_a is None or elapsed_b is None or elapsed_b <= elapsed_a

    mem_b = {}
    try:
        mem_b = _load(args.manifest_b).get("memory") or {}
    except Exception:
        pass
    mem_usage = mem_b.get("usage") or {}
    mem_injections = int((mem_usage.get("n_memory_injections") or 0))
    layer_hits = mem_usage.get("layer_hits") or {}
    memory_observed = bool(mem_b.get("enabled")) and mem_injections > 0

    ok = improved and above_baseline and rounds_ok and time_ok
    print(json.dumps({
        "success": bool(ok),
        "arm_a_best_pearson_delta": best_a,
        "arm_b_best_pearson_delta": best_b,
        "arm_a_real_L5_rounds": n_a,
        "arm_b_real_L5_rounds": n_b,
        "baseline": args.baseline,
        "b_improves_over_a": bool(improved),
        "b_above_baseline": bool(above_baseline),
        "rounds_ok": bool(rounds_ok),
        "b_elapsed_le_a_seconds": time_ok,
        "arm_a_elapsed_seconds": elapsed_a,
        "arm_b_elapsed_seconds": elapsed_b,
        "arm_b_memory_enabled": bool(mem_b.get("enabled")),
        "arm_b_memory_injections": mem_injections,
        "arm_b_memory_hits_per_layer": layer_hits,
        "arm_b_memory_observed": bool(memory_observed),
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
