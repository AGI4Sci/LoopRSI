#!/usr/bin/env python3
"""P2 rollout driver: generate >10k decision trajectories in parallel shards.

Runs the HierarchicalLoop in ``worker.mode=rollout`` across multiple seeds and
multiple ``loop_range`` slices, writing one JSONL decision shard per (seed,
slice).  Pure stdlib; no GPU, no network, no real training.

Each shard is written to ``out/rollout/decisions/<name>.jsonl`` by determining
the writer shard name from the seed/slice index.  A single invocation can be
parallelized by running many processes with different ``--slice-index`` /
``--worker`` indices and a shared ``--out``; every shard is independently
counted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        raise SystemExit(f"cannot load config {path!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="hierarchical loop rollout")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="output/rsi_step0/rollout")
    ap.add_argument("--target", type=int, default=10000, help="target decision count")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--num-slices", type=int, default=5)
    ap.add_argument("--num-seeds", type=int, default=4)
    ap.add_argument("--slice-index", type=int, default=0, help="which loop_range slice to run")
    ap.add_argument("--seed-offset", type=int, default=0, help="seed batch offset for this worker")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hier_loop import HierarchicalLoop
    from hier_loop.layers import resolve_loop_range
    from hier_loop.decisions import DecisionWriter, count_decisions
    from hier_loop.vcc25_worker import Vcc25Worker

    cfg = load_config(args.config)
    slices = _slices(args.num_slices)
    loop_slice = slices[args.slice_index % len(slices)]
    cfg["loop_range"] = {"start": loop_slice[0], "end": loop_slice[1]}
    cfg.setdefault("run", {})["rounds"] = args.rounds
    cfg.setdefault("worker", {})["mode"] = "rollout"
    cfg.setdefault("out", {})["dir"] = args.out
    cfg.setdefault("decisions", {})["dir"] = args.out

    total = 0
    for si in range(args.num_seeds):
        seed = args.seed_base + args.seed_offset + si
        cfg.setdefault("run", {})["seed"] = seed
        shard_name = f"seed{seed}_slice{args.slice_index}"
        writer = DecisionWriter(args.out, shard=shard_name)
        worker = Vcc25Worker(config=cfg)
        loop = HierarchicalLoop(
            cfg, worker,
            run_id=f"rollout_seed{seed}_slice{args.slice_index}",
            decision_writer=writer,
        )
        loop.run()
        total += writer.count

    stats = count_decisions(args.out)
    print(json.dumps({"slice": loop_slice, "shard_total": total, "stats": stats},
                     ensure_ascii=False, indent=2))
    return 0


def _slices(n: int):
    layers = ["L1", "L2", "L3", "L4", "L5"]
    if n <= 1:
        return [("L1", "L5")]
    out = []
    for i in range(n):
        start = layers[min(i, len(layers) - 1)]
        end = layers[min(i + 3, len(layers) - 1)]
        out.append((start, end))
    return out


if __name__ == "__main__":
    sys.exit(main())
