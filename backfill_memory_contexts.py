#!/usr/bin/env python3
"""Backfill per-layer memory stores with decision context text (P3/R).

Legacy memory stores and decision shards predate the semantic (cosine)
retrieval layer and carry no ``context`` / ``context_text``, so they cannot
participate in embedding retrieval.  This script reads the decision shards
under ``<decisions_root>/decisions/*.jsonl``, re-derives a deterministic
context text per record (``decisions.synthesize_context_text``), migrates
existing store entries whose context is empty, appends missing records
(deduped by ``context_fingerprint+op+variant+seed``) and persists a per-layer
MemoryStore, exactly like the one ``rjob_hier_vcc25.py inner`` loads at
startup.  Pure stdlib; run on any node with the project checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _entry_key(e) -> str:
    import hashlib
    payload = "|".join([
        e.layer or "",
        e.context_fingerprint or "",
        e.context or "",
        e.op or "",
        e.variant or "",
        str(e.seed or ""),
        str(e.reward if e.reward is not None else ""),
        str(e.outcome_score if e.outcome_score is not None else ""),
        str(e.round or ""),
        str(e.step or ""),
        e.kind or "",
        e.metric_source or "",
        (e.reasoning or "")[:120],
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", default="output/rsi_step0/rollout")
    ap.add_argument("--store", default="output/rsi_step0/memory")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hier_loop import layers as L
    from hier_loop.decisions import read_decisions, synthesize_context_text
    from hier_loop.memory import MemoryEntry, MemoryStore

    store = MemoryStore(args.store)
    merged = {}
    for layer in L.layer_ids():
        merged[layer] = list(store.load(layer))

    shard_dir = os.path.join(args.decisions, "decisions")
    names = sorted(n for n in os.listdir(shard_dir) if n.endswith(".jsonl")) if os.path.isdir(shard_dir) else []
    n_records = 0
    for name in names:
        for record in read_decisions(os.path.join(shard_dir, name)):
            n_records += 1
            layer = record.get("layer")
            if layer not in merged:
                continue
            entry = MemoryEntry.from_decision(record)
            if not entry.context:
                entry.context = synthesize_context_text(record)
            merged[layer].append(entry)

    # Dedupe in order while migrating legacy entries without context.
    counts = {}
    for layer, entries in merged.items():
        seen = set()
        out = []
        for e in entries:
            if not e.context:
                e.context = synthesize_context_text({
                    "layer": e.layer or layer,
                    "task_id": "vcc25",
                    "metric": "pearson_delta",
                    "round": e.round,
                    "step": e.step,
                })
            key = _entry_key(e)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        merged[layer] = out
        counts[layer] = len(out)

    os.makedirs(args.store, exist_ok=True)
    for layer, entries in merged.items():
        with open(os.path.join(args.store, f"{layer}.jsonl"), "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")

    print(json.dumps({
        "shards": len(names),
        "records": n_records,
        "store": args.store,
        "layers": counts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
