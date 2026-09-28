"""Decision trajectories for the Hierarchical Looped RSI Researcher (P2).

A single decision is one ``rsi.decision.v1`` record derived from an emitted
loop message of kind ``step`` / ``transfer`` / ``exit``.  Each record carries
enough provenance to trace back to its ``(round, layer, step)`` and its parent
message / candidate, plus the LLM planning/reasoning text and the observed
outcome score / reward.  Decisions are appended to sharded JSONL files under
``out/rollout/decisions/<shard>.jsonl``.

Pure stdlib, deterministic, Python 3.10+.
"""

from __future__ import annotations

import hashlib
import json
import os
import time as _time
from typing import Any, Dict, List, Optional

DECISION_SCHEMA = "rsi.decision.v1"
DECISION_KINDS = ("step", "transfer", "exit")


def context_fingerprint(context: Dict[str, Any]) -> str:
    """Stable fingerprint of a decision's task/context vector.

    Used as the retrieval key for the per-loop memory index.  Only a small,
    ordered subset of scalars are hashed so that semantically-identical
    decisions (same task, layer, gates, baseline, active layers) collide.
    """
    active = list(context.get("active_layers") or [])
    basis = "|".join(
        [
            str(context.get("task_id", "")),
            str(context.get("metric", "")),
            str(context.get("baseline", "")),
            ",".join(active),
        ]
    )
    return "fp_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def context_text(context: Dict[str, Any]) -> str:
    """Deterministic text rendering of a decision's task/context vector.

    This is the *semantic* retrieval key: it is embedded and compared by cosine
    similarity.  It renders the same canonical keys that
    :func:`context_fingerprint` hashes (plus the step position), so that
    semantically-identical contexts render identically and semantic rewrites
    still land near each other in embedding space.  The deterministic
    fingerprint remains a dedupe / provenance field only.
    """
    active = list(context.get("active_layers") or [])
    parts = {
        "task_id": str(context.get("task_id", "")),
        "metric": str(context.get("metric", "")),
        "baseline": str(context.get("baseline", "")),
        "active_layers": ",".join(active),
        "layer": str(context.get("layer", "")),
        "round": str(context.get("round", "")),
        "step": str(context.get("step", "")),
    }
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def synthesize_context_text(record: Dict[str, Any]) -> str:
    """Best-effort reconstruction of a decision's context text.

    Used to backfill legacy decision shards that predate the ``context_text``
    field so they can participate in semantic retrieval.
    """
    layer = record.get("layer") or record.get("from_layer") or ""
    parts = {
        "task_id": str(record.get("task_id") or ""),
        "metric": str(record.get("metric") or "pearson_delta"),
        "baseline": "",
        "active_layers": layer,
        "layer": layer,
        "round": str(record.get("round") or ""),
        "step": str(record.get("step") or ""),
    }
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def _reasoning_from(payload: Dict[str, Any], fallback: str) -> str:
    """Extract the LLM planning/reasoning text from a payload, if present."""
    action = payload.get("action") or {}
    for key in ("reasoning", "plan", "prompt", "detail"):
        val = action.get(key)
        if isinstance(val, str) and val.strip():
            return val
    ctx = payload.get("context") or {}
    for key in ("reasoning", "plan"):
        val = ctx.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return fallback


def make_decision(
    msg: Dict[str, Any],
    run_id: str,
    reward: Optional[float] = None,
    metric: str = "pearson_delta",
    card_id: Optional[str] = None,
    metric_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one ``rsi.decision.v1`` record from an emitted loop message dict.

    ``msg`` is the dict returned by ``HierarchicalLoop._emit`` (i.e. an on-disk
    ``rsi.msg.v1`` record).
    """
    payload = msg.get("payload") or {}
    context = payload.get("context") or {}
    action = payload.get("action") or {}
    outcome = payload.get("outcome") or {}
    metrics = outcome.get("metrics") or {}

    layer = msg.get("from_layer", "")
    decision_id = "dec_" + hashlib.sha256(
        f"{run_id}|{msg.get('msg_id')}|{msg.get('hop')}".encode("utf-8")
    ).hexdigest()[:16]
    parent_ref = msg.get("parent_msg_id") or action.get("parent_ref")

    step = action.get("step")
    if step is None:
        step = context.get("step")

    outcome_score: Optional[float] = None
    for key in ("score", "best_score"):
        if outcome.get(key) is not None:
            outcome_score = outcome.get(key)
            break
    if outcome_score is None:
        score = msg.get("score")
        if score is None:
            score = metrics.get(metric)
        outcome_score = score

    reasoning = _reasoning_from(payload, action.get("detail") or f"{layer} step")
    memory_used = bool(context.get("memory_used") or action.get("memory_used"))
    memory_hits = context.get("memory_hits") or action.get("memory_hits")

    return {
        "schema": DECISION_SCHEMA,
        "decision_id": decision_id,
        "run_id": run_id,
        "msg_id": msg.get("msg_id"),
        "parent_ref": parent_ref,
        "from_layer": layer,
        "to_layer": msg.get("to_layer"),
        "kind": msg.get("kind"),
        "route": msg.get("route"),
        "round": msg.get("round"),
        "layer": layer,
        "step": step,
        "operator": action.get("operator"),
        "variant": action.get("variant"),
        "seed": action.get("seed") or context.get("seed"),
        "metric_source": metric_source,
        "parent_candidate_id": action.get("parent_candidate_id"),
        "reasoning": reasoning,
        "context_fingerprint": context_fingerprint(context),
        "context_text": context_text(context),
        "outcome_score": outcome_score,
        "metric": metric,
        "reward": reward,
        "card_id": card_id,
        "task_id": context.get("task_id"),
        "memory_used": memory_used,
        "memory_hits": memory_hits,
        "created_at": msg.get("created_at") or _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


class DecisionWriter:
    """Append-only sharded writer for decision trajectories."""

    def __init__(self, out_dir: str, shard: str = "shard_0000"):
        self.out_dir = out_dir
        self.shard = shard
        self.count = 0

    def root(self) -> str:
        return os.path.join(self.out_dir, "decisions")

    def path(self) -> str:
        return os.path.join(self.root(), f"{self.shard}.jsonl")

    def write(self, record: Dict[str, Any]) -> Dict[str, Any]:
        os.makedirs(self.root(), exist_ok=True)
        with open(self.path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
        return record


def read_decisions(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def count_decisions(root: str) -> Dict[str, Any]:
    """Count decision records across all shards under ``root/decisions``."""
    shard_dir = os.path.join(root, "decisions")
    total = 0
    shards: List[Dict[str, Any]] = []
    if os.path.isdir(shard_dir):
        for path in sorted(os.listdir(shard_dir)):
            if not path.endswith(".jsonl"):
                continue
            recs = read_decisions(os.path.join(shard_dir, path))
            total += len(recs)
            shards.append({"shard": path, "count": len(recs)})
    return {"total": total, "shards": shards}


def validate_decision(record: dict) -> List[str]:
    """Return a list of validation errors for one decision record."""
    errors: List[str] = []
    if record.get("schema") != DECISION_SCHEMA:
        errors.append("schema != rsi.decision.v1")
    if record.get("kind") not in DECISION_KINDS:
        errors.append(f"kind not in {DECISION_KINDS}: {record.get('kind')!r}")
    for field in ("round", "layer"):
        if record.get(field) is None:
            errors.append(f"missing {field}")
    if not record.get("reasoning"):
        errors.append("missing reasoning/planning text")
    if not record.get("context_fingerprint"):
        errors.append("missing context_fingerprint")
    return errors
