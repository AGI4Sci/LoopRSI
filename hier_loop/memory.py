"""Per-loop memory models for the Hierarchical Looped RSI Researcher (P3).

Builds, persists and queries one memory index per layer from the P2 decision
trajectories.  Retrieval ("retrieval layer", mandatory) maps a decision context
fingerprint to the top-k historical high-reward decisions for few-shot prompt
injection.  An optional "learning layer" (a lightweight trained reranker) may
refine the ordering; when its training preconditions are not met the retrieval
layer is used as-is and P3 never blocks the loop.

Every index is persisted as ``out/memory/<layer>.jsonl`` (append + rebuild on
capacity) and retrieval failure degrades gracefully to the no-memory policy.
Pure stdlib, Python 3.10+.
"""

from __future__ import annotations

import json
import os
import re
import time as _time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import layers as L
from .decisions import read_decisions, synthesize_context_text
from . import embedding as emb

MEMORY_SCHEMA = "rsi.memory.v1"
MEMORY_KEYS = (
    "context_fingerprint",
    "op",
    "variant",
    "parent_ref",
    "reasoning",
    "outcome_score",
    "reward",
    "card_id",
    "seed",
    "metric_source",
)


@dataclass
class MemoryEntry:
    context_fingerprint: str
    context: str = ""
    op: Optional[str] = None
    variant: Optional[str] = None
    parent_ref: Optional[str] = None
    reasoning: str = ""
    outcome_score: Optional[float] = None
    reward: Optional[float] = None
    card_id: Optional[str] = None
    seed: Optional[int] = None
    metric_source: Optional[str] = None
    layer: Optional[str] = None
    round: Optional[int] = None
    step: Optional[int] = None
    kind: Optional[str] = None
    created_at: Optional[str] = None
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": MEMORY_SCHEMA,
            "layer": self.layer,
            "context_fingerprint": self.context_fingerprint,
            "context": self.context,
            "op": self.op,
            "variant": self.variant,
            "parent_ref": self.parent_ref,
            "reasoning": self.reasoning,
            "outcome_score": self.outcome_score,
            "reward": self.reward,
            "card_id": self.card_id,
            "seed": self.seed,
            "metric_source": self.metric_source,
            "round": self.round,
            "step": self.step,
            "kind": self.kind,
            "created_at": self.created_at,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            layer=raw.get("layer"),
            context_fingerprint=raw.get("context_fingerprint") or "",
            context=raw.get("context") or "",
            op=raw.get("op"),
            variant=raw.get("variant"),
            parent_ref=raw.get("parent_ref"),
            reasoning=raw.get("reasoning") or "",
            outcome_score=raw.get("outcome_score"),
            reward=raw.get("reward"),
            card_id=raw.get("card_id"),
            seed=raw.get("seed"),
            metric_source=raw.get("metric_source"),
            round=raw.get("round"),
            step=raw.get("step"),
            kind=raw.get("kind"),
            created_at=raw.get("created_at"),
            embedding=raw.get("embedding"),
        )

    @classmethod
    def from_decision(cls, decision: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            layer=decision.get("layer"),
            context_fingerprint=decision.get("context_fingerprint") or "",
            context=(
                decision.get("context_text")
                or (synthesize_context_text(decision) if decision.get("layer") else "")
            ),
            op=decision.get("operator") or decision.get("op"),
            variant=decision.get("variant"),
            parent_ref=decision.get("parent_ref") or decision.get("parent_candidate_id"),
            reasoning=decision.get("reasoning") or "",
            outcome_score=decision.get("outcome_score"),
            reward=decision.get("reward"),
            card_id=decision.get("card_id"),
            seed=decision.get("seed"),
            metric_source=decision.get("metric_source"),
            round=decision.get("round"),
            step=decision.get("step"),
            kind=decision.get("kind"),
            created_at=decision.get("created_at"),
            embedding=decision.get("embedding"),
        )


class MemoryIndex:
    """In-memory per-layer decision index with top-k fingerprint retrieval."""

    def __init__(self, layer: str, top_k: int = 5, embedder: Optional[Any] = None,
                 min_similarity: float = 0.75, similarity: str = "cosine"):
        L.validate_layer_id(layer)
        self.layer = layer
        self.top_k = max(1, int(top_k))
        self.entries: List[MemoryEntry] = []
        self.embedder = embedder
        self.min_similarity = float(min_similarity)
        self.similarity = similarity or "cosine"
        self._vec_cache: Dict[str, Any] = {}

    def add(self, entry: MemoryEntry) -> None:
        if entry.layer is None:
            entry.layer = self.layer
        self.entries.append(entry)
        self._vec_cache.clear()

    def add_decision(self, decision: Dict[str, Any]) -> None:
        self.add(MemoryEntry.from_decision(decision))

    def __len__(self) -> int:
        return len(self.entries)

    # -- retrieval -----------------------------------------------------------
    def retrieve(
        self,
        context_text: Optional[str] = None,
        context_fingerprint: Optional[str] = None,
        top_k: Optional[int] = None,
        min_reward: Optional[float] = None,
    ) -> List[MemoryEntry]:
        """Return top-k best matches for a query, best first.

        When an embedder and a non-empty ``context_text`` are available the
        lookup is *semantic*: entries are scored by cosine similarity of their
        ``context`` and the best score must reach ``min_similarity`` to count as
        a hit (otherwise an empty list is returned, degrading to no-memory).
        Ties break by ``reward`` then recency.  If no embedder or text is
        available it falls back to the legacy exact ``context_fingerprint``
        match, so existing stored indexes stay usable.
        """
        k = self.top_k if top_k is None else max(1, int(top_k))
        pool = self.entries
        if min_reward is not None:
            pool = [e for e in pool if e.reward is not None and float(e.reward) >= float(min_reward)]
        if context_text and (context_text or "").strip() and self.embedder is not None:
            return self._semantic_retrieve(context_text, pool, k)
        # Legacy exact-fingerprint path.
        best = []
        for e in pool:
            if context_fingerprint and e.context_fingerprint != context_fingerprint:
                continue
            best.append(e)
        rank_key = lambda e: (   # noqa: E731
            float(e.reward if e.reward is not None else (e.outcome_score if e.outcome_score is not None else 0.0)),
            e.created_at or "",
        )
        best.sort(key=rank_key, reverse=True)
        return best[:k]

    def _semantic_retrieve(self, query_text: str, pool: List[MemoryEntry], k: int) -> List[MemoryEntry]:
        """Rank ``pool`` by cosine similarity of an embedded query context."""
        candidates = [e for e in pool if (e.context or "").strip()]
        if not candidates:
            return []
        embedder = self.embedder
        scored: List[Tuple[MemoryEntry, float]] = []
        if getattr(embedder, "name", "") == "tfidf":
            embedder.fit([e.context for e in candidates])
            qv = embedder.embed(query_text)
            for e in candidates:
                v = self._vec_cache.get(e.context)
                if v is None:
                    v = embedder.embed(e.context)
                    self._vec_cache[e.context] = v
                scored.append((e, emb.cosine(qv, v)))
        else:
            qv = self._vec_cache.get(("__q__", query_text))
            if qv is None:
                qv = embedder.embed(query_text)
                self._vec_cache[("__q__", query_text)] = qv
            for e in candidates:
                v = self._vec_cache.get(e.context)
                if v is None:
                    v = embedder.embed(e.context)
                    self._vec_cache[e.context] = v
                scored.append((e, emb.cosine(qv, v)))
        best = max((s for _, s in scored), default=0.0)
        if best < self.min_similarity:
            return []
        scored.sort(key=lambda item: (
            item[1],
            float(item[0].reward if item[0].reward is not None
                   else (item[0].outcome_score if item[0].outcome_score is not None else 0.0)),
            item[0].created_at or "",
        ), reverse=True)
        return [e for e, _ in scored[:k]]

    def hit_rate(self, fingerprints: List[str]) -> float:
        if not fingerprints:
            return 0.0
        hits = sum(1 for fp in fingerprints if any(e.context_fingerprint == fp for e in self.entries))
        return hits / len(fingerprints)


class MemoryStore:
    """Persistent per-layer memory: ``out/memory/<layer>.jsonl``."""

    def __init__(self, root: str):
        self.root = root

    def path(self, layer: str) -> str:
        return os.path.join(self.root, f"{layer}.jsonl")

    def load(self, layer: str) -> List[MemoryEntry]:
        if not os.path.isfile(self.path(layer)):
            return []
        out: List[MemoryEntry] = []
        with open(self.path(layer), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(MemoryEntry.from_dict(json.loads(line)))
                except Exception:
                    continue
        return out

    def append(self, entry: MemoryEntry) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.path(entry.layer or "L5"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def rebuild(self, layer: str, decisions_root: str) -> int:
        """Rebuild one layer's persisted memory from all rollout decision shards."""
        idx = MemoryIndex(layer)
        shard_dir = os.path.join(decisions_root, "decisions")
        n = 0
        if os.path.isdir(shard_dir):
            for name in sorted(os.listdir(shard_dir)):
                if not name.endswith(".jsonl"):
                    continue
                for rec in read_decisions(os.path.join(shard_dir, name)):
                    if rec.get("layer") == layer:
                        idx.add_decision(rec)
                        n += 1
        path = self.path(layer)
        os.makedirs(self.root, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for e in idx.entries:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
        return n


# --------------------------------------------------------------------------
# Lightweight learned reranker (optional "learning layer"; never blocks P3).
# --------------------------------------------------------------------------
class LearnedReranker:
    """A tiny linear reranker trained on decision trajectories.

    Training conditions (>= ``min_samples`` labelled decisions with rewards)
    must be met, otherwise ``can_train`` is False and the caller uses the
    retrieval layer only.  ``score(entry, fingerprint)`` returns a float used
    to re-order retrieval candidates; it degrades to the fingerprint match.
    """

    def __init__(self, min_samples: int = 100):
        self.min_samples = int(min_samples)
        self._weights: List[float] = []
        self._op_index: Dict[str, float] = {}
        self.trained = False

    def can_train(self, decisions: List[Dict[str, Any]]) -> bool:
        labelled = [d for d in decisions if d.get("reward") is not None]
        return len(labelled) >= self.min_samples

    def train(self, decisions: List[Dict[str, Any]]) -> bool:
        """OLS-style estimate of per-op reward bias (+ intercept)."""
        labelled = [d for d in decisions if d.get("reward") is not None]
        if len(labelled) < self.min_samples:
            self.trained = False
            return False
        ops = sorted({(d.get("operator") or d.get("op") or "draft") for d in labelled})
        self._op_index = {op: i for i, op in enumerate(ops)}
        n = len(ops)
        X: List[List[float]] = []
        y: List[float] = []
        for d in labelled:
            row = [1.0] + [0.0] * n
            op = d.get("operator") or d.get("op") or "draft"
            row[1 + self._op_index.get(op, 0)] = 1.0
            X.append(row)
            y.append(float(d["reward"]))
        # (X^T X)^-1 X^T y (Gaussian elimination; n features are tiny).
        k = n + 1
        A = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(k)] for i in range(k)]
        b = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(k)]
        w = _solve_linear(A, b, k)
        self._weights = w
        self.trained = True
        return True

    def score(self, entry: MemoryEntry, context_fingerprint: str) -> float:
        if entry.context_fingerprint != context_fingerprint:
            return 0.0
        if not self.trained:
            return float(entry.reward if entry.reward is not None else (entry.outcome_score or 0.0))
        intercept = self._weights[0] if self._weights else 0.0
        op = entry.op or "draft"
        bias = self._weights[1 + self._op_index.get(op, 0)] if self._weights and self._op_index else 0.0
        return intercept + bias


def _solve_linear(A: List[List[float]], b: List[float], n: int) -> List[float]:
    """Gaussian elimination for the small normal equations (n <= ~10)."""
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = next((r for r in range(col, n) if abs(M[r][col]) > 1e-12), None)
        if piv is None:
            return [0.0] * n
        M[col], M[piv] = M[piv], M[col]
        v = M[col][col]
        M[col] = [x / v for x in M[col]]
        for r in range(n):
            if r != col and abs(M[r][col]) > 1e-12:
                f = M[r][col]
                M[r] = [M[r][j] - f * M[col][j] for j in range(n + 1)]
    return [M[i][n] for i in range(n)]


# --------------------------------------------------------------------------
# Wiring helpers.
# --------------------------------------------------------------------------
def _resolve_model_dir(model: Optional[str]) -> str:
    if not model:
        return ""
    if os.path.isabs(model):
        return model
    base = os.path.abspath(os.path.join(os.getcwd(), model))
    return base if os.path.isdir(base) else str(model)


def _build_embedder(backend: Optional[str], model: Optional[str]):
    """Instantiate the configured retrieval embedding backend.

    Returns ``(embedder, actual_backend)`` where ``actual_backend`` is one of
    ``"tfidf"``, ``"dense"`` or ``"dense-unavailable"``.  A configured dense
    backend that cannot be loaded degrades to *no memory* (empty retrieval)
    rather than crashing the loop; tfidf is the always-available default.
    """
    backend = (backend or "tfidf").lower()
    if backend == "dense":
        try:
            return emb.OnnxEmbedder(_resolve_model_dir(model)), "dense"
        except Exception as exc:  # pragma: no cover
            print(f"[memory] dense embedder unavailable ({exc}); "
                  f"retrieval degrades to no-memory")
            return None, "dense-unavailable"
    return emb.TfidfEmbedder(), "tfidf"


def _prior_context_text(pr: Dict[str, Any], layer: str) -> str:
    parts = {
        "task_id": pr.get("task_id") or "vcc25",
        "metric": pr.get("metric") or "pearson_delta",
        "baseline": "",
        "active_layers": layer,
        "layer": layer,
        "round": "",
        "step": "",
    }
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def build_memory_manager(config: Dict[str, Any], decisions_root: str) -> "MemoryManager":
    mem_cfg = config.get("memory") or {}
    out_root = (mem_cfg.get("out") or config.get("out") or {}).get("dir", "output/rsi_step0/memory")
    ret_cfg = mem_cfg.get("retrieval") or {}
    emb_cfg = ret_cfg.get("embedding") or {}
    embedder, actual_backend = _build_embedder(emb_cfg.get("backend"), emb_cfg.get("model"))
    manager = MemoryManager(
        store=MemoryStore(out_root),
        decisions_root=decisions_root,
        top_k=int(ret_cfg.get("top_k", 5)),
        min_reward=ret_cfg.get("min_reward"),
        learn_enabled=bool((mem_cfg.get("learn") or {}).get("enabled", False)),
        learn_min_samples=int((mem_cfg.get("learn") or {}).get("min_samples", 100)),
        prefer_metric_source=ret_cfg.get("prefer_metric_source"),
        embedder=embedder,
        min_similarity=float(ret_cfg.get("min_similarity", 0.75)),
        similarity=ret_cfg.get("similarity", "cosine"),
        embedding_backend=actual_backend,
    )
    for pr in (mem_cfg.get("prior_real_results") or []):
        layer = pr.get("layer") or "L5"
        idx = manager.ensure(layer)
        if any(e.metric_source == "real" and e.seed == pr.get("seed") and e.variant == pr.get("variant")
               for e in idx.entries):
            continue
        idx.add(MemoryEntry(
            layer=layer,
            context_fingerprint=pr.get("context_fingerprint") or "",
            context=pr.get("context_text") or _prior_context_text(pr, layer),
            op=pr.get("op") or "implement",
            variant=pr.get("variant"),
            seed=pr.get("seed"),
            outcome_score=pr.get("outcome_score"),
            reward=pr.get("reward"),
            metric_source="real",
            reasoning=pr.get("reasoning") or "prior real result",
            kind="prior",
        ))
    if mem_cfg.get("prior_real_results"):
        manager.persist()
    return manager


class MemoryManager:
    """Owns per-layer indexes and the optional learned reranker."""

    def __init__(
        self,
        store: MemoryStore,
        decisions_root: str,
        top_k: int = 5,
        min_reward: Optional[float] = None,
        learn_enabled: bool = False,
        learn_min_samples: int = 100,
        prefer_metric_source: Optional[str] = None,
        embedder: Optional[Any] = None,
        min_similarity: float = 0.75,
        similarity: str = "cosine",
        embedding_backend: Optional[str] = None,
    ):
        self.store = store
        self.decisions_root = decisions_root
        self.top_k = top_k
        self.min_reward = min_reward
        self.learn_enabled = learn_enabled
        self.learn_min_samples = learn_min_samples
        self.prefer_metric_source = prefer_metric_source
        self.embedder = embedder
        self.min_similarity = float(min_similarity)
        self.similarity = similarity or "cosine"
        self.embedding_backend = embedding_backend
        self._semantic_queries = 0
        self._semantic_hits = 0
        self._fp_queries = 0
        self._fp_hits = 0
        self.indexes: Dict[str, MemoryIndex] = {}
        self.reranker: Optional[LearnedReranker] = None

    def ensure(self, layer: str) -> MemoryIndex:
        if layer not in self.indexes:
            idx = MemoryIndex(
                layer, top_k=self.top_k, embedder=self.embedder,
                min_similarity=self.min_similarity, similarity=self.similarity,
            )
            for e in self.store.load(layer):
                idx.add(e)
            self.indexes[layer] = idx
        return self.indexes[layer]

    def build_from_decisions(self) -> Dict[str, int]:
        """Rebuild every layer index from the rollout decision shards."""
        built: Dict[str, int] = {}
        decisions: List[Dict[str, Any]] = []
        shard_dir = os.path.join(self.decisions_root, "decisions")
        if os.path.isdir(shard_dir):
            for name in sorted(os.listdir(shard_dir)):
                if name.endswith(".jsonl"):
                    decisions.extend(read_decisions(os.path.join(shard_dir, name)))
        for layer in L.layer_ids():
            idx = MemoryIndex(
                layer, top_k=self.top_k, embedder=self.embedder,
                min_similarity=self.min_similarity, similarity=self.similarity,
            )
            n = 0
            for d in decisions:
                if d.get("layer") == layer:
                    idx.add_decision(d)
                    n += 1
            self.indexes[layer] = idx
            built[layer] = n
        if self.learn_enabled:
            rerank = LearnedReranker(self.learn_min_samples)
            if rerank.can_train(decisions):
                rerank.train(decisions)
            self.reranker = rerank
        return built

    def persist(self) -> None:
        for layer, idx in self.indexes.items():
            os.makedirs(self.store.root, exist_ok=True)
            with open(self.store.path(layer), "w", encoding="utf-8") as f:
                for e in idx.entries:
                    f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")

    def _reorder_preferred(self, hits: List[MemoryEntry]) -> List[MemoryEntry]:
        """Prefer entries of ``self.prefer_metric_source`` (e.g. real native_eval)
        over plan-proxy rollout entries, preserving reward-desc within groups.
        """
        if not self.prefer_metric_source:
            return hits
        pref = [e for e in hits if e.metric_source == self.prefer_metric_source]
        rest = [e for e in hits if e.metric_source != self.prefer_metric_source]
        return pref + rest

    def usage(self) -> Dict[str, Any]:
        return {
            "embedding_backend": self.embedding_backend,
            "min_similarity": self.min_similarity,
            "similarity": self.similarity,
            "semantic_queries": self._semantic_queries,
            "semantic_hits": self._semantic_hits,
            "fp_queries": self._fp_queries,
            "fp_hits": self._fp_hits,
        }

    def retrieve(self, layer: str, context: Optional[str] = None,
                 fingerprint: Optional[str] = None, top_k: Optional[int] = None) -> List[MemoryEntry]:
        idx = self.ensure(layer)
        query_text = (context or "").strip()
        if query_text and idx.embedder is not None:
            self._semantic_queries += 1
            hits = idx.retrieve(context_text=query_text, context_fingerprint=fingerprint,
                                top_k=top_k, min_reward=self.min_reward)
            if hits:
                self._semantic_hits += 1
        else:
            self._fp_queries += 1
            hits = idx.retrieve(context_fingerprint=fingerprint, top_k=top_k,
                                min_reward=self.min_reward)
            if hits:
                self._fp_hits += 1
        if self.reranker is not None and self.reranker.trained and self.embedder is None:
            # Re-rank the union (exact-fingerprint pool is all same-fp), then cap.
            hits = sorted(hits, key=lambda e: self.reranker.score(e, fingerprint or ""), reverse=True)
            hits = hits[: (top_k or self.top_k)]
        hits = self._reorder_preferred(hits)
        return hits[: (top_k or self.top_k)]

    def real_candidates(self, layer: str) -> List[MemoryEntry]:
        """All real native_eval entries for a layer, best-reward first.

        Used by the L5 policy to select the next (variant, seed) from observed
        real outcomes rather than plan-proxy rollout guesses.  Never blocks;
        returns an empty list when no real outcome has been observed yet.
        """
        idx = self.ensure(layer)
        pool = [e for e in idx.entries if e.metric_source == "real"]
        pool.sort(key=lambda e: float(e.reward if e.reward is not None else (e.outcome_score if e.outcome_score is not None else 0.0)),
                  reverse=True)
        return pool

    def remember_live(self, entry: MemoryEntry) -> None:
        """Add a just-observed decision to the layer index and persist it.

        This is how a real run learns its own round results so the next round
        can exploit/explore from observed outcomes (within-run self-improvement).
        """
        if entry.layer is None:
            return
        idx = self.ensure(entry.layer)
        idx.add(entry)
        self.persist()

    def render_prompt_block(self, layer: str, context: Optional[str] = None,
                            fingerprint: Optional[str] = None, top_k: Optional[int] = None) -> str:
        """Render the few-shot memory block injected into a layer prompt.

        Empty string when there is no memory hit (graceful no-memory path).
        """
        hits = self.retrieve(layer, context=context, fingerprint=fingerprint, top_k=top_k)
        if not hits:
            return ""
        lines = [f"[memory:{layer}] top {len(hits)} historical decisions:"]
        for i, e in enumerate(hits, 1):
            lines.append(
                f"{i}. op={e.op} variant={e.variant} reward={e.reward} outcome={e.outcome_score} "
                f"parent={e.parent_ref} reasoning={e.reasoning[:600]}"
            )
        return "\n".join(lines)
