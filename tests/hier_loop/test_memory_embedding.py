"""Tests for the semantic (cosine) memory retrieval layer (P3).

Covers the pluggable embedding backends (tfidf + dense ONNX) and their wiring
into MemoryEntry / MemoryIndex / MemoryManager / decision context text.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from hier_loop import memory as mem
from hier_loop import embedding as emb
from hier_loop.decisions import context_text, synthesize_context_text, make_decision


def _ctx(task="vcc25", metric="pearson_delta", baseline="0.30614", active=("L1", "L2", "L3"),
         layer="L3", round_=1, step=2):
    return {
        "task_id": task, "metric": metric, "baseline": baseline,
        "active_layers": list(active), "layer": layer, "round": round_, "step": step,
    }


def _entry(layer, ctx, reward=0.1, variant="v"):
    return mem.MemoryEntry(
        layer=layer, context_fingerprint="fp_x", context=context_text(ctx),
        op="improve", variant=variant, reward=reward, outcome_score=0.2,
        metric_source="real", kind="step", round=1, step=1,
    )


class ContextTextTests(unittest.TestCase):
    def test_context_text_deterministic(self):
        self.assertEqual(context_text(_ctx()), context_text(_ctx()))
        self.assertIn("task_id", context_text(_ctx()))
        self.assertIn("pearson_delta", context_text(_ctx()))

    def test_synthesize_from_record(self):
        rec = {"layer": "L4", "task_id": "vcc25", "metric": "pearson_delta",
               "round": 2, "step": 1}
        text = synthesize_context_text(rec)
        self.assertIn("L4", text)
        self.assertIn("vcc25", text)

    def test_decision_record_carries_context_text(self):
        msg = {
            "schema": "rsi.msg.v1", "msg_id": "m1", "run_id": "r", "hop": 1,
            "kind": "step", "from_layer": "L3", "to_layer": "L3", "round": 1,
            "payload": {"context": _ctx(), "action": {"step": 2, "operator": "improve"}},
        }
        rec = make_decision(msg, "run1", reward=0.1)
        self.assertEqual(rec["context_text"], context_text(_ctx()))


class TfidfRetrievalTests(unittest.TestCase):
    def _index(self, entries, min_sim=0.75):
        idx = mem.MemoryIndex("L3", top_k=5, embedder=emb.TfidfEmbedder(),
                              min_similarity=min_sim)
        for e in entries:
            idx.add(e)
        return idx

    def test_semantic_hit_on_similar_context(self):
        idx = self._index([_entry("L3", _ctx(step=1), reward=0.05),
                           _entry("L3", _ctx(step=1), reward=0.2, variant="w")])
        hits = idx.retrieve(context_text=context_text(_ctx(step=2)))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].reward, 0.2)  # best reward first on ties

    def test_rewrite_semantically_similar_context_hits(self):
        # Different wording of the same semantic context still retrieves.
        idx = self._index([_entry("L3", _ctx(baseline="0.30614"), reward=0.1)])
        rewritten = _ctx(baseline="0.306")  # slight numeric noise
        hits = idx.retrieve(context_text=context_text(rewritten))
        self.assertEqual(len(hits), 1)

    def test_miss_below_min_similarity_returns_empty(self):
        idx = self._index([_entry("L3", _ctx(task="naturebench", metric="accuracy",
                                            baseline="nan", active=("L1",),
                                            layer="L1", round_=9, step=9), reward=0.1)])
        hits = idx.retrieve(context_text=context_text(_ctx(task="vcc25")))
        self.assertEqual(hits, [])

    def test_fingerprint_fallback_for_entries_without_context(self):
        idx = mem.MemoryIndex("L3", top_k=5, embedder=emb.TfidfEmbedder())
        idx.add(mem.MemoryEntry(layer="L3", context_fingerprint="fp_abc", reward=0.3))
        idx.add(mem.MemoryEntry(layer="L3", context_fingerprint="fp_xyz", reward=0.9))
        hits = idx.retrieve(context_fingerprint="fp_abc")
        self.assertEqual([h.reward for h in hits], [0.3])

    def test_min_reward_filter(self):
        idx = self._index([_entry("L3", _ctx(), reward=0.01), _entry("L3", _ctx(), reward=0.5)])
        hits = idx.retrieve(context_text=context_text(_ctx()), min_reward=0.1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].reward, 0.5)


class ManagerWiringTests(unittest.TestCase):
    def test_build_memory_manager_tfidf_from_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "out": {"dir": os.path.join(tmp, "out")},
                "memory": {
                    "enabled": True,
                    "retrieval": {
                        "top_k": 4, "min_similarity": 0.6,
                        "embedding": {"backend": "tfidf"},
                    },
                    "prior_real_results": [
                        {"layer": "L5", "variant": "v1", "seed": 1,
                         "outcome_score": 0.05, "reward": 0.0},
                    ],
                    "out": {"dir": os.path.join(tmp, "out", "memory")},
                },
            }
            mgr = mem.build_memory_manager(cfg, os.path.join(tmp, "decisions"))
            self.assertEqual(mgr.embedding_backend, "tfidf")
            hits = mgr.retrieve("L5", context=context_text(_ctx(layer="L5")),
                                fingerprint="fp_xyz")
            self.assertEqual(len(hits), 1)
            usage = mgr.usage()
            self.assertEqual(usage["embedding_backend"], "tfidf")
            self.assertGreaterEqual(usage["semantic_queries"], 1)


@unittest.skipUnless(
    __import__("importlib.util").util.find_spec("onnxruntime") is not None,
    "onnxruntime not installed",
)
class DenseOnnxTests(unittest.TestCase):
    MODEL_DIR = str(Path(__file__).resolve().parents[2] / "models" / "bge-small-zh-v1.5-onnx")

    def setUp(self):
        if not os.path.isfile(os.path.join(self.MODEL_DIR, "onnx", "model.onnx")):
            self.skipTest("ONNX model files not present")
        self.embdr = emb.OnnxEmbedder(self.MODEL_DIR)

    def test_embedding_dimensions_and_norm(self):
        v = self.embdr.embed("vcc25 pearson_delta")
        self.assertEqual(len(v), self.embdr.hidden)
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_cosine_ordering_semantic(self):
        a = self.embdr.embed("vcc25 pearson_delta baseline")
        b = self.embdr.embed("vcc25 pearson_delta baseline")
        c = self.embdr.embed("naturebench accuracy")
        self.assertGreater(emb.cosine(a, b), emb.cosine(a, c))

    def test_semantic_retrieval_with_dense_backend(self):
        idx = mem.MemoryIndex("L5", top_k=5, embedder=self.embdr, min_similarity=0.5)
        idx.add(_entry("L5", _ctx(layer="L5", round_=1), reward=0.3, variant="r17"))
        idx.add(_entry("L5", _ctx(layer="L5", round_=9), reward=0.1, variant="r1"))
        hits = idx.retrieve(context_text=context_text(_ctx(layer="L5", round_=2)))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].variant, "r17")


if __name__ == "__main__":
    unittest.main()
