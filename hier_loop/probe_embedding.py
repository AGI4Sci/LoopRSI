#!/usr/bin/env python3
"""Probe the dense ONNX embedding backend and print cosine similarities.

Used to verify the downloaded ONNX model is wired and working, and to pick a
sensible ``retrieval.min_similarity`` for real vcc25 contexts.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from hier_loop.embedding import OnnxEmbedder, cosine  # noqa: E402
from hier_loop.decisions import context_text  # noqa: E402


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _ROOT, "models", "bge-small-zh-v1.5-onnx"
    )
    embdr = OnnxEmbedder(model_dir)
    print(f"backend=dense model={model_dir} hidden={embdr.hidden}")
    samples: Dict[str, Dict[str, Any]] = {
        "q_l5_r1": {"task_id": "vcc25", "metric": "pearson_delta", "baseline": "0.30614",
                    "active_layers": ["L5"], "layer": "L5", "round": "1", "step": "1"},
        "l5_prior": {"task_id": "vcc25", "metric": "pearson_delta", "baseline": "",
                     "active_layers": ["L5"], "layer": "L5", "round": "", "step": ""},
        "l5_r2": {"task_id": "vcc25", "metric": "pearson_delta", "baseline": "0.30614",
                  "active_layers": ["L5"], "layer": "L5", "round": "2", "step": "1"},
        "q_l3_r1": {"task_id": "vcc25", "metric": "pearson_delta", "baseline": "0.30614",
                    "active_layers": ["L1", "L2", "L3"], "layer": "L3", "round": "1", "step": "2"},
        "l3_synth": {"task_id": "vcc25", "metric": "pearson_delta", "baseline": "",
                     "active_layers": ["L3"], "layer": "L3", "round": "", "step": ""},
        "other_task": {"task_id": "naturebench", "metric": "accuracy", "baseline": "0.5",
                       "active_layers": ["L1"], "layer": "L1", "round": "1", "step": "1"},
    }
    vecs = {k: embdr.embed(context_text(v)) for k, v in samples.items()}
    keys = list(samples)
    print("keys:", keys)
    print("similarity matrix (rows=query):")
    for a in keys:
        row = [round(cosine(vecs[a], vecs[b]), 4) for b in keys]
        print(f"{a:12s} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
