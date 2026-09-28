#!/usr/bin/env python3
"""RSI Step0 minimal rjob training driver.

Transcribes a real heuresis run into trajectory records, builds the SFT
dataset, fits the Pi behavior-clone policy with the torch backend and writes a
structured result manifest. Used as the rjob entrypoint to close the
"launch a training via rjob, get a result" loop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import rsi_step0.trajectory as TRAJ
import rsi_step0.training as TRAIN
import rsi_step0.reward as R


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_cfg(path: str) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="RSI Step0 rjob training driver")
    ap.add_argument("--run-dir", default="output/rsi_step0/runs/real_engine_r1")
    ap.add_argument("--traj-dir", default="output/rsi_step0/trajectories/rjob")
    ap.add_argument("--sft-out", default="output/rsi_step0/datasets/sft_rjob")
    ap.add_argument("--ckpt-out", default="output/rsi_step0/checkpoints/rjob")
    ap.add_argument("--result-out", default="output/rsi_step0/rjob_training_result.json")
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--config", default="configs/rsi_step0/train_pi.yaml")
    args = ap.parse_args()

    start = time.time()
    result = {
        "command": "rjob_train_rsi",
        "created_at": _now(),
        "run_dir": os.path.abspath(args.run_dir),
        "backend": args.backend,
        "status": "failed",
        "steps": {},
        "error": None,
        "host": os.uname().nodename,
    }

    steps = result["steps"]

    # 1) transcribe
    try:
        t = TRAJ.transcribe(args.run_dir, args.traj_dir)
        steps["transcribe"] = t
    except Exception as exc:
        result["error"] = f"transcribe: {exc!r}"
        _write_result(args.result_out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    # 2) build SFT dataset
    try:
        ds = TRAIN.build_sft_dataset(args.traj_dir, args.sft_out)
        steps["sft_dataset"] = {
            "num_positive": ds.get("num_positive"),
            "num_sft_records": ds.get("num_sft_records"),
            "tasks": ds.get("tasks"),
            "operators": ds.get("operators"),
            "record_path": ds.get("record_path"),
        }
    except Exception as exc:
        result["error"] = f"build_sft_dataset: {exc!r}"
        _write_result(args.result_out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    # 3) torch training
    cfg = _load_cfg(args.config)
    train_cfg = dict(cfg.get("train", {}) if isinstance(cfg, dict) else {})
    train_cfg["backend"] = args.backend
    train_cfg["sft_iters"] = args.iters
    try:
        trainer = TRAIN.PiTrainer(backend=args.backend, seed=int(train_cfg.get("seed", 0)),
                                  config=train_cfg)
        meta = trainer.train_sft(ds, args.ckpt_out)
        steps["train"] = {
            "backend": meta.get("backend"),
            "num_records": meta.get("num_records"),
            "num_train_samples": meta.get("num_train_samples"),
            "artifact": meta.get("artifact"),
            "artifact_path": meta.get("artifact_path"),
            "hash": meta.get("hash"),
            "note": meta.get("manifest", {}).get("note"),
            "policy": meta.get("policy", {}).get("trained", meta.get("policy", {}).get("weights")),
        }
        if meta.get("backend") == "torch" and meta.get("artifact_path") and os.path.exists(meta["artifact_path"]):
            result["status"] = "succeeded"
        else:
            result["status"] = "succeeded_no_artifact"
    except Exception as exc:
        result["error"] = f"train: {exc!r}"
        result["status"] = "failed"
        # still surface steps + error, but keep the "got a result" semantics
        _write_result(args.result_out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    result["elapsed_seconds"] = round(time.time() - start, 3)
    _write_result(args.result_out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "succeeded" else 2


def _write_result(path: str, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
