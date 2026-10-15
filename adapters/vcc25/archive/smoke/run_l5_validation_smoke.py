"""Run a bounded train/validation L5 smoke from an offline rjob folder."""
from __future__ import annotations

import json
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ASSETS = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/zhangzhicheng/naturebench/lingshu_cell_h1")
OUTPUT_ROOT = Path("/mnt/shared-storage-gpfs2/beam-gpfs02/gaozhangyang/looprsi-l5-closure-20261006")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="smoke-8x4")
    parser.add_argument("--train-targets", type=int, default=8)
    parser.add_argument("--val-targets", type=int, default=4)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seeds", default="20260907")
    args = parser.parse_args()
    if not args.name.replace("-", "").isalnum() or min(args.train_targets, args.val_targets, args.rows, args.steps) <= 0:
        raise ValueError("invalid smoke bounds")
    output = OUTPUT_ROOT / args.name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing validation run: {output}")
    required = {
        "train": ASSETS / "assets/official_2025/train/adata_Training.h5ad",
        "validation": ASSETS / "assets/official_2025/validation/adata_Validation.h5ad",
        "embedding": ASSETS / "assets/lingshu_hf_b77f980/gene_embeddings.npz",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"H1 assets absent in rjob: {missing}")
    wheels = sorted((ROOT / "wheels").glob("*.whl"))
    if len(wheels) != 2:
        raise FileNotFoundError("expected pinned numpy and h5py wheels")
    target = ROOT / "python_deps"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                    "--target", str(target), *(str(path) for path in wheels)], check=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(target), str(ROOT / "implementation")])
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    command = [sys.executable, str(ROOT / "implementation/official_h1_autonomous_research_loop.py"),
               "--output-dir", str(output), "--n-train-targets", str(args.train_targets), "--n-val-targets", str(args.val_targets),
               "--max-rows-per-target", str(args.rows), "--max-steps", str(args.steps), "--seeds", args.seeds,
               "--device", "cuda"]
    print(json.dumps({"command": command, "test_expression_read": False}), flush=True)
    subprocess.run(command, check=True, env=env, timeout=1800)
    summary = json.loads((output / "lineage_summary.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": summary["status"], "best_candidate": summary["best_candidate"],
                      "data_contract": summary["data_contract"],
                      "results": {key: value["aggregate"] for key, value in summary["results"].items()}},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
