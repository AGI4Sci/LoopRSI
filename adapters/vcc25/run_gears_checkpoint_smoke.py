"""Train bounded real H1 GEARS smoke and emit its small checkpoint to logs."""

import json
import os
import sys
import argparse
from pathlib import Path

from probe_gears_import import main as install_and_check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--checkpoint-log-limit-mib", type=int, default=32)
    parser.add_argument("--compute-smoke-de", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    install_and_check()
    sys.path.insert(0, str(root / "python_deps"))
    os.environ["LOOPRSI_REMOTE_GPU_JOB"] = "1"
    os.environ["LOOPRSI_CHARGED_GROUP"] = "deepdivegzy_gpu_pool"
    from adapters.vcc25.gears_h1 import main as train

    train_args = [
        "--source-repository", str(root / "gears-source"),
        "--processed-dataset", str(root / "dataset"),
        "--split-json", str(root / "split.json"),
        "--gene2go", str(root / "gene2go_all.pkl"),
        "--gene-names", str(root / "gene_names.txt"),
        "--shard-dir", str(root / "shards"),
        "--official-go-csv", str(root / "go_essential_all.csv"),
        "--epochs", str(args.epochs),
        "--export-checkpoint-to-logs",
        "--checkpoint-log-limit-mib", str(args.checkpoint_log_limit_mib),
        "--output", str(root / "output"),
    ]
    if args.compute_smoke_de:
        train_args.append("--compute-smoke-de")
    result = train(train_args)
    print(json.dumps({"runner_exit": result,
                      "run_json": (root / "output" / "run.json").read_text()}), flush=True)
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
