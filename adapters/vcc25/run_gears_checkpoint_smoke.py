"""Train bounded real H1 GEARS smoke and emit its small checkpoint to logs."""

import json
import os
import sys
from pathlib import Path

from probe_gears_import import main as install_and_check


def main() -> None:
    root = Path(__file__).resolve().parent
    install_and_check()
    sys.path.insert(0, str(root / "python_deps"))
    os.environ["LOOPRSI_REMOTE_GPU_JOB"] = "1"
    os.environ["LOOPRSI_CHARGED_GROUP"] = "deepdivegzy_gpu_pool"
    from adapters.vcc25.gears_h1 import main as train

    result = train([
        "--source-repository", str(root / "gears-source"),
        "--processed-dataset", str(root / "dataset"),
        "--split-json", str(root / "split.json"),
        "--gene2go", str(root / "gene2go_all.pkl"),
        "--gene-names", str(root / "gene_names.txt"),
        "--shard-dir", str(root / "shards"),
        "--compute-smoke-de",
        "--official-go-csv", str(root / "go_essential_all.csv"),
        "--epochs", "3",
        "--export-checkpoint-to-logs",
        "--checkpoint-log-limit-mib", "32",
        "--output", str(root / "output"),
    ])
    print(json.dumps({"runner_exit": result,
                      "run_json": (root / "output" / "run.json").read_text()}), flush=True)
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
