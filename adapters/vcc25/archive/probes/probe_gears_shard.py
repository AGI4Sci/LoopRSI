"""Check one packaged GEARS shard batch inside an rjob GPU container."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parent
    wheel = next(root.glob("torch_geometric-*.whl"))
    target = root / "python_deps"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)], check=True)
    sys.path.insert(0, str(target))
    from torch_geometric.data import Batch

    manifest = json.loads((root / "manifest.json").read_text())
    entry = next(item for item in manifest["shards"] if item["split"] == "train")
    with (root / entry["file"]).open("rb") as stream:
        graphs = pickle.load(stream)
    batch = Batch.from_data_list(graphs[:2])
    result = {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "batch_graphs": batch.num_graphs,
        "x_shape": list(batch.x.shape),
        "y_shape": list(batch.y.shape),
        "condition_count": len(batch.pert),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["cuda"] or result["batch_graphs"] != 2:
        raise RuntimeError("GPU shard probe failed")


if __name__ == "__main__":
    main()
