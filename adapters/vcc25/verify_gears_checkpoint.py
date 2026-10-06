"""Load a recovered GEARS checkpoint and report tensor integrity."""

import hashlib
import json
import pickle
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent


def main() -> None:
    checkpoint = ROOT / "checkpoint" / "model.pt"
    config_path = ROOT / "checkpoint" / "config.pkl"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with config_path.open("rb") as stream:
        config = pickle.load(stream)
    tensors = {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}
    if len(tensors) != len(state) or not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("checkpoint contains non-tensor or non-finite state")
    print(json.dumps({
        "status": "verified",
        "epoch_count_from_run": 3,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "tensor_count": len(tensors),
        "tensor_elements": sum(value.numel() for value in tensors.values()),
        "gene_count": config.get("num_genes"),
        "state_shapes": {key: list(value.shape) for key, value in tensors.items()},
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
