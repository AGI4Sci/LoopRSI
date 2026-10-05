"""Memory-bounded loader for official GEARS pickle shards."""

from __future__ import annotations

import json
import pickle
from pathlib import Path


class ShardBatchLoader:
    """Iterate official GEARS ``Data`` objects without joining all shards."""

    def __init__(self, shard_dir: str | Path, split: str, batch_size: int, seed: int = 0):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.root = Path(shard_dir)
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.entries = [entry for entry in manifest["shards"] if entry["split"] == split]
        if not self.entries:
            raise ValueError(f"no shards for split {split!r}")
        self.batch_size, self.seed = batch_size, seed

    def __len__(self) -> int:
        cells = sum(entry["cells"] for entry in self.entries)
        return (cells + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        import torch
        from torch_geometric.data import Batch

        rng = torch.Generator().manual_seed(self.seed)
        entries = list(self.entries)
        order = torch.randperm(len(entries), generator=rng).tolist()
        pending = []
        for index in order:
            with (self.root / entries[index]["file"]).open("rb") as stream:
                graphs = pickle.load(stream)
            for graph in graphs:
                pending.append(graph)
                if len(pending) == self.batch_size:
                    yield Batch.from_data_list(pending)
                    pending = []
        if pending:
            yield Batch.from_data_list(pending)
