from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    if len(args.inputs) < 2 or args.output.exists():
        raise ValueError("ensemble requires at least two inputs and a new output")

    handles = [h5py.File(path, "r") for path in args.inputs]
    try:
        shape = handles[0]["X"].shape
        if any(handle["X"].shape != shape for handle in handles[1:]):
            raise ValueError("prediction shapes differ")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as out:
            for key in handles[0]:
                if key != "X":
                    handles[0].copy(key, out)
            x = out.create_dataset(
                "X", shape=shape, dtype="float32",
                chunks=(min(args.chunk_size, shape[0]), shape[1]),
            )
            x.attrs.update(handles[0]["X"].attrs)
            for start in range(0, shape[0], args.chunk_size):
                end = min(start + args.chunk_size, shape[0])
                total = np.zeros((end - start, shape[1]), dtype=np.float64)
                for handle in handles:
                    total += handle["X"][start:end]
                x[start:end] = (total / len(handles)).astype(np.float32)
            out["uns"].attrs["seed_ensemble"] = json.dumps({
                "aggregation": "elementwise_arithmetic_mean",
                "inputs": [str(path) for path in args.inputs],
                "members": len(args.inputs),
            }, sort_keys=True)
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
