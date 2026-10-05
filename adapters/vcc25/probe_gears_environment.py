"""Report GEARS runtime dependencies available in an rjob container."""

import importlib.util
import json
import sys


MODULES = (
    "numpy", "torch", "torch_geometric", "scanpy", "scipy", "sklearn",
    "dcor", "h5py", "pandas", "anndata", "networkx", "wandb", "tqdm",
)


if __name__ == "__main__":
    print(json.dumps({
        "python": sys.version.split()[0],
        "modules": {name: importlib.util.find_spec(name) is not None for name in MODULES},
    }, sort_keys=True), flush=True)
