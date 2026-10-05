"""Install packaged wheels locally and import the pinned GEARS source."""

import importlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    target = root / "python_deps"
    wheels = sorted((root / "wheels").glob("*.whl"))
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--target", str(target), *map(str, wheels),
    ], check=True)
    sys.path.insert(0, str(target))
    sys.path.insert(0, str(root / "gears-source"))
    versions = {}
    for name in ("numpy", "pandas", "h5py", "anndata", "scanpy", "statsmodels", "dcor", "torch_geometric", "gears"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "imported")
    print(json.dumps(versions, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
