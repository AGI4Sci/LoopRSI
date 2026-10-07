"""Install a bounded offline PRiMeFlow environment and probe imports."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "numpy==2.2.6", "pandas==2.3.3", "scipy==1.16.3", "anndata",
    "array-api-compat", "h5py==3.14.0", "legacy-api-wrap", "natsort", "packaging",
    "scverse-misc", "typing-extensions", "zarr", "python-dateutil", "pytz", "tzdata",
    "six", "donfig", "google-crc32c", "msgspec", "numcodecs", "pydantic-settings",
    "pydantic", "pydantic-core", "annotated-types", "typing-inspection", "session-info2",
    "tqdm", "lightning", "torchvision==0.24.1", "tensorboard",
    "lightning-utilities", "torchmetrics", "pytorch-lightning", "fsspec", "filelock",
    "sympy", "jinja2",
    "hydra-core", "hydra-colorlog", "matplotlib", "seaborn",
    "scikit-learn", "scikit-misc", "adjusttext", "pytest", "rich",
    "psycopg2-binary", "optuna", "ray", "python-dotenv", "geomloss",
    "pytorch-metric-learning", "pdex", "cell-eval==0.6.1", "lightning-bolts",
    "ema-pytorch", "jax==0.6.1", "ott-jax", "torchdyn", "setuptools==81.0.0",
    "huggingface-hub==1.33.0",
    "sqlparse",
    "httpx2==2.13.1", "httpcore2==2.13.1", "anyio", "idna", "truststore", "h11",
)


def build_install_command(wheelhouse: Path, target: Path) -> List[str]:
    return [
        sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
        "--find-links", str(wheelhouse), "--target", str(target), *PACKAGES,
    ]


def build_resolved_install_command(wheelhouse: Path, target: Path) -> List[str]:
    return [
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", str(wheelhouse), "--target", str(target),
        "mlflow-skinny==3.16.1", "scanpy==1.12.4",
    ]


def resolve_assets(package_root: Path):
    source_root = package_root / "sources"
    return package_root / "wheelhouse", {
        "primeflow": source_root / "primeflow-443295258e364e394b60de27a861d8d1cf3b1fa6.tar.gz",
        "perturbench": source_root / "perturbench-c84038bc1ea409aa54f3832cfa6f34f5059adf0c.tar.gz",
    }


def activate_paths(target: Path, source_paths: List[Path], path_list: List[str]) -> None:
    path_list[:0] = [*(str(path) for path in source_paths), str(target)]


def main() -> None:
    wheelhouse, source_archives = resolve_assets(PACKAGE_ROOT)
    with tempfile.TemporaryDirectory(prefix="primeflow-offline-") as tmp:
        root = Path(tmp)
        target = root / "deps"
        sources = root / "sources"
        target.mkdir()
        sources.mkdir()
        subprocess.run(build_install_command(wheelhouse, target), check=True)
        subprocess.run(build_resolved_install_command(wheelhouse, target), check=True)
        source_paths = []
        for name, archive in source_archives.items():
            with tarfile.open(archive) as handle:
                handle.extractall(sources, filter="data")
            candidates = sorted(sources.glob(f"{name}-*/src"))
            if len(candidates) != 1:
                raise RuntimeError(f"expected one extracted {name} source tree, found {candidates}")
            source_paths.append(candidates[0])
        activate_paths(target, source_paths, sys.path)
        results = {}
        for module in ("torch", "anndata", "perturbench", "primeflow", "primeflow.modelcore.train"):
            try:
                imported = importlib.import_module(module)
                results[module] = {"ok": True, "version": str(getattr(imported, "__version__", "unknown"))}
            except Exception as exc:
                results[module] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({
            "schema_version": "vcc25.primeflow-offline-install-probe/v1",
            "imports": results,
            "final_test_expression_read": False,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
