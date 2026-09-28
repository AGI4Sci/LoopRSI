#!/usr/bin/env python3
"""Small machine-readable CUDA probe; it never loads a research dataset."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def probe() -> dict:
    started = time.time()
    result = {
        "schema_version": "omni-ar-gpu-probe/v1", "status": "failed",
        "cuda_available": False, "python": platform.python_version(),
    }
    try:
        import torch

        available = torch.cuda.is_available()
        result.update({
            "torch": torch.__version__, "cuda_available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if available else [],
        })
        if available:
            left = torch.ones((256, 256), device="cuda")
            right = torch.ones((256, 256), device="cuda")
            value = (left @ right).mean().item()
            torch.cuda.synchronize()
            result.update(status="ok", matrix_mean=value)
        else:
            result["error"] = "torch.cuda.is_available() is false"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["runtime_seconds"] = round(time.time() - started, 4)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
