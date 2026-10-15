"""Verify that an rjob container can persist artifacts to the user's GPFS path."""

import os
import shutil
from pathlib import Path


OUTPUT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/gaozhangyang/"
    "looprsi-l5-closure-20261006/gpu-write-probe-20261006.txt"
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("container-to-login persistence verified\n", encoding="utf-8")
    print(f"persisted={OUTPUT} bytes={OUTPUT.stat().st_size}", flush=True)
    print("cwd=", os.getcwd(), "workdir=", Path.cwd(), flush=True)
    print("mounts:", flush=True)
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        if any(token in line.lower() for token in ("gpfs", "juice", "shared-storage", "nfs")):
            print(line, flush=True)
    for path in (OUTPUT.parent, Path("/data"), Path("/workdir"), Path("/tmp")):
        try:
            stat = path.stat()
            print(f"path={path} exists={path.exists()} dev={stat.st_dev} ino={stat.st_ino} writable={os.access(path, os.W_OK)} free={shutil.disk_usage(path).free}", flush=True)
        except OSError as exc:
            print(f"path={path} error={exc}", flush=True)


if __name__ == "__main__":
    main()
