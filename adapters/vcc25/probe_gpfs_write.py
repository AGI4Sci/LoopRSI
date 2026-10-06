"""Verify that an rjob container can persist artifacts to the user's GPFS path."""

from pathlib import Path


OUTPUT = Path(
    "/mnt/shared-storage-gpfs2/beam-gpfs02/gaozhangyang/"
    "looprsi-l5-closure-20261006/gpu-write-probe-20261006.txt"
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("container-to-login persistence verified\n", encoding="utf-8")
    print(f"persisted={OUTPUT} bytes={OUTPUT.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
