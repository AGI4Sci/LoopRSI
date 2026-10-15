"""Emit a synthetic 1 MiB artifact as tagged log lines for transport testing."""

import base64
import hashlib
import os
import time
import zlib


def main() -> None:
    raw = os.urandom(1024 * 1024)
    encoded = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    print(f"ARTIFACT_BEGIN {hashlib.sha256(raw).hexdigest()} {len(raw)}", flush=True)
    for index, offset in enumerate(range(0, len(encoded), 2048)):
        print(f"ARTIFACT_CHUNK {index:06d} {encoded[offset:offset + 2048]}", flush=True)
        if index and index % 32 == 0:
            time.sleep(0.01)
    print("ARTIFACT_END", flush=True)


if __name__ == "__main__":
    main()
