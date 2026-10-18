"""Verify staged model files against MODEL_LOCK.json (size and SHA-256). Used at image build and self-test.

    python -m extractor_service.model_check /models /app/MODEL_LOCK.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: str, lock_path: str) -> list[str]:
    """Returns a list of problems; empty means every locked file is present with the locked bytes."""
    with open(lock_path, "r", encoding="utf-8") as fh:
        lock = json.load(fh)
    problems = []
    for entry in lock["files"]:
        rel = entry["path"]
        if rel.startswith("/") or ".." in rel.split("/"):
            problems.append(f"{rel}: unsafe path in lock")
            continue
        path = os.path.join(root, *rel.split("/"))
        if not os.path.isfile(path):
            problems.append(f"{rel}: missing")
            continue
        size = os.path.getsize(path)
        if size != entry["bytes"]:
            problems.append(f"{rel}: {size} bytes, expected {entry['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            problems.append(f"{rel}: sha256 {digest}, expected {entry['sha256']}")
    return problems


def main(argv: list[str]) -> int:
    root, lock_path = argv[1], argv[2]
    problems = verify(root, lock_path)
    for p in problems:
        print("MODEL_CHECK_FAIL", p)
    if problems:
        return 1
    with open(lock_path, "r", encoding="utf-8") as fh:
        count = len(json.load(fh)["files"])
    print(f"MODEL_CHECK_OK {count} files under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
