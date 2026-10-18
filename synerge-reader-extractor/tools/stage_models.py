"""Copy the reviewed model files into the extractor build context and verify them (no downloads).

    python tools/stage_models.py <source-root> <dest-root> <MODEL_LOCK.json>

<source-root> holds the D12 prototype layout: ``models/<repo-cache-dir>/...`` and ``tessdata/...``
(for example ``...\\prototype\\docling_d12_20261016\\artifacts``). Every file is hashed before
and after copying, nothing outside the lock is copied, and an existing destination file is only
accepted when it already has the locked bytes. Writes ``STAGED_MODELS.json`` into <dest-root>.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from extractor_service.model_check import sha256_file, verify  # noqa: E402


def source_path(source_root: str, rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "tessdata":
        return os.path.join(source_root, *parts)
    return os.path.join(source_root, "models", *parts)


def main(argv: list[str]) -> int:
    source_root, dest_root, lock_path = argv[1:4]
    with open(lock_path, "r", encoding="utf-8") as fh:
        lock = json.load(fh)
    record = []
    for entry in lock["files"]:
        rel = entry["path"]
        src = source_path(source_root, rel)
        dst = os.path.join(dest_root, *rel.split("/"))
        if os.path.getsize(src) != entry["bytes"] or sha256_file(src) != entry["sha256"]:
            print(f"STAGE_FAIL source does not match the lock: {rel}")
            return 1
        if os.path.exists(dst):
            if sha256_file(dst) != entry["sha256"]:
                print(f"STAGE_FAIL destination exists with different bytes: {rel}")
                return 1
            action = "already_present"
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".partial"
            shutil.copyfile(src, tmp)
            if sha256_file(tmp) != entry["sha256"]:
                os.remove(tmp)
                print(f"STAGE_FAIL copy hash mismatch: {rel}")
                return 1
            os.replace(tmp, dst)
            action = "copied"
        record.append({"path": rel, "sha256": entry["sha256"], "bytes": entry["bytes"], "action": action})
    problems = verify(dest_root, lock_path)
    if problems:
        for p in problems:
            print("STAGE_FAIL", p)
        return 1
    with open(os.path.join(dest_root, "STAGED_MODELS.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"staged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "model_lock_sha256": sha256_file(lock_path), "files": record}, fh, indent=1)
        fh.write("\n")
    print(f"STAGE_OK {len(record)} files -> {dest_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
