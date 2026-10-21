#!/usr/bin/env python3
"""Stage the compatibility runtime's dedicated Ollama model store from existing
caches, copying ONLY the pinned manifests and the blobs they reference.

Standard library only (it runs inside an existing image with no network):

    stage_model_store.py plan   --pins PINS.json --source NAME=DIR [...] --dest DIR [--headroom-gib N]
    stage_model_store.py stage  --pins PINS.json --source NAME=DIR [...] --dest DIR [--headroom-gib N]
    stage_model_store.py verify --pins PINS.json --store DIR [--quick]

A source DIR is an Ollama models directory (the one holding ``manifests/`` and
``blobs/``), mounted read-only. Sources are only ever opened for reading, file
by file, without following symbolic links; nothing in them is created,
modified, pruned or deleted, and nothing is downloaded.

``plan`` reads each pinned manifest, checks its sha256 against the pin (the
digest ``/api/tags`` reports), and checks that every referenced blob -- the
config and every layer, including template, licence, parameter and projector
layers -- is present as a regular file of the declared size. It then measures
the space required against the destination's free space plus headroom. It
writes nothing.

``stage`` does the same and refuses before copying anything when the space is
insufficient, when ``DEST/models`` already exists, or when another staging run
holds the lock. It copies into a fresh hidden directory under DEST, hashing
every byte as it is copied, re-hashes the staged copy, and only then publishes
it with one rename to ``DEST/models``. Any failure removes the partial staging
directory it created and nothing is published. Staged files are read-only
(0444, directories 0755) so a container whose root has dropped every
capability can still read them.

``verify`` re-checks a published store read-only: each pinned manifest digest,
each referenced blob's size and (unless ``--quick``) its sha256, and that the
store holds nothing else.

Output is one JSON report on stdout (tags, digests, sizes and counts; never
file content). Exit status: 0 success, 2 usage, 3 refused or failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

CHUNK = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
GIB = 1024 ** 3
DEFAULT_HEADROOM_GIB = 16
STAGING_PREFIX = ".staging-"
LOCK_NAME = ".stage.lock"
REPORT_NAME = "STAGED.json"

_HEX = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:([0-9a-f]{64})$")
# Library models only ("name:tag"): Ollama files them under
# manifests/registry.ollama.ai/library/<name>/<tag>, the path /api/tags names.
_TAG = re.compile(r"^([a-z0-9][a-z0-9._-]{0,79}):([a-z0-9][a-z0-9._-]{0,79})$")
_SOURCE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class StageError(Exception):
    """A refusal or failure, with a fixed message naming only tags/digests."""


@dataclass(frozen=True)
class ModelPin:
    tag: str
    manifest_digest: str  # 64 hex
    source: str


@dataclass(frozen=True)
class BlobRef:
    hexdigest: str
    size: int


@dataclass(frozen=True)
class PlannedModel:
    pin: ModelPin
    manifest: bytes
    blobs: Tuple[BlobRef, ...]


# --- pins and paths --------------------------------------------------------------------


def load_pins(path: str) -> List[ModelPin]:
    with open(path, "rb") as fh:
        doc = json.loads(fh.read().decode("utf-8"))
    entries = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or not entries:
        raise StageError("the pins file lists no models")
    pins, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise StageError("a pinned model entry is not an object")
        tag, digest, source = entry.get("tag"), entry.get("manifest_digest"), entry.get("source")
        if not isinstance(tag, str) or not _TAG.match(tag):
            raise StageError("a pinned model tag is not a library name:tag")
        if not isinstance(digest, str) or not _HEX.match(digest):
            raise StageError(f"{tag}: the pinned manifest digest is not 64 lowercase hex characters")
        if not isinstance(source, str) or not _SOURCE.match(source):
            raise StageError(f"{tag}: the pinned source name is invalid")
        if tag in seen:
            raise StageError(f"{tag}: pinned twice")
        seen.add(tag)
        pins.append(ModelPin(tag, digest, source))
    return pins


def manifest_parts(tag: str) -> Tuple[str, ...]:
    name, version = tag.split(":", 1)
    return ("manifests", "registry.ollama.ai", "library", name, version)


def blob_parts(hexdigest: str) -> Tuple[str, ...]:
    return ("blobs", f"sha256-{hexdigest}")


# --- read-only, no-follow file access ----------------------------------------------------

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


def _open_regular(root: str, parts: Sequence[str]) -> int:
    """A read-only descriptor for ROOT/parts..., refusing a symbolic link at
    any component and anything that is not a regular file. On Linux every
    component is opened relative to its parent (openat) with O_NOFOLLOW."""
    if os.open in os.supports_dir_fd and _O_NOFOLLOW and _O_DIRECTORY:
        dir_fd = os.open(root, os.O_RDONLY | _O_DIRECTORY)
        try:
            for part in parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=dir_fd)
                os.close(dir_fd)
                dir_fd = next_fd
            fd = os.open(parts[-1], os.O_RDONLY | _O_NOFOLLOW, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
    else:  # portable fallback (used by the Windows unit tests)
        path = root
        for part in parts:
            path = os.path.join(path, part)
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise StageError("a symbolic link was found in a model store; refused")
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_BINARY)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise StageError("a model store entry is not a regular file; refused")
    return fd


def _read_small(root: str, parts: Sequence[str], limit: int) -> bytes:
    fd = _open_regular(root, parts)
    with os.fdopen(fd, "rb") as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        raise StageError("a manifest is larger than any Ollama manifest should be; refused")
    return data


def _hash_fd(fd: int, expected_size: int) -> str:
    digest, total = hashlib.sha256(), 0
    while True:
        chunk = os.read(fd, CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise StageError("a blob is larger than its manifest declares")
        digest.update(chunk)
    if total != expected_size:
        raise StageError("a blob is smaller than its manifest declares")
    return digest.hexdigest()


# --- plan ----------------------------------------------------------------------------------


def _blob_refs(tag: str, manifest: bytes) -> Tuple[BlobRef, ...]:
    try:
        doc = json.loads(manifest.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise StageError(f"{tag}: the manifest is not JSON") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("config"), dict) or not isinstance(doc.get("layers"), list):
        raise StageError(f"{tag}: the manifest has no config and layer list")
    refs, seen = [], set()
    for entry in [doc["config"], *doc["layers"]]:
        if not isinstance(entry, dict):
            raise StageError(f"{tag}: a manifest entry is not an object")
        match = _DIGEST.match(entry.get("digest") or "") if isinstance(entry.get("digest"), str) else None
        size = entry.get("size")
        if match is None or type(size) is not int or size < 0:
            raise StageError(f"{tag}: a manifest entry has no valid digest and size")
        hexd = match.group(1)
        if hexd in seen:
            continue
        seen.add(hexd)
        refs.append(BlobRef(hexd, size))
    if len(refs) < 2:
        raise StageError(f"{tag}: the manifest references no model layers")
    return tuple(refs)


def plan_models(pins: Sequence[ModelPin], sources: Dict[str, str]) -> List[PlannedModel]:
    missing = sorted({p.source for p in pins} - set(sources))
    if missing:
        raise StageError("no source directory was given for: " + ", ".join(missing))
    planned = []
    for pin in pins:
        root = sources[pin.source]
        try:
            manifest = _read_small(root, manifest_parts(pin.tag), MAX_MANIFEST_BYTES)
        except OSError:
            raise StageError(f"{pin.tag}: the pinned manifest is not in source '{pin.source}'") from None
        if hashlib.sha256(manifest).hexdigest() != pin.manifest_digest:
            raise StageError(f"{pin.tag}: the manifest in source '{pin.source}' differs from the pinned digest")
        refs = _blob_refs(pin.tag, manifest)
        for ref in refs:
            try:
                fd = _open_regular(root, blob_parts(ref.hexdigest))
            except OSError:
                raise StageError(f"{pin.tag}: blob sha256:{ref.hexdigest} is missing from source '{pin.source}'") from None
            try:
                if os.fstat(fd).st_size != ref.size:
                    raise StageError(f"{pin.tag}: blob sha256:{ref.hexdigest} has a different size than declared")
            finally:
                os.close(fd)
        planned.append(PlannedModel(pin, manifest, refs))
    return planned


def required_bytes(planned: Sequence[PlannedModel]) -> int:
    unique = {}
    for model in planned:
        for ref in model.blobs:
            unique[ref.hexdigest] = ref.size
    return sum(unique.values()) + sum(len(m.manifest) for m in planned)


def free_bytes(path: str) -> int:
    if hasattr(os, "statvfs"):
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    return shutil.disk_usage(path).free


def _model_report(planned: Sequence[PlannedModel]) -> list:
    return [{"tag": m.pin.tag, "manifest_digest": m.pin.manifest_digest, "source": m.pin.source,
             "blobs": len(m.blobs), "bytes": sum(r.size for r in m.blobs)} for m in planned]


def plan(pins, sources, dest: str, headroom: int) -> dict:
    planned = plan_models(pins, sources)
    need, free = required_bytes(planned), free_bytes(dest)
    return {"result": "PLAN_OK" if free >= need + headroom else "INSUFFICIENT_SPACE",
            "models": _model_report(planned), "required_bytes": need, "headroom_bytes": headroom,
            "free_bytes": free, "published": os.path.lexists(os.path.join(dest, "models"))}


# --- stage ---------------------------------------------------------------------------------


def _copy_blob(root: str, ref: BlobRef, blobs_dir: str) -> None:
    target = os.path.join(blobs_dir, f"sha256-{ref.hexdigest}")
    partial = target + ".partial"
    src = _open_regular(root, blob_parts(ref.hexdigest))
    try:
        dst = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY, 0o644)
        try:
            digest, total = hashlib.sha256(), 0
            while True:
                chunk = os.read(src, CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > ref.size:
                    raise StageError(f"blob sha256:{ref.hexdigest} grew while it was copied")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(dst, view)
                    view = view[written:]
            if total != ref.size or digest.hexdigest() != ref.hexdigest:
                raise StageError(f"blob sha256:{ref.hexdigest} does not match its digest; nothing published")
            os.fsync(dst)
        finally:
            os.close(dst)
    finally:
        os.close(src)
    os.chmod(partial, 0o444)
    os.replace(partial, target)


def _write_file(path: str, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o444)


def _fsync_dir(path: str) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY | _O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def stage(pins, sources, dest: str, headroom: int) -> dict:
    if not os.path.isdir(dest) or os.path.islink(dest):
        raise StageError("the destination is not a directory")
    final = os.path.join(dest, "models")
    if os.path.lexists(final):
        raise StageError("the destination already holds a published store; verify it instead")
    planned = plan_models(pins, sources)
    need, free = required_bytes(planned), free_bytes(dest)
    if free < need + headroom:
        raise StageError(f"insufficient space: {need} bytes required plus {headroom} headroom, {free} free; "
                         "nothing was copied")
    lock = os.path.join(dest, LOCK_NAME)
    try:
        os.close(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        raise StageError("another staging run holds the lock (or an interrupted run left it); refused") from None
    staging = os.path.join(dest, STAGING_PREFIX + uuid.uuid4().hex)
    previous_umask = os.umask(0o022)
    published = False
    try:
        models = os.path.join(staging, "models")
        blobs_dir = os.path.join(models, "blobs")
        os.makedirs(blobs_dir, 0o755)
        copied = set()
        for model in planned:
            for ref in model.blobs:
                if ref.hexdigest not in copied:
                    _copy_blob(sources[model.pin.source], ref, blobs_dir)
                    copied.add(ref.hexdigest)
        for model in planned:
            manifest_dir = os.path.join(models, *manifest_parts(model.pin.tag)[:-1])
            os.makedirs(manifest_dir, 0o755, exist_ok=True)
            _write_file(os.path.join(manifest_dir, manifest_parts(model.pin.tag)[-1]), model.manifest)
        # Re-hash what was written before anything is published.
        check = verify_store(pins, models, quick=False)
        if check["result"] != "VERIFIED":
            raise StageError("the staged copy did not verify; nothing published")
        for directory, _subdirs, _files in os.walk(models):
            _fsync_dir(directory)
        if os.path.lexists(final):
            raise StageError("a store was published meanwhile; nothing published")
        os.rename(models, final)  # the single publishing step
        published = True
        _fsync_dir(dest)
        report = {"result": "STAGED", "models": _model_report(planned), "required_bytes": need,
                  "headroom_bytes": headroom, "free_bytes_before": free, "store": final}
        partial_report = os.path.join(staging, REPORT_NAME)
        _write_file(partial_report, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
        os.replace(partial_report, os.path.join(dest, REPORT_NAME))
        os.rmdir(staging)
        return report
    except BaseException as exc:
        # Remove only the staging directory this run created.
        if os.path.basename(staging).startswith(STAGING_PREFIX) and os.path.isdir(staging):
            shutil.rmtree(staging, onerror=_make_writable_and_retry)
        if published and isinstance(exc, Exception):
            raise StageError("the verified store WAS published, but its report could not be written; "
                             "run verify on it") from None
        raise
    finally:
        os.umask(previous_umask)
        os.remove(lock)


def _make_writable_and_retry(func, path, _exc):
    os.chmod(path, 0o644)
    func(path)


# --- verify --------------------------------------------------------------------------------


def verify_store(pins, store: str, quick: bool = False) -> dict:
    """Read-only check of a published (or staged) store against the pins."""
    expected_files = set()
    report = []
    for pin in pins:
        try:
            manifest = _read_small(store, manifest_parts(pin.tag), MAX_MANIFEST_BYTES)
        except OSError:
            raise StageError(f"{pin.tag}: the manifest is missing from the store") from None
        if hashlib.sha256(manifest).hexdigest() != pin.manifest_digest:
            raise StageError(f"{pin.tag}: the stored manifest differs from the pinned digest")
        expected_files.add(os.path.join(*manifest_parts(pin.tag)))
        refs = _blob_refs(pin.tag, manifest)
        for ref in refs:
            expected_files.add(os.path.join(*blob_parts(ref.hexdigest)))
            try:
                fd = _open_regular(store, blob_parts(ref.hexdigest))
            except OSError:
                raise StageError(f"{pin.tag}: blob sha256:{ref.hexdigest} is missing from the store") from None
            try:
                if os.fstat(fd).st_size != ref.size:
                    raise StageError(f"{pin.tag}: stored blob sha256:{ref.hexdigest} has the wrong size")
                if not quick and _hash_fd(fd, ref.size) != ref.hexdigest:
                    raise StageError(f"{pin.tag}: stored blob sha256:{ref.hexdigest} does not match its digest")
            finally:
                os.close(fd)
        report.append({"tag": pin.tag, "manifest_digest": pin.manifest_digest, "blobs": len(refs),
                       "bytes": sum(r.size for r in refs)})
    found = set()
    for directory, subdirs, files in os.walk(store, followlinks=False):
        for name in subdirs:
            if os.path.islink(os.path.join(directory, name)):
                raise StageError("the store contains a symbolic link")
        for name in files:
            found.add(os.path.relpath(os.path.join(directory, name), store))
    extra = sorted(found - expected_files)
    if extra:
        raise StageError(f"the store holds {len(extra)} file(s) that no pinned manifest references")
    return {"result": "VERIFIED", "mode": "sizes-only" if quick else "full-sha256", "models": report,
            "files": len(found)}


# --- command line --------------------------------------------------------------------------


def _sources(values: Optional[List[str]]) -> Dict[str, str]:
    sources = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not _SOURCE.match(name) or not path or name in sources:
            raise StageError("each --source must be a unique NAME=DIR")
        if not os.path.isdir(path):
            raise StageError(f"source '{name}' is not a directory")
        sources[name] = path
    return sources


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("plan", "stage", "verify"))
    parser.add_argument("--pins", required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--dest")
    parser.add_argument("--store")
    parser.add_argument("--headroom-gib", type=int, default=DEFAULT_HEADROOM_GIB)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    try:
        pins = load_pins(args.pins)
        if args.command == "verify":
            if not args.store or args.source or args.dest:
                parser.error("verify takes --store only")
            report = verify_store(pins, args.store, quick=args.quick)
        else:
            if not args.dest or args.store or args.quick:
                parser.error(f"{args.command} takes --source and --dest")
            if args.headroom_gib < 0:
                parser.error("--headroom-gib must not be negative")
            sources = _sources(args.source)
            headroom = args.headroom_gib * GIB
            report = (plan if args.command == "plan" else stage)(pins, sources, args.dest, headroom)
    except StageError as exc:
        print(json.dumps({"result": "REFUSED", "reason": str(exc)}, indent=2))
        return 3
    except OSError as exc:
        print(json.dumps({"result": "FAILED", "reason": f"{type(exc).__name__} (errno {exc.errno})"}, indent=2))
        return 3
    print(json.dumps(report, indent=2))
    return 0 if report["result"] in ("PLAN_OK", "STAGED", "VERIFIED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
