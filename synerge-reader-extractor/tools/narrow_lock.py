"""Narrow a ``uv pip compile --generate-hashes`` lock to the single wheel pip will install on one target.

uv lists the hashes of every published file for each pinned version. This keeps only the hash of
the best-matching wheel for CPython 3.11 on manylinux (glibc <= 2.28) for one architecture, and
writes a JSON manifest (filename, URL, SHA-256, bytes) next to it. Files are identified from the
index metadata; nothing is installed. Run on the build-preparation machine, not in the image:

    python tools/narrow_lock.py <uv-lock> <arch: x86_64|aarch64> <out-lock> <out-manifest.json>
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import re
import sys
import urllib.request
from urllib.parse import unquote, urljoin

from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

PYTORCH_CPU = "https://download.pytorch.org/whl/cpu/"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"


def target_tags(arch: str) -> list[Tag]:
    platforms = [f"manylinux_2_{minor}_{arch}" for minor in range(28, 16, -1)] + [f"manylinux2014_{arch}"]
    tags = list(cpython_tags(python_version=(3, 11), abis=["cp311"], platforms=platforms))
    tags += list(compatible_tags(python_version=(3, 11), interpreter="cp311", platforms=platforms))
    return tags


def parse_lock(path: str) -> list[tuple[str, str, str, set[str]]]:
    """-> [(raw_line_name, name, version, {hashes})]"""
    entries = []
    current = None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._\-\[\],]*)==([^\s\\;]+)", line)
        if m:
            current = [m.group(1), canonicalize_name(re.sub(r"\[.*\]", "", m.group(1))), m.group(2), set()]
            entries.append(current)
        for h in re.findall(r"--hash=sha256:([0-9a-f]{64})", line):
            current[3].add(h)
    return [tuple(e) for e in entries]


class _Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "synergereader-lock-narrower"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def index_files(name: str, version: str, torch_index: bool) -> list[dict]:
    if torch_index:
        parser = _Links()
        parser.feed(_get(PYTORCH_CPU + name + "/").decode("utf-8"))
        out = []
        for href in parser.links:
            url, _, frag = href.partition("#sha256=")
            filename = unquote(url.rsplit("/", 1)[-1])
            if not filename.endswith(".whl") or not frag:
                continue
            _n, ver, _b, _t = parse_wheel_filename(filename)
            if str(ver) == version:
                out.append({"filename": filename, "url": urljoin(PYTORCH_CPU + name + "/", url), "sha256": frag,
                            "size": None})
        return out
    data = json.loads(_get(PYPI_JSON.format(name=name, version=version)))
    return [{"filename": u["filename"], "url": u["url"], "sha256": u["digests"]["sha256"], "size": u.get("size")}
            for u in data["urls"] if u["filename"].endswith(".whl")]


def choose(files: list[dict], tags: list[Tag], allowed: set[str]) -> dict:
    rank = {t: i for i, t in enumerate(tags)}
    best = None
    for f in files:
        if f["sha256"] not in allowed:
            continue
        _n, _v, _b, ftags = parse_wheel_filename(f["filename"])
        score = min((rank[t] for t in ftags if t in rank), default=None)
        if score is not None and (best is None or score < best[0]):
            best = (score, f)
    if best is None:
        raise SystemExit(f"no compatible wheel among the locked hashes: {[f['filename'] for f in files]}")
    return best[1]


def main(argv: list[str]) -> int:
    src, arch, out_lock, out_manifest = argv[1:5]
    tags = target_tags(arch)
    entries = parse_lock(src)
    lines = [f"# CPython 3.11 / manylinux_2_28 {arch}; one wheel per pin; install with:",
             "#   pip install --no-deps --require-hashes --only-binary=:all: -r <this file>",
             f"# narrowed from a uv pip compile lock by tools/narrow_lock.py; source lock sha256 "
             f"{hashlib.sha256(open(src, 'rb').read()).hexdigest()}"]
    manifest = []
    for raw, name, version, hashes in entries:
        torch_index = name in ("torch", "torchvision")
        chosen = choose(index_files(name, version, torch_index), tags, hashes)
        lines.append(f"{raw}=={version} \\\n    --hash=sha256:{chosen['sha256']}")
        manifest.append({"name": name, "version": version, **chosen,
                         "index": "pytorch-cpu" if torch_index else "pypi"})
    with open(out_lock, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(out_manifest, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"target": {"python": "cp311", "platform": f"manylinux_2_28_{arch}"}, "wheels": manifest}, fh,
                  indent=1)
        fh.write("\n")
    print(f"{len(manifest)} wheels -> {out_lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
