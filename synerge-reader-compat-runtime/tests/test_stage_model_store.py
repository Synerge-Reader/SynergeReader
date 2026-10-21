"""Contracts for stage_model_store.py, on synthetic Ollama stores in a temp dir.

What these prove: only the pinned manifests and the blobs they reference are
copied (config and every layer, shared blobs once); a manifest or blob that
differs from its digest or declared size, a missing blob, a symbolic link or a
non-regular file is refused before anything is published; insufficient space
refuses before copying; an existing store or a held lock refuses; a failure
leaves no partial store and no staging directory; the sources are never
changed; a published store verifies and an extra or altered file fails
verification. No real model, network or container is used.
"""

import hashlib
import json
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stage_model_store as sms  # noqa: E402


def _blob(store, data):
    hexd = hashlib.sha256(data).hexdigest()
    blobs = os.path.join(store, "blobs")
    os.makedirs(blobs, exist_ok=True)
    with open(os.path.join(blobs, f"sha256-{hexd}"), "wb") as fh:
        fh.write(data)
    return hexd


def _model(store, tag, layers):
    """Write a manifest for ``tag`` referencing a config and ``layers``;
    returns the manifest digest."""
    config = _blob(store, json.dumps({"model_format": "gguf", "tag": tag}).encode())
    refs = [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + _blob(store, data),
             "size": len(data)} for data in layers]
    manifest = json.dumps({"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                           "config": {"digest": "sha256:" + config, "size": len(json.dumps(
                               {"model_format": "gguf", "tag": tag}).encode())},
                           "layers": refs}).encode()
    name, version = tag.split(":")
    directory = os.path.join(store, "manifests", "registry.ollama.ai", "library", name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, version), "wb") as fh:
        fh.write(manifest)
    return hashlib.sha256(manifest).hexdigest()


def _snapshot(root):
    """(relative path, sha256, mtime) for every file under root."""
    out = []
    for directory, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.join(directory, name)
            out.append((os.path.relpath(path, root), hashlib.sha256(open(path, "rb").read()).hexdigest(),
                        os.stat(path).st_mtime_ns))
    return sorted(out)


@pytest.fixture
def world(tmp_path):
    qwen, mistral, dest = tmp_path / "qwen" / "models", tmp_path / "mistral", tmp_path / "dest"
    for directory in (qwen, mistral, dest):
        directory.mkdir(parents=True)
    shared = b"SHARED-LICENCE-LAYER"
    digests = {
        "qwen3.8:27b": _model(str(qwen), "qwen3.8:27b", [b"Q" * 5000, shared, b"template"]),
        "mistral-medium-3.5:128b": _model(str(mistral), "mistral-medium-3.5:128b", [b"M" * 9000, shared]),
    }
    # Unpinned models and stray files in the sources must never be copied.
    _model(str(qwen), "qwen3.8:other", [b"UNPINNED-WEIGHTS"])
    _model(str(mistral), "gemma4:31b", [b"GEMMA-WEIGHTS"])
    _blob(str(mistral), b"orphan blob")
    pins_path = tmp_path / "PINS.json"
    pins_path.write_text(json.dumps({"models": [
        {"tag": "qwen3.8:27b", "manifest_digest": digests["qwen3.8:27b"], "source": "qwen"},
        {"tag": "mistral-medium-3.5:128b", "manifest_digest": digests["mistral-medium-3.5:128b"],
         "source": "mistral"},
    ]}))
    sources = {"qwen": str(qwen), "mistral": str(mistral)}
    return {"pins": str(pins_path), "sources": sources, "dest": str(dest), "digests": digests, "tmp": tmp_path}


def _args(world, command, *extra):
    args = [command, "--pins", world["pins"], "--dest", world["dest"], "--headroom-gib", "0"]
    for name, path in world["sources"].items():
        args += ["--source", f"{name}={path}"]
    return args + list(extra)


def _run(capsys, args):
    code = sms.main(args)
    return code, json.loads(capsys.readouterr().out)


def test_plan_measures_without_writing(world, capsys):
    before = {name: _snapshot(path) for name, path in world["sources"].items()}
    code, report = _run(capsys, _args(world, "plan"))
    assert code == 0 and report["result"] == "PLAN_OK"
    assert [m["tag"] for m in report["models"]] == ["qwen3.8:27b", "mistral-medium-3.5:128b"]
    assert report["published"] is False and os.listdir(world["dest"]) == []
    assert {name: _snapshot(path) for name, path in world["sources"].items()} == before


def test_stage_copies_only_pinned_content_verifies_and_publishes_once(world, capsys):
    before = {name: _snapshot(path) for name, path in world["sources"].items()}
    code, report = _run(capsys, _args(world, "stage"))
    assert code == 0 and report["result"] == "STAGED"
    store = os.path.join(world["dest"], "models")
    assert sorted(os.listdir(world["dest"])) == ["STAGED.json", "models"], "no staging dir or lock left behind"
    manifests = sorted(os.path.relpath(os.path.join(d, f), store) for d, _, fs in os.walk(store) for f in fs
                       if "manifests" in d)
    assert manifests == [os.path.join("manifests", "registry.ollama.ai", "library", "mistral-medium-3.5", "128b"),
                         os.path.join("manifests", "registry.ollama.ai", "library", "qwen3.8", "27b")]
    blobs = os.listdir(os.path.join(store, "blobs"))
    # 2 configs + qwen's 3 layers + mistral's 2 layers, the shared layer once.
    assert len(blobs) == 6 and not any(b.endswith(".partial") for b in blobs)
    staged = {f for _, _, fs in os.walk(store) for f in fs}
    unpinned = hashlib.sha256(b"UNPINNED-WEIGHTS").hexdigest()
    assert f"sha256-{unpinned}" not in staged and f"sha256-{hashlib.sha256(b'orphan blob').hexdigest()}" not in staged
    for directory, _dirs, files in os.walk(store):
        for name in files:
            assert not os.stat(os.path.join(directory, name)).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert {name: _snapshot(path) for name, path in world["sources"].items()} == before, "sources untouched"
    assert json.load(open(os.path.join(world["dest"], "STAGED.json")))["result"] == "STAGED"
    code, verified = _run(capsys, ["verify", "--pins", world["pins"], "--store", store])
    assert code == 0 and verified["result"] == "VERIFIED" and verified["mode"] == "full-sha256"
    assert verified["files"] == 8


def test_a_second_stage_is_refused_and_leaves_the_published_store(world, capsys):
    assert _run(capsys, _args(world, "stage"))[0] == 0
    before = _snapshot(os.path.join(world["dest"], "models"))
    code, report = _run(capsys, _args(world, "stage"))
    assert code == 3 and "already holds a published store" in report["reason"]
    assert _snapshot(os.path.join(world["dest"], "models")) == before


def _tamper_manifest(world):
    path = os.path.join(world["sources"]["qwen"], "manifests", "registry.ollama.ai", "library", "qwen3.8", "27b")
    with open(path, "ab") as fh:
        fh.write(b" ")


def _corrupt_blob(world):
    data = b"Q" * 5000
    path = os.path.join(world["sources"]["qwen"], "blobs", f"sha256-{hashlib.sha256(data).hexdigest()}")
    os.chmod(path, 0o644)
    with open(path, "wb") as fh:
        fh.write(b"X" * 5000)  # same size, different bytes


def _shorten_blob(world):
    data = b"M" * 9000
    path = os.path.join(world["sources"]["mistral"], "blobs", f"sha256-{hashlib.sha256(data).hexdigest()}")
    with open(path, "wb") as fh:
        fh.write(b"M" * 10)


def _remove_blob(world):
    data = b"template"
    os.remove(os.path.join(world["sources"]["qwen"], "blobs", f"sha256-{hashlib.sha256(data).hexdigest()}"))


@pytest.mark.parametrize(
    "damage, reason, at_plan",
    [
        (_tamper_manifest, "differs from the pinned digest", True),
        (_shorten_blob, "different size than declared", True),
        (_remove_blob, "is missing from source", True),
        (_corrupt_blob, "does not match its digest", False),
    ],
    ids=["manifest", "blob-size", "blob-missing", "blob-bytes"],
)
def test_damaged_sources_are_refused_and_nothing_is_published(world, capsys, damage, reason, at_plan):
    damage(world)
    code, report = _run(capsys, _args(world, "plan"))
    if at_plan:
        assert code == 3 and reason in report["reason"]
    code, report = _run(capsys, _args(world, "stage"))
    assert code == 3 and report["result"] == "REFUSED" and reason in report["reason"]
    assert os.listdir(world["dest"]) == [], "no partial store, staging directory or lock remains"


def test_insufficient_space_refuses_before_copying(world, capsys, monkeypatch):
    monkeypatch.setattr(sms, "free_bytes", lambda path: 1000)
    code, report = _run(capsys, _args(world, "plan"))
    assert code == 3 and report["result"] == "INSUFFICIENT_SPACE" and report["required_bytes"] > 1000
    code, report = _run(capsys, _args(world, "stage"))
    assert code == 3 and "insufficient space" in report["reason"] and "nothing was copied" in report["reason"]
    assert os.listdir(world["dest"]) == []


def test_headroom_counts_toward_the_space_check(world, capsys, monkeypatch):
    planned = sms.plan_models(sms.load_pins(world["pins"]), world["sources"])
    need = sms.required_bytes(planned)
    monkeypatch.setattr(sms, "free_bytes", lambda path: need + sms.GIB - 1)
    code, report = _run(capsys, _args(world, "stage")[:-2] + ["--headroom-gib", "1"]
                        + [a for a in _args(world, "stage")[-2:]])
    assert code == 3 and "insufficient space" in report["reason"]


def test_a_held_lock_refuses(world, capsys):
    open(os.path.join(world["dest"], sms.LOCK_NAME), "w").close()
    code, report = _run(capsys, _args(world, "stage"))
    assert code == 3 and "lock" in report["reason"]
    assert os.listdir(world["dest"]) == [sms.LOCK_NAME], "a lock it did not take is never removed"


def test_a_failure_while_copying_removes_only_its_own_staging_directory(world, capsys, monkeypatch):
    other = os.path.join(world["dest"], "unrelated")
    os.mkdir(other)
    calls = []
    real = sms._copy_blob

    def failing(root, ref, blobs_dir):
        calls.append(ref.hexdigest)
        if len(calls) == 3:
            raise OSError(28, "No space left on device")
        return real(root, ref, blobs_dir)

    monkeypatch.setattr(sms, "_copy_blob", failing)
    code, report = _run(capsys, _args(world, "stage"))
    assert code == 3 and report["result"] == "FAILED" and "errno 28" in report["reason"]
    assert os.listdir(world["dest"]) == ["unrelated"]


def test_a_symbolic_link_in_a_source_is_refused(world, capsys):
    data = b"template"
    path = os.path.join(world["sources"]["qwen"], "blobs", f"sha256-{hashlib.sha256(data).hexdigest()}")
    target = os.path.join(str(world["tmp"]), "elsewhere")
    with open(target, "wb") as fh:
        fh.write(data)
    os.remove(path)
    try:
        os.symlink(target, path)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available to this account")
    code, report = _run(capsys, _args(world, "stage"))
    assert code == 3
    assert os.listdir(world["dest"]) == []


def test_only_pinned_sources_and_valid_pins_are_accepted(world, capsys, tmp_path):
    code, report = _run(capsys, ["plan", "--pins", world["pins"], "--dest", world["dest"],
                                 "--source", f"qwen={world['sources']['qwen']}"])
    assert code == 3 and "mistral" in report["reason"]
    for bad in ({"models": []}, {"models": [{"tag": "Qwen:27b", "manifest_digest": "a" * 64, "source": "q"}]},
                {"models": [{"tag": "q:1", "manifest_digest": "A" * 64, "source": "q"}]},
                {"models": [{"tag": "ns/q:1", "manifest_digest": "a" * 64, "source": "q"}]},
                {"models": [{"tag": "q:1", "manifest_digest": "a" * 64, "source": "q"}] * 2}):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        code, report = _run(capsys, ["plan", "--pins", str(path), "--dest", world["dest"]])
        assert code == 3 and report["result"] == "REFUSED"


def test_verify_rejects_extra_or_altered_files_in_a_published_store(world, capsys):
    assert _run(capsys, _args(world, "stage"))[0] == 0
    store = os.path.join(world["dest"], "models")
    extra = os.path.join(store, "blobs", "sha256-" + "0" * 64)
    with open(extra, "wb") as fh:
        fh.write(b"x")
    code, report = _run(capsys, ["verify", "--pins", world["pins"], "--store", store])
    assert code == 3 and "no pinned manifest references" in report["reason"]
    os.remove(extra)
    blob = os.path.join(store, "blobs", f"sha256-{hashlib.sha256(b'Q' * 5000).hexdigest()}")
    os.chmod(blob, 0o644)
    with open(blob, "wb") as fh:
        fh.write(b"Z" * 5000)
    code, report = _run(capsys, ["verify", "--pins", world["pins"], "--store", store, "--quick"])
    assert code == 0 and report["mode"] == "sizes-only", "the quick check reads sizes only"
    code, report = _run(capsys, ["verify", "--pins", world["pins"], "--store", store])
    assert code == 3 and "does not match its digest" in report["reason"]


def test_the_report_never_contains_file_content(world, capsys):
    code, report = _run(capsys, _args(world, "stage"))
    text = json.dumps(report)
    assert code == 0 and "QQQQ" not in text and "SHARED-LICENCE" not in text and "template" not in text
