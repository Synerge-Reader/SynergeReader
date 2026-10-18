import hashlib
import json

from extractor_service.model_check import verify


def _lock(tmp_path, entries):
    path = tmp_path / "MODEL_LOCK.json"
    path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    return str(path)


def test_verify_accepts_locked_bytes_and_reports_every_problem(tmp_path):
    root = tmp_path / "models"
    (root / "tessdata").mkdir(parents=True)
    good = b"weights"
    (root / "tessdata" / "eng.traineddata").write_bytes(good)
    (root / "changed.bin").write_bytes(b"other!!")
    entries = [
        {"path": "tessdata/eng.traineddata", "bytes": len(good), "sha256": hashlib.sha256(good).hexdigest()},
        {"path": "changed.bin", "bytes": 7, "sha256": hashlib.sha256(b"origin!").hexdigest()},
        {"path": "missing.bin", "bytes": 1, "sha256": "0" * 64},
        {"path": "../escape.bin", "bytes": 1, "sha256": "0" * 64},
    ]
    problems = verify(str(root), _lock(tmp_path, entries))
    assert len(problems) == 3
    assert any(p.startswith("changed.bin: sha256") for p in problems)
    assert "missing.bin: missing" in problems
    assert "../escape.bin: unsafe path in lock" in problems
    assert verify(str(root), _lock(tmp_path, entries[:1])) == []


def test_repository_model_lock_is_well_formed():
    import pathlib
    lock = json.loads((pathlib.Path(__file__).resolve().parents[1] / "MODEL_LOCK.json").read_text(encoding="utf-8"))
    paths = [f["path"] for f in lock["files"]]
    assert len(paths) == len(set(paths)) == 10
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in lock["files"])
    revisions = {f["repository"]: f["revision"] for f in lock["files"]}
    assert revisions["docling-project/docling-layout-heron"] == "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
    assert revisions["docling-project/docling-models"] == "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
