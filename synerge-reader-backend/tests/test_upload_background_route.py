"""Background upload jobs through the real /upload route (LIVE-EXTRACTION-20261017).

MANDATORY PATCHING RULE: every TestClient context patches main._initialize_application
before entry. No database, worker or model is contacted: the ingestion service and
caller identity are doubles, and the job registry uses an inert thread factory so jobs
run deterministically on the test thread.
"""

import contextlib
import importlib
import os
import sys

import dotenv
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_ingestion import BatchIngestionResult, IngestionResult, ResultStatus  # noqa: E402
from upload_jobs import UploadJobRegistry, UploadJobSettings  # noqa: E402

_EMBEDDING_PROFILE_KEYS = (
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION",
    "EMBEDDING_QUERY_PREFIX", "EMBEDDING_DOCUMENT_PREFIX", "EMBEDDING_PROFILE_UNVERIFIED_ACK",
)
_TOKENS = {"tok-alice": "alice-id", "tok-bob": "bob-id"}


@contextlib.contextmanager
def _imported_main():
    previous = sys.modules.pop("main", None)
    try:
        yield importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
        if previous is not None:
            sys.modules["main"] = previous


class _InertThread:
    def __init__(self, target, name, daemon):
        pass

    def start(self):
        pass


class _Service:
    def __init__(self, record, mode):
        self.record, self.mode = record, mode

    def ingest_batch(self, uploads):
        uploads = list(uploads)
        self.record.append((self.mode, [(u.filename, u.content, u.uploader_id) for u in uploads]))
        return BatchIngestionResult(
            results=tuple(IngestionResult(filename=u.filename, status=ResultStatus.INDEXED, document_id=900 + i,
                                          chunks_count=1, warnings=("Page 1 had no usable embedded text; its text "
                                                                    "was recovered with OCR.",))
                          for i, u in enumerate(uploads)),
            http_status=200,
        )


def _resolve(token):
    from fastapi import HTTPException
    if token not in _TOKENS:
        raise HTTPException(401, "Unauthorized")
    return _TOKENS[token]


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for key in _EMBEDDING_PROFILE_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("EXTRACTOR_ENABLE_TABLES", raising=False)
    with _imported_main() as main:
        record = []
        monkeypatch.setattr(main, "_initialize_application", lambda: None)
        monkeypatch.setattr(main, "connect_to_postgres", lambda: (_ for _ in ()).throw(AssertionError("no DB")))
        monkeypatch.setattr(main, "_resolve_caller_id", _resolve)
        monkeypatch.setattr(main, "_build_ingestion_service", lambda mode="standard": _Service(record, mode))
        registry = UploadJobRegistry(UploadJobSettings(max_per_owner=1), thread_factory=_InertThread, log=lambda m: None)
        monkeypatch.setattr(main, "_UPLOAD_JOBS", registry)
        client = TestClient(main.app)
        try:
            with client:
                yield client, registry, record
        finally:
            client.close()


def _post(client, token="tok-alice", **form):
    data = {"auth_token": token, **form}
    return client.post("/upload", data=data, files=[("files", ("scan.pdf", b"%PDF-1.4 original bytes", "application/pdf"))])


def test_background_upload_returns_202_and_the_owner_gets_the_result(harness, monkeypatch):
    client, registry, record = harness
    monkeypatch.setenv("EXTRACTOR_ENABLE_TABLES", "1")
    response = _post(client, processing="background", extraction="enhanced")
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and body["extraction"] == "enhanced" and record == []
    job_id = body["job_id"]
    status = client.get(f"/upload/jobs/{job_id}", headers={"Authorization": "Bearer tok-alice"})
    assert status.status_code == 200 and status.json()["status"] == "queued"
    assert registry.run_next_for_test()
    done = client.get(f"/upload/jobs/{job_id}", headers={"Authorization": "Bearer tok-alice"}).json()
    assert done["status"] == "completed" and done["http_status"] == 200
    assert done["result"]["results"][0]["status"] == "indexed"
    assert record == [("enhanced", [("scan.pdf", b"%PDF-1.4 original bytes", "alice-id")])]


@pytest.mark.parametrize("processing", ["background", "sync"])
def test_table_extraction_is_refused_until_an_operator_enables_it(harness, processing):
    client, registry, record = harness
    response = _post(client, processing=processing, extraction="enhanced")
    assert response.status_code == 422
    assert response.json()["detail"] == "Table extraction is not available on this server yet."
    assert record == [] and registry.snapshot()["queued"] == 0


def test_standard_ocr_uploads_do_not_need_the_table_switch(harness):
    client, registry, record = harness
    assert _post(client, processing="background").status_code == 202


def test_another_owner_and_anonymous_callers_cannot_see_the_job(harness):
    client, registry, _ = harness
    job_id = _post(client, processing="background").json()["job_id"]
    other = client.get(f"/upload/jobs/{job_id}", headers={"Authorization": "Bearer tok-bob"})
    assert other.status_code == 404
    anonymous = client.get(f"/upload/jobs/{job_id}")
    assert anonymous.status_code == 401


def test_owner_limit_is_a_429_with_retry_after(harness):
    client, _, _ = harness
    assert _post(client, processing="background").status_code == 202
    refused = _post(client, processing="background")
    assert refused.status_code == 429 and refused.headers.get("Retry-After") == "60"


def test_invalid_modes_are_refused_before_any_work(harness):
    client, _, record = harness
    assert _post(client, processing="later").status_code == 422
    assert _post(client, extraction="magic").status_code == 422
    assert record == []


def test_sync_default_still_returns_the_envelope_and_uses_standard_mode(harness):
    client, _, record = harness
    response = _post(client)
    assert response.status_code == 200 and set(response.json()) == {"results", "summary", "http_status"}
    assert record[0][0] == "standard"
