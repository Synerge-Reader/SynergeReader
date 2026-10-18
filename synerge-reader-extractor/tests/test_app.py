import threading

import pytest
from fastapi.testclient import TestClient

from extractor_service import SERVICE_SCHEMA
from extractor_service.app import BadRequest, create_app, parse_page_spec
from extractor_service.config import ServiceSettings
from extractor_service.supervisor import JobFailed, JobStats, WorkerCrashed, WorkerMemoryExceeded, WorkerTimeout

TOKEN = "test-token-not-a-secret"
PDF = b"%PDF-1.4\n" + b"0" * 64
AUTH = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/pdf"}


class FakeSupervisor:
    def __init__(self, outcome=None):
        self.outcome = outcome
        self.jobs = []
        self.timeouts = []
        self.shut = False

    def run(self, job, timeout_s):
        self.jobs.append(job)
        self.timeouts.append(timeout_s)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if callable(self.outcome):
            return self.outcome(job)
        pages = [{"page_number": p, "method": "ocr", "text": f"page {p}", "mean_confidence": 91.0, "tables": [],
                  "warnings": []} for p in job["ocr_pages"]]
        pages += [{"page_number": p, "method": "table", "text": "", "mean_confidence": None, "tables": [],
                   "warnings": []} for p in job["table_pages"]]
        return ({"pages": pages, "engine": {"tesseract": "5"}, "timings": {}},
                JobStats(elapsed_s=0.5, peak_rss_bytes=1024, worker_pid=1, worker_respawned=False))

    def shutdown(self):
        self.shut = True


def _client(sup=None, logs=None, **overrides):
    settings = ServiceSettings(token=TOKEN, **overrides)
    app = create_app(settings, supervisor=sup or FakeSupervisor(), log=(logs.append if logs is not None else lambda s: None))
    return TestClient(app)


@pytest.mark.parametrize("raw,expected", [(None, []), ("", []), ("3", [3]), ("1,3-5", [1, 3, 4, 5]), (" 2 , 2 ", [2])])
def test_page_spec_parses(raw, expected):
    assert parse_page_spec(raw, 40) == expected


@pytest.mark.parametrize("raw,status", [("a", 400), ("0", 400), ("3-1", 400), ("1,,2", 400), ("-2", 400),
                                        ("1-1000000000", 413), ("1-41", 413), ("x" * 3000, 400)])
def test_page_spec_rejects(raw, status):
    with pytest.raises(BadRequest) as info:
        parse_page_spec(raw, 40)
    assert info.value.status == status


def test_token_is_required_to_start():
    with pytest.raises(RuntimeError):
        create_app(ServiceSettings(token=""), supervisor=FakeSupervisor())


def test_settings_repr_hides_token():
    assert TOKEN not in repr(ServiceSettings(token=TOKEN))
    assert "token" not in ServiceSettings(token=TOKEN).engine_config()


def test_healthz_is_open_and_minimal():
    with _client() as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "schema": SERVICE_SCHEMA}


@pytest.mark.parametrize("header", [None, "Bearer wrong", f"bearer {TOKEN}", TOKEN, f"Bearer {TOKEN} "])
def test_extract_requires_exact_bearer_token(header):
    sup = FakeSupervisor()
    headers = {"Content-Type": "application/pdf"}
    if header is not None:
        headers["Authorization"] = header
    with _client(sup) as c:
        r = c.post("/v1/extract?ocr_pages=1", content=PDF, headers=headers)
    assert r.status_code == 401
    assert sup.jobs == []


def test_extract_returns_schema_for_requested_pages_and_logs_no_content():
    sup, logs = FakeSupervisor(), []
    with _client(sup, logs) as c:
        r = c.post("/v1/extract?ocr_pages=1,3&table_pages=2&timeout_s=120", content=PDF, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["schema"] == SERVICE_SCHEMA
    assert [p["page_number"] for p in body["pages"]] == [1, 3, 2]
    assert sup.jobs[0]["ocr_pages"] == [1, 3] and sup.jobs[0]["table_pages"] == [2]
    assert sup.jobs[0]["pdf_bytes"] == PDF
    assert 100 < sup.timeouts[0] <= 120
    assert len(logs) == 1 and "page 1" not in logs[0] and TOKEN not in logs[0]
    assert sup.shut is True


def test_timeout_is_capped_by_settings():
    sup = FakeSupervisor()
    with _client(sup, max_job_timeout_s=60) as c:
        assert c.post("/v1/extract?ocr_pages=1&timeout_s=99999", content=PDF, headers=AUTH).status_code == 200
    assert sup.timeouts[0] <= 60


@pytest.mark.parametrize("query,status", [
    ("", 400),
    ("ocr_pages=1&table_pages=1", 400),
    ("ocr_pages=abc", 400),
    ("ocr_pages=1&timeout_s=-5", 400),
    ("table_pages=1-11", 413),
])
def test_bad_queries_are_rejected_before_running(query, status):
    sup = FakeSupervisor()
    with _client(sup) as c:
        r = c.post(f"/v1/extract?{query}", content=PDF, headers=AUTH)
    assert r.status_code == status
    assert sup.jobs == []


def test_non_pdf_and_oversized_bodies_are_rejected():
    sup = FakeSupervisor()
    with _client(sup, max_body_bytes=2048) as c:
        assert c.post("/v1/extract?ocr_pages=1", content=b"hello", headers=AUTH).status_code == 415
        assert c.post("/v1/extract?ocr_pages=1", content=PDF,
                      headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "text/plain"}).status_code == 415
        assert c.post("/v1/extract?ocr_pages=1", content=b"%PDF-" + b"0" * 4096, headers=AUTH).status_code == 413
    assert sup.jobs == []


@pytest.mark.parametrize("exc,status", [
    (WorkerTimeout("t"), 504),
    (WorkerMemoryExceeded("m"), 503),
    (WorkerCrashed("c"), 503),
    (JobFailed("input", "a requested page is outside the PDF"), 422),
    (JobFailed("engine", "RuntimeError"), 500),
])
def test_worker_failures_map_to_status(exc, status):
    logs = []
    with _client(FakeSupervisor(exc), logs) as c:
        r = c.post("/v1/extract?ocr_pages=1", content=PDF, headers=AUTH)
    assert r.status_code == status
    if status == 422:
        assert r.json()["error"] == "a requested page is outside the PDF"
    if status == 500:
        assert "RuntimeError" not in r.text
    assert len(logs) == 1


def test_busy_beyond_waiting_cap_answers_429():
    release = threading.Event()
    started = threading.Event()

    def slow(job):
        started.set()
        release.wait(10)
        return ({"pages": [], "engine": {}, "timings": {}},
                JobStats(elapsed_s=0.1, peak_rss_bytes=None, worker_pid=1, worker_respawned=False))

    sup = FakeSupervisor(slow)
    results = []
    with _client(sup, max_waiting=0) as c:
        t = threading.Thread(target=lambda: results.append(
            c.post("/v1/extract?ocr_pages=1", content=PDF, headers=AUTH).status_code))
        t.start()
        assert started.wait(10)
        r = c.post("/v1/extract?ocr_pages=2", content=PDF, headers=AUTH)
        release.set()
        t.join(10)
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "30"
    assert results == [200]
