"""Private HTTP front for the extraction worker. Run with exactly one uvicorn worker.

POST /v1/extract?ocr_pages=1,3-4&table_pages=2&timeout_s=300
    Authorization: Bearer <EXTRACTOR_TOKEN>
    Content-Type: application/pdf
    body: the PDF bytes

Returns ``synergereader.extractor.v1`` JSON for exactly the requested pages. One job runs
at a time; at most ``max_waiting`` more may wait, beyond that the answer is 429. A job that
exceeds its deadline answers 504, a worker that breaches its memory limit or crashes
answers 503; neither is retried here. GET /healthz needs no token and reveals no settings.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import threading
import time
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import SERVICE_SCHEMA
from .config import ServiceSettings
from .supervisor import JobFailed, Supervisor, WorkerCrashed, WorkerMemoryExceeded, WorkerTimeout

MAX_PAGE_SPEC_CHARS = 2000


class BadRequest(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def parse_page_spec(raw: Optional[str], limit: int) -> list[int]:
    """'1,3-5' -> [1, 3, 4, 5]. Rejects malformed input and anything beyond ``limit`` pages."""
    if raw is None or not raw.strip():
        return []
    if len(raw) > MAX_PAGE_SPEC_CHARS:
        raise BadRequest(400, "page list too long")
    pages: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise BadRequest(400, "malformed page list")
        lo_s, sep, hi_s = part.partition("-")
        if not lo_s.isdigit() or (sep and not hi_s.isdigit()):
            raise BadRequest(400, "malformed page list")
        lo = int(lo_s)
        hi = int(hi_s) if sep else lo
        if lo < 1 or hi < lo:
            raise BadRequest(400, "malformed page list")
        if hi - lo + 1 > limit:
            raise BadRequest(413, "too many pages requested")
        pages.update(range(lo, hi + 1))
        if len(pages) > limit:
            raise BadRequest(413, "too many pages requested")
    return sorted(pages)


def _error(status: int, message: str, headers: Optional[dict] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message}, headers=headers)


def _log_line(**fields) -> str:
    return "extractor " + json.dumps(fields, sort_keys=True, separators=(",", ":"))


def create_app(settings: ServiceSettings, supervisor: Optional[Supervisor] = None,
               log: Callable[[str], None] = print) -> FastAPI:
    if not settings.token:
        raise RuntimeError("EXTRACTOR_TOKEN is required")
    if supervisor is None:
        from .engine import worker_main
        supervisor = Supervisor(target=worker_main, engine_config=settings.engine_config(),
                                rss_limit_bytes=settings.worker_rss_limit_bytes)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            supervisor.shutdown()

    app = FastAPI(title="SynergeReader extractor", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    expected = ("Bearer " + settings.token).encode("utf-8")
    run_slot = asyncio.Semaphore(1)
    admission = threading.Lock()
    state = {"admitted": 0, "jobs": 0}

    def authorized(request: Request) -> bool:
        header = request.headers.get("authorization", "")
        return hmac.compare_digest(header.encode("utf-8", "replace"), expected)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "schema": SERVICE_SCHEMA}

    @app.post("/v1/extract")
    async def extract(request: Request):
        arrived = time.monotonic()
        if not authorized(request):
            return _error(401, "unauthorized")
        try:
            ocr_pages = parse_page_spec(request.query_params.get("ocr_pages"), settings.max_ocr_pages)
            table_pages = parse_page_spec(request.query_params.get("table_pages"), settings.max_table_pages)
            raw_timeout = request.query_params.get("timeout_s", "")
            timeout_s = settings.max_job_timeout_s
            if raw_timeout:
                if not raw_timeout.isdigit():
                    raise BadRequest(400, "timeout_s must be a positive integer")
                timeout_s = max(10, min(int(raw_timeout), settings.max_job_timeout_s))
        except BadRequest as exc:
            return _error(exc.status, exc.message)
        if not ocr_pages and not table_pages:
            return _error(400, "no pages requested")
        if set(ocr_pages) & set(table_pages):
            return _error(400, "a page cannot be requested for both OCR and tables")
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in ("application/pdf", "application/octet-stream"):
            return _error(415, "body must be a PDF")
        declared = request.headers.get("content-length")
        if declared is not None:
            if not declared.isdigit():
                return _error(400, "invalid content-length")
            if int(declared) > settings.max_body_bytes:
                return _error(413, "PDF too large")
        with admission:
            if state["admitted"] >= 1 + settings.max_waiting:
                return _error(429, "extractor busy", headers={"Retry-After": "30"})
            state["admitted"] += 1
        try:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > settings.max_body_bytes:
                    return _error(413, "PDF too large")
            if not body.startswith(b"%PDF-"):
                return _error(415, "body must be a PDF")
            job = {"pdf_bytes": bytes(body), "ocr_pages": ocr_pages, "table_pages": table_pages}
            del body
            async with run_slot:
                started = time.monotonic()
                state["jobs"] += 1
                job_no = state["jobs"]
                # Time spent waiting behind another job counts against the caller's deadline.
                remaining = timeout_s - (started - arrived)
                if remaining < 5:
                    log(_log_line(job=job_no, outcome="expired_in_queue", ocr=len(ocr_pages), tables=len(table_pages),
                                  waited_s=round(started - arrived, 1), timeout_s=timeout_s))
                    return _error(504, "extraction timed out while waiting for the worker")
                try:
                    result, stats = await asyncio.to_thread(supervisor.run, job, float(remaining))
                except WorkerTimeout:
                    log(_log_line(job=job_no, outcome="timeout", ocr=len(ocr_pages), tables=len(table_pages),
                                  elapsed_s=round(time.monotonic() - started, 1), timeout_s=timeout_s))
                    return _error(504, "extraction timed out")
                except WorkerMemoryExceeded:
                    log(_log_line(job=job_no, outcome="memory_limit", ocr=len(ocr_pages), tables=len(table_pages),
                                  elapsed_s=round(time.monotonic() - started, 1)))
                    return _error(503, "extraction exceeded its memory limit")
                except WorkerCrashed:
                    log(_log_line(job=job_no, outcome="crashed", ocr=len(ocr_pages), tables=len(table_pages),
                                  elapsed_s=round(time.monotonic() - started, 1)))
                    return _error(503, "extraction worker failed")
                except JobFailed as exc:
                    status = 422 if exc.kind == "input" else 500
                    log(_log_line(job=job_no, outcome=f"failed_{exc.kind}", ocr=len(ocr_pages),
                                  tables=len(table_pages), elapsed_s=round(time.monotonic() - started, 1)))
                    return _error(status, str(exc) if exc.kind == "input" else "extraction failed")
            payload = {
                "schema": SERVICE_SCHEMA,
                "pages": result["pages"],
                "engine": result.get("engine", {}),
                "stats": {"elapsed_s": round(stats.elapsed_s, 3), "peak_rss_bytes": stats.peak_rss_bytes,
                          "worker_respawned": stats.worker_respawned, "timings": result.get("timings", {})},
            }
            log(_log_line(job=job_no, outcome="ok", ocr=len(ocr_pages), tables=len(table_pages),
                          elapsed_s=round(stats.elapsed_s, 1), peak_rss_mb=(stats.peak_rss_bytes or 0) // (1024 * 1024),
                          tables_found=sum(len(p.get("tables", [])) for p in result["pages"])))
            return JSONResponse(status_code=200, content=payload)
        finally:
            with admission:
                state["admitted"] -= 1

    return app


def build_app_from_env() -> FastAPI:
    return create_app(ServiceSettings.from_env())
