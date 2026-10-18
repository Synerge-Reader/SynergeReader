"""Bounded, owner-scoped background ingestion jobs for slow (OCR / table) uploads.

Scanned-page OCR and table extraction can take minutes, longer than a browser
or reverse-proxy request window. Instead of raising timeouts, ``/upload`` can
accept a batch as a background job and the client polls
``GET /upload/jobs/{job_id}`` with the same signed-in identity.

Design limits (all explicit, none silently raised):

* one worker thread, so at most one ingestion batch runs at a time (the
  extraction worker and the embedder are shared, finite resources);
* a bounded queue, a per-owner cap and a cap on queued upload bytes; a full
  queue is refused with an outcome the route turns into HTTP 429;
* finished results are retained for a bounded time and count, then expire;
* jobs live only in this process. A restart forgets them: the client then sees
  "not found" and must re-check its document list. Because each document is
  committed atomically by the ingestion service, a lost job can never leave a
  half-indexed document or be reported as indexed when it was not.

Importing the module starts nothing; the worker thread starts on the first
accepted job. Job ids are random and unguessable, and a job is visible only to
the owner that submitted it -- any other caller gets the same answer as for a
job that does not exist.
"""

from __future__ import annotations

import enum
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class JobState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SubmitOutcome(enum.Enum):
    ACCEPTED = "accepted"
    QUEUE_FULL = "queue_full"
    OWNER_LIMIT = "owner_limit"
    TOO_LARGE = "too_large"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class UploadJobSettings:
    max_queued: int = 8
    max_per_owner: int = 2
    max_queued_bytes: int = 200 * 1024 * 1024
    max_job_bytes: int = 120 * 1024 * 1024
    max_retained: int = 64
    result_ttl_s: float = 3600.0


@dataclass
class _Job:
    job_id: str
    owner: Any
    work: Optional[Callable[[], tuple[int, dict]]]
    size_bytes: int
    created_at: float
    file_count: int
    state: JobState = JobState.QUEUED
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    http_status: Optional[int] = None
    result: Optional[dict] = None


@dataclass(frozen=True)
class JobView:
    job_id: str
    state: JobState
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    file_count: int
    queue_position: Optional[int]
    http_status: Optional[int]
    result: Optional[dict]

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "file_count": self.file_count,
            "queue_position": self.queue_position,
            "http_status": self.http_status,
            "result": self.result,
        }


@dataclass(frozen=True)
class SubmitResult:
    outcome: SubmitOutcome
    job_id: Optional[str] = None


_FAILED_RESULT_MESSAGE = "Background processing failed before the upload could be indexed. Please upload again."


class UploadJobRegistry:
    def __init__(
        self,
        settings: UploadJobSettings = UploadJobSettings(),
        *,
        clock: Callable[[], float] = time.time,
        thread_factory: Callable[..., Any] = threading.Thread,
        log: Callable[[str], None] = print,
    ):
        self._settings = settings
        self._clock = clock
        self._thread_factory = thread_factory
        self._log = log
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._jobs: dict[str, _Job] = {}
        self._queue: deque[str] = deque()
        self._worker_started = False
        self._closed = False

    # -- admission ---------------------------------------------------------

    def submit(self, owner: Any, work: Callable[[], tuple[int, dict]], *, size_bytes: int,
               file_count: int) -> SubmitResult:
        if owner is None:
            return SubmitResult(SubmitOutcome.UNAVAILABLE)
        with self._lock:
            if self._closed:
                return SubmitResult(SubmitOutcome.UNAVAILABLE)
            self._expire_locked()
            if size_bytes > self._settings.max_job_bytes:
                return SubmitResult(SubmitOutcome.TOO_LARGE)
            pending = [self._jobs[j] for j in self._queue] + [
                job for job in self._jobs.values() if job.state is JobState.RUNNING
            ]
            if len(self._queue) >= self._settings.max_queued:
                return SubmitResult(SubmitOutcome.QUEUE_FULL)
            if sum(job.size_bytes for job in pending) + size_bytes > self._settings.max_queued_bytes:
                return SubmitResult(SubmitOutcome.QUEUE_FULL)
            if sum(1 for job in pending if job.owner == owner) >= self._settings.max_per_owner:
                return SubmitResult(SubmitOutcome.OWNER_LIMIT)
            job_id = secrets.token_urlsafe(18)
            self._jobs[job_id] = _Job(job_id=job_id, owner=owner, work=work, size_bytes=size_bytes,
                                      created_at=self._clock(), file_count=file_count)
            self._queue.append(job_id)
            if not self._worker_started:
                try:
                    thread = self._thread_factory(target=self._worker_loop, name="upload-job-worker", daemon=True)
                    thread.start()
                    self._worker_started = True
                except Exception:
                    self._queue.pop()
                    del self._jobs[job_id]
                    self._log("[UploadJobs] worker could not be started")
                    return SubmitResult(SubmitOutcome.UNAVAILABLE)
            self._wake.notify()
            return SubmitResult(SubmitOutcome.ACCEPTED, job_id)

    # -- status ------------------------------------------------------------

    def get(self, job_id: str, owner: Any) -> Optional[JobView]:
        """The owner's job, or None for unknown/expired jobs AND other owners' jobs."""
        if not isinstance(job_id, str) or owner is None:
            return None
        with self._lock:
            self._expire_locked()
            job = self._jobs.get(job_id)
            if job is None or job.owner != owner:
                return None
            position = None
            if job.state is JobState.QUEUED:
                try:
                    position = list(self._queue).index(job_id) + 1
                except ValueError:
                    position = None
            return JobView(job_id=job.job_id, state=job.state, created_at=job.created_at,
                           started_at=job.started_at, finished_at=job.finished_at, file_count=job.file_count,
                           queue_position=position, http_status=job.http_status, result=job.result)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "queued": len(self._queue),
                "running": sum(1 for j in self._jobs.values() if j.state is JobState.RUNNING),
                "retained": len(self._jobs),
                "worker_started": self._worker_started,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._wake.notify_all()

    # -- worker ------------------------------------------------------------

    def run_next_for_test(self) -> bool:
        """Run one queued job on the calling thread (tests use an inert thread factory)."""
        with self._lock:
            if not self._queue:
                return False
            job = self._take_locked()
        self._run(job)
        return True

    def _take_locked(self) -> _Job:
        job = self._jobs[self._queue.popleft()]
        job.state = JobState.RUNNING
        job.started_at = self._clock()
        return job

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                job = self._take_locked()
            self._run(job)

    def _run(self, job: _Job) -> None:
        work = job.work
        status: Optional[int] = None
        result: Optional[dict] = None
        state = JobState.FAILED
        try:
            status, result = work()
            state = JobState.COMPLETED
        except Exception as exc:
            # Fixed-shape log: never the exception text or document content.
            self._log(f"[UploadJobs] job failed: {type(exc).__name__}")
            status, result = 503, None
        with self._lock:
            job.work = None  # release the queued upload bytes
            job.state = state
            job.finished_at = self._clock()
            job.http_status = status
            job.result = result if state is JobState.COMPLETED else {"error_message": _FAILED_RESULT_MESSAGE}

    # -- retention ---------------------------------------------------------

    def _expire_locked(self) -> None:
        now = self._clock()
        finished = [j for j in self._jobs.values() if j.state in (JobState.COMPLETED, JobState.FAILED)]
        for job in finished:
            if job.finished_at is not None and now - job.finished_at > self._settings.result_ttl_s:
                del self._jobs[job.job_id]
        finished = sorted(
            (j for j in self._jobs.values() if j.state in (JobState.COMPLETED, JobState.FAILED)),
            key=lambda j: j.finished_at or 0,
        )
        overflow = len(self._jobs) - self._settings.max_retained
        for job in finished[: max(overflow, 0)]:
            del self._jobs[job.job_id]
