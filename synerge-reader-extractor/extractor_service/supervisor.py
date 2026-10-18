"""Supervised extraction worker process: one job at a time, hard timeout, real RSS limit.

The heavy engines (Tesseract, Docling, torch) live in a child process so that a
runaway page can be killed without taking the HTTP front down. The parent:

* sends one job at a time over a pipe;
* polls the child's resident set size from ``/proc/<pid>/status`` (VmRSS) and
  kills it above the configured limit;
* kills it when the job's wall-clock deadline passes;
* never retries a job. A killed worker is respawned lazily for the NEXT job, so
  a timeout or memory breach is reported to the caller as such.

The container's own memory limit remains the outer backstop.
"""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


class WorkerTimeout(Exception):
    pass


class WorkerMemoryExceeded(Exception):
    pass


class WorkerCrashed(Exception):
    pass


class JobFailed(Exception):
    """The worker ran the job and reported a contained failure (e.g. unreadable PDF)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def read_rss_bytes(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass
class JobStats:
    elapsed_s: float
    peak_rss_bytes: Optional[int]
    worker_pid: Optional[int]
    worker_respawned: bool


class Supervisor:
    def __init__(self, *, target: Callable[..., None], engine_config: dict, rss_limit_bytes: int,
                 poll_interval_s: float = 0.25, rss_reader: Callable[[int], Optional[int]] = read_rss_bytes,
                 context: Any = None):
        self._target = target
        self._engine_config = dict(engine_config)
        self._rss_limit = rss_limit_bytes
        self._poll = poll_interval_s
        self._rss_reader = rss_reader
        self._ctx = context or multiprocessing.get_context("spawn")
        self._proc = None
        self._conn = None
        self._lock = threading.Lock()
        self._spawns = 0

    @property
    def spawns(self) -> int:
        return self._spawns

    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def _spawn(self) -> None:
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(target=self._target, args=(child, self._engine_config), daemon=True,
                                 name="extractor-engine")
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent
        self._spawns += 1

    def _kill(self) -> None:
        proc, conn = self._proc, self._conn
        self._proc, self._conn = None, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=10)

    def shutdown(self) -> None:
        with self._lock:
            self._kill()

    def run(self, job: dict, timeout_s: float) -> tuple[dict, JobStats]:
        """Run one job; raises WorkerTimeout / WorkerMemoryExceeded / WorkerCrashed / JobFailed."""
        with self._lock:
            respawned = False
            if not self.alive():
                self._kill()
                self._spawn()
                respawned = self._spawns > 1
            started = time.monotonic()
            deadline = started + timeout_s
            peak: Optional[int] = None
            pid = self._proc.pid
            try:
                self._conn.send(job)
            except Exception:
                self._kill()
                raise WorkerCrashed("worker pipe closed") from None
            while True:
                if self._conn.poll(self._poll):
                    try:
                        message = self._conn.recv()
                    except Exception:
                        self._kill()
                        raise WorkerCrashed("worker exited during the job") from None
                    break
                rss = self._rss_reader(pid)
                if rss is not None:
                    peak = rss if peak is None else max(peak, rss)
                    if rss > self._rss_limit:
                        self._kill()
                        raise WorkerMemoryExceeded(f"worker RSS {rss} exceeded {self._rss_limit}")
                if not self._proc.is_alive():
                    self._kill()
                    raise WorkerCrashed("worker exited during the job")
                if time.monotonic() > deadline:
                    self._kill()
                    raise WorkerTimeout(f"job exceeded {timeout_s:.0f} s")
            stats = JobStats(elapsed_s=time.monotonic() - started, peak_rss_bytes=peak, worker_pid=pid,
                             worker_respawned=respawned)
            if not isinstance(message, dict):
                self._kill()
                raise WorkerCrashed("worker returned a malformed message")
            if message.get("ok"):
                return message["result"], stats
            raise JobFailed(str(message.get("kind", "failed")), str(message.get("message", "")))
