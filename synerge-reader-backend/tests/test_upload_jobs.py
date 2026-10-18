"""Bounded owner-scoped background upload jobs (LIVE-EXTRACTION-20261017). No threads are started."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from upload_jobs import JobState, SubmitOutcome, UploadJobRegistry, UploadJobSettings  # noqa: E402


class _InertThread:
    started = 0

    def __init__(self, target, name, daemon):
        self.target = target

    def start(self):
        _InertThread.started += 1


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _registry(**settings):
    clock = _Clock()
    reg = UploadJobRegistry(UploadJobSettings(**settings), clock=clock, thread_factory=_InertThread, log=lambda m: None)
    return reg, clock


def test_construction_starts_nothing_and_first_submit_starts_one_worker():
    _InertThread.started = 0
    reg, _ = _registry()
    assert reg.snapshot()["worker_started"] is False and _InertThread.started == 0
    reg.submit("alice", lambda: (200, {"ok": 1}), size_bytes=10, file_count=1)
    reg.submit("bob", lambda: (200, {"ok": 2}), size_bytes=10, file_count=1)
    assert _InertThread.started == 1


def test_owner_sees_progress_and_result_other_owners_see_nothing():
    reg, clock = _registry()
    sub = reg.submit("alice", lambda: (207, {"results": ["r"]}), size_bytes=5, file_count=1)
    assert sub.outcome is SubmitOutcome.ACCEPTED and len(sub.job_id) >= 20
    queued = reg.get(sub.job_id, "alice")
    assert queued.state is JobState.QUEUED and queued.queue_position == 1 and queued.result is None
    assert reg.get(sub.job_id, "bob") is None and reg.get("guess", "alice") is None
    clock.now += 5
    assert reg.run_next_for_test()
    done = reg.get(sub.job_id, "alice").to_dict()
    assert done["status"] == "completed" and done["http_status"] == 207 and done["result"] == {"results": ["r"]}
    assert reg.get(sub.job_id, "bob") is None


def test_queue_owner_and_size_limits_are_enforced():
    reg, _ = _registry(max_queued=2, max_per_owner=1, max_job_bytes=100, max_queued_bytes=150)
    assert reg.submit("a", lambda: (200, {}), size_bytes=101, file_count=1).outcome is SubmitOutcome.TOO_LARGE
    assert reg.submit("a", lambda: (200, {}), size_bytes=10, file_count=1).outcome is SubmitOutcome.ACCEPTED
    assert reg.submit("a", lambda: (200, {}), size_bytes=10, file_count=1).outcome is SubmitOutcome.OWNER_LIMIT
    assert reg.submit("b", lambda: (200, {}), size_bytes=100, file_count=1).outcome is SubmitOutcome.ACCEPTED
    assert reg.submit("c", lambda: (200, {}), size_bytes=10, file_count=1).outcome is SubmitOutcome.QUEUE_FULL
    assert reg.submit(None, lambda: (200, {}), size_bytes=1, file_count=1).outcome is SubmitOutcome.UNAVAILABLE


def test_a_crashing_job_is_failed_with_a_generic_message_never_a_success():
    reg, _ = _registry()

    def boom():
        raise RuntimeError("secret internal detail")

    sub = reg.submit("alice", boom, size_bytes=1, file_count=1)
    reg.run_next_for_test()
    view = reg.get(sub.job_id, "alice").to_dict()
    assert view["status"] == "failed" and view["http_status"] == 503
    assert "secret" not in str(view) and "upload again" in view["result"]["error_message"]


def test_finished_jobs_release_their_bytes_and_expire():
    reg, clock = _registry(result_ttl_s=60)
    sub = reg.submit("alice", lambda: (200, {}), size_bytes=1, file_count=1)
    reg.run_next_for_test()
    assert reg._jobs[sub.job_id].work is None  # queued upload bytes released
    clock.now += 61
    assert reg.get(sub.job_id, "alice") is None


def test_retention_cap_drops_oldest_finished_jobs():
    reg, clock = _registry(max_retained=2, max_per_owner=5)
    ids = []
    for i in range(3):
        ids.append(reg.submit("alice", lambda: (200, {}), size_bytes=1, file_count=1).job_id)
        clock.now += 1
        reg.run_next_for_test()
    reg.submit("bob", lambda: (200, {}), size_bytes=1, file_count=1)
    assert reg.get(ids[0], "alice") is None and reg.get(ids[2], "alice") is not None
