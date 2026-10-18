import multiprocessing

import pytest

from extractor_service.supervisor import JobFailed, Supervisor, WorkerCrashed, WorkerMemoryExceeded, WorkerTimeout
from supervisor_targets import echo_worker


def _supervisor(rss=None, limit=10 ** 12, poll=0.05):
    return Supervisor(target=echo_worker, engine_config={"k": "v"}, rss_limit_bytes=limit, poll_interval_s=poll,
                      rss_reader=(lambda pid: rss) if rss is not None else (lambda pid: None),
                      context=multiprocessing.get_context("spawn"))


def test_runs_a_job_and_reuses_the_worker():
    sup = _supervisor()
    try:
        result, stats = sup.run({"value": 7}, timeout_s=60)
        assert result == {"echo": 7, "cfg": {"k": "v"}}
        assert stats.worker_respawned is False
        result2, _ = sup.run({"value": 8}, timeout_s=60)
        assert result2["echo"] == 8
        assert sup.spawns == 1
    finally:
        sup.shutdown()


def test_timeout_kills_the_worker_and_the_next_job_respawns():
    sup = _supervisor()
    try:
        sup.run({"value": 1}, timeout_s=60)
        with pytest.raises(WorkerTimeout):
            sup.run({"mode": "sleep", "seconds": 30}, timeout_s=1)
        assert not sup.alive()
        result, stats = sup.run({"value": 2}, timeout_s=60)
        assert result["echo"] == 2
        assert stats.worker_respawned is True
        assert sup.spawns == 2
    finally:
        sup.shutdown()


def test_memory_limit_kills_the_worker():
    sup = _supervisor(rss=2_000, limit=1_000)
    try:
        with pytest.raises(WorkerMemoryExceeded):
            sup.run({"mode": "sleep", "seconds": 30}, timeout_s=60)
        assert not sup.alive()
    finally:
        sup.shutdown()


def test_worker_exit_is_reported_as_a_crash():
    sup = _supervisor()
    try:
        with pytest.raises(WorkerCrashed):
            sup.run({"mode": "die"}, timeout_s=60)
    finally:
        sup.shutdown()


def test_contained_failure_keeps_the_worker():
    sup = _supervisor()
    try:
        with pytest.raises(JobFailed) as info:
            sup.run({"mode": "fail"}, timeout_s=60)
        assert info.value.kind == "input"
        assert sup.alive()
        assert sup.spawns == 1
    finally:
        sup.shutdown()
