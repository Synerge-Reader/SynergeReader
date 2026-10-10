"""The bounded Gemma capacity probe, exercised only against fakes.

No Ollama, database, embedding server or network, and no real child process:
every "worker" runs the REAL worker code in-process against a fake
http.client connection (the pattern of tests/test_gemma_qualification.py),
with synthetic expected digests. Two answer lanes really run on two threads,
so overlap, the in-flight limit and the cross-lane stop are exercised, not
simulated.
"""

import ast
import dataclasses
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diagnostics import gemma_capacity_probe as probe  # noqa: E402
from diagnostics import gemma_qualification as gq  # noqa: E402
from diagnostics import generation_diagnostic as base  # noqa: E402
from generation_admission import CallState  # noqa: E402
from generation_profiles import GEMMA_CANDIDATE, LLAMA_CONTROL, ModelRegistry, default_model_registry  # noqa: E402

_PROBE_PATH = Path(probe.__file__)
SYNTHETIC_GEMMA = "9a" * 32
SYNTHETIC_LLAMA = "1a" * 32
EMBED_DIGEST = "e5" * 32
EMBED_MODEL = "mxbai-embed-large:335m"
_DIGESTS = {"llama3.1:8b": SYNTHETIC_LLAMA, "gemma4:31b": SYNTHETIC_GEMMA, EMBED_MODEL: EMBED_DIGEST}
# The C04/C05 Gemma answer payloads the qualification harness sends (and sent
# on 9 Oct 2026): the probe's lanes send the same bytes.
_PAYLOAD_SHAS = {
    "C04": "4cc910aabe41a0ef55501f5cad644b9d0dc96b6393feab211832202f72142350",
    "C05": "719e56ce5f630f829c6257ccec86753af24e2f7dc0afcead6b30a63c49b011ab",
}
_C04_ANSWER = "IRCoT alternates retrieval and chain-of-thought steps [C1]. Later queries use the last CoT sentence [C1]."
_C05_ANSWER = "IRCoT stops on the answer-is string or a step limit [C1]. It used 8 steps and 15 paragraphs [C1]."
_THINKING = "PRIVATE-PROBE-THINKING"


def _synthetic_registry():
    registry = default_model_registry()
    return ModelRegistry(
        profiles=tuple(dataclasses.replace(p, expected_digest=_DIGESTS[p.model_tag]) for p in registry.profiles),
        default_profile_id=LLAMA_CONTROL,
    )


def _ndjson(*records):
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def _stream(text, *, thinking=None, done_reason="stop"):
    records = [{"thinking": thinking, "response": ""}] if thinking else []
    records += [{"response": text[i:i + 9], "done": False} for i in range(0, len(text), 9)]
    records.append({"response": "", "done": True, "done_reason": done_reason, "eval_count": 9,
                    "eval_duration": 3_000_000_000, "prompt_eval_count": 960, "prompt_eval_duration": 2_000_000_000,
                    "load_duration": 1_000_000_000, "total_duration": 6_000_000_000})
    return _ndjson(*records)


def _vector(dimension=1024, value=0.01):
    return [value] * dimension


def _role(body):
    return {32: "smoke", 1200: "answer", 16: "verifier"}[body["options"]["num_predict"]]


def _fixture_of(body):
    return "C04" if base.load_fixture("C04").question in body["prompt"] else "C05"


class FakeOllama:
    """Native endpoints. Answers wait for each other at a barrier (proving
    the two lanes overlap) and are held briefly so the residency sampler sees
    the workload; ``hold_c05_until`` holds the C05 answer until a condition."""

    def __init__(self, *, version="0.30.8", digests=None, capabilities=("completion", "thinking"),
                 embedding_length=1024, overlap_barrier=True, hold_s=0.3):
        self.version = version
        self.digests = dict(_DIGESTS if digests is None else digests)
        self.capabilities = list(capabilities)
        self.embedding_length = embedding_length
        self.answers = {"C04": _stream(_C04_ANSWER, thinking=_THINKING), "C05": _stream(_C05_ANSWER)}
        self.embed_reply = (200, json.dumps({"model": EMBED_MODEL, "embeddings": [_vector()],
                                             "total_duration": 40_000_000, "load_duration": 9_000_000,
                                             "prompt_eval_count": 14}).encode())
        self.barrier = threading.Barrier(2, timeout=5) if overlap_barrier else None
        self.hold_s = hold_s
        self.hold_c05_until = None
        self.requests = []
        self._lock = threading.Lock()
        self.active_generate = 0
        self.peak_generate = 0

    def connection(self, scheme, host, port, timeout):
        assert (scheme, host, port) == ("http", "fake-ollama", 11434)
        return FakeConnection(self)

    def respond(self, method, path, body):
        with self._lock:
            self.requests.append((method, path, body))
        if path == "/api/version":
            return 200, json.dumps({"version": self.version}).encode()
        if path == "/api/tags":
            models = [{"name": tag, "model": tag, "digest": digest, "size": 1} for tag, digest in self.digests.items()]
            return 200, json.dumps({"models": models}).encode()
        if path == "/api/show":
            if body["model"] == EMBED_MODEL:
                info = {"bert.embedding_length": self.embedding_length, "bert.context_length": 512}
                return 200, json.dumps({"model_info": info, "capabilities": ["embedding"]}).encode()
            return 200, json.dumps({"model_info": {"gemma4.context_length": 262144},
                                    "capabilities": self.capabilities}).encode()
        if path == "/api/ps":
            models = [{"name": "gemma4:31b", "model": "gemma4:31b", "size": 24, "size_vram": 24,
                       "context_length": 8192, "expires_at": "t"},
                      {"name": EMBED_MODEL, "model": EMBED_MODEL, "size": 6, "size_vram": 6,
                       "context_length": 512, "expires_at": "t"}]
            return 200, json.dumps({"models": models}).encode()
        if path == "/api/embed":
            return self.embed_reply
        assert path == "/api/generate", path
        role = _role(body)
        if role == "smoke":
            return 200, _stream("ready")
        if role == "verifier":
            return 200, json.dumps({"response": "supported", "done": True, "done_reason": "stop",
                                    "eval_count": 1, "eval_duration": 200_000_000, "prompt_eval_count": 300,
                                    "prompt_eval_duration": 900_000_000, "total_duration": 1_200_000_000}).encode()
        fixture = _fixture_of(body)
        with self._lock:
            self.active_generate += 1
            self.peak_generate = max(self.peak_generate, self.active_generate)
        try:
            if self.barrier is not None:
                self.barrier.wait()
            if fixture == "C05" and self.hold_c05_until is not None:
                deadline = time.monotonic() + 5
                while not self.hold_c05_until() and time.monotonic() < deadline:
                    time.sleep(0.005)
            time.sleep(self.hold_s)
            return 200, self.answers[fixture]
        finally:
            with self._lock:
                self.active_generate -= 1

    def paths(self):
        return [path for _, path, _ in self.requests]

    def generate_roles(self):
        return [_role(body) for _, path, body in self.requests if path == "/api/generate"]


class FakeConnection:
    def __init__(self, server):
        self.server = server
        self._response = None

    def connect(self):
        pass

    def request(self, method, path, body=None, headers=None):
        payload = None if body is None else json.loads(body)
        status, data = self.server.respond(method, path, payload)
        self._response = FakeResponse(status, data)

    def getresponse(self):
        return self._response

    def close(self):
        pass


class FakeResponse:
    def __init__(self, status, data):
        self.status = status
        self._data = data
        self._offset = 0

    def read1(self, size):
        chunk = self._data[self._offset:self._offset + min(size, 11)]
        self._offset += len(chunk)
        return chunk


class FakeProcess:
    def __init__(self, launcher, argv, behaviour):
        self.launcher = launcher
        self.argv = argv
        self.behaviour = behaviour
        self.pid = 80_000 + len(launcher.processes)
        self.returncode = None
        self.killed = False

    def wait(self, timeout=None):
        self.launcher.wait_timeouts.append(timeout)
        if self.returncode is not None:
            return self.returncode
        if self.behaviour == "block" and not self.killed:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        if self.killed:
            self.returncode = 1
            return 1
        request_path, result_path, body_path = (Path(p) for p in self.argv[6:9])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = gq._new_worker_result()
        result["nonce"] = request["nonce"]
        gq.worker_call(request, body_path, result, connection_factory=self.launcher.server.connection)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        self.launcher.kills.append(self)


class FakeLauncher:
    def __init__(self, server, choose=None):
        self.server = server
        self.choose = choose or (lambda request: "respond")
        self.processes, self.kills, self.wait_timeouts, self.argvs = [], [], [], []
        self._lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        request = json.loads(Path(argv[6]).read_text(encoding="utf-8"))
        with self._lock:
            self.argvs.append(list(argv))
            process = FakeProcess(self, list(argv), self.choose(request))
            self.processes.append(process)
        return process


def _block_when(predicate):
    return lambda request: "block" if predicate(request) else "respond"


def _is_answer(fixture):
    return lambda request: (request["path"] == "/api/generate" and _role(request["body"]) == "answer"
                            and _fixture_of(request["body"]) == fixture)


def _args(out, **overrides):
    values = {"--profile": GEMMA_CANDIDATE, "--ollama-url": "http://fake-ollama:11434",
              "--expected-ollama-version": "0.30.8", "--embedding-model": EMBED_MODEL,
              "--embedding-digest": EMBED_DIGEST, "--embedding-keep-alive": "30m",
              "--client-budget-seconds": "30", "--probe-budget-seconds": "60", "--out": str(out)}
    values.update(overrides)
    args = ["--live"]
    for flag, value in values.items():
        if value is not None:
            args += [flag, value]
    return args


def _probe(tmp_path, server, choose=None, *, clock=time.monotonic, **overrides):
    launcher = FakeLauncher(server, choose)
    latches = []

    def factory(url, *, budget_s, work_root, latch, allow_embedding):
        latches.append(latch)
        return gq.OwnedWorkerSupervisor(url, budget_s=budget_s, work_root=work_root, launcher=launcher,
                                        python="fake-python", worker_script=Path("fake-worker.py"),
                                        latch=latch, allow_embedding=allow_embedding)

    server.shared_latches = latches
    out = tmp_path / "probe.jsonl"
    stdout = io.StringIO()
    code = probe.main(_args(out, **overrides), stdout=stdout, registry_factory=_synthetic_registry,
                      supervisor_factory=factory, clock=clock, sample_interval_s=0.05)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(Path(f"{out}.summary.json").read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("gq-*")), "every per-call work directory is removed"
    return SimpleNamespace(code=code, records=records, summary=summary, stdout=stdout.getvalue(), out=out,
                           launcher=launcher, server=server, latches=latches)


def _calls(run, role=None):
    calls = [r for r in run.records if r["record_type"] == "call"]
    return [c for c in calls if role is None or c["role"] == role]


# --------------------------------------------------------------------------
# A completed probe
# --------------------------------------------------------------------------


def test_a_completed_probe_runs_identity_one_embedding_one_smoke_then_two_overlapping_lanes(tmp_path):
    run = _probe(tmp_path, FakeOllama())
    assert run.code == 0 and run.summary["outcome"] == "completed" and run.summary["stop_category"] is None

    paths = run.server.paths()
    first_model_call = min(paths.index("/api/embed"), paths.index("/api/generate"))
    assert set(paths[:first_model_call]) == {"/api/version", "/api/tags", "/api/show", "/api/ps"}
    assert paths.count("/api/embed") == 1 and paths.index("/api/embed") < paths.index("/api/generate")
    assert run.server.generate_roles()[0] == "smoke"
    assert sorted(run.server.generate_roles()) == ["answer", "answer", "smoke"] + ["verifier"] * 4
    assert run.summary["model_calls_dispatched"] == 8 == probe.MAX_MODEL_CALLS

    smoke = next(b for _, p, b in run.server.requests if p == "/api/generate" and _role(b) == "smoke")
    assert smoke["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 32}
    assert smoke["think"] is False and smoke["keep_alive"] == "5m"
    [embed] = [b for _, p, b in run.server.requests if p == "/api/embed"]
    assert embed == {"model": EMBED_MODEL, "truncate": False, "keep_alive": "30m",
                     "input": ["Represent this sentence for searching relevant passages: "
                               + probe.SYNTHETIC_EMBEDDING_TEXT]}
    answers = {_fixture_of(b): b for _, p, b in run.server.requests if p == "/api/generate" and _role(b) == "answer"}
    assert {f: gq._sha(base._canonical(b)) for f, b in answers.items()} == _PAYLOAD_SHAS

    assert run.server.peak_generate == 2, "the two answers were in flight together"
    assert run.summary["peak_generation_calls_in_flight"] == 2 <= probe.MAX_CONCURRENT_GENERATIONS
    assert run.summary["max_overlapping_generation_intervals"] == 2
    assert run.summary["answer_overlap_s"] > 0
    jobs = run.summary["jobs"]
    assert {name: (job["fixture"], job["verifier_calls"], job["stopped"]) for name, job in jobs.items()} == {
        "A": ("C04", 2, None), "B": ("C05", 2, None)}
    for job in jobs.values():
        assert job["answer_plus_verifier_s"] > 0 and job["exceeds_app_job_budget"] is False
    assert run.summary["application_job_budget_s"] == 420.0


def test_identities_settings_metrics_and_residency_are_recorded_separately(tmp_path):
    run = _probe(tmp_path, FakeOllama())
    identity = run.summary["identity"]
    assert identity["runtime"] == {"expected": "0.30.8", "observed": "0.30.8"}
    assert identity["generation"]["expected_digest"] == identity["generation"]["observed_digest"] == SYNTHETIC_GEMMA
    assert identity["generation"]["think_sent"] is False
    embedding = identity["embedding"]
    assert (embedding["expected_digest"], embedding["observed_digest"], embedding["check"]) == (
        EMBED_DIGEST, EMBED_DIGEST, "match")
    assert (embedding["expected_dimension"], embedding["observed_embedding_length"],
            embedding["observed_context_length"]) == (1024, 1024, 512)
    assert run.summary["embedding_validation"] == {"accepted": True, "vectors": 1, "dimension": 1024,
                                                   "failure": None}

    requested = run.summary["requested"]
    assert requested["answer_options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
    assert requested["verifier_options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16}
    assert requested["embedding"] == {"model": EMBED_MODEL, "dimension": 1024, "query_prefix_applied": True,
                                      "truncate": False, "keep_alive": "30m"}
    for call in run.summary["calls"]:
        assert call["started_s"] <= call["ended_s"] and call["lane"] in ("setup", "A", "B")
        assert call["observed"] is not None, f"server metrics recorded for {call['role']}"
    answer = next(c for c in run.summary["calls"] if c["role"] == "answer")
    assert answer["observed"]["observed_output_tokens_per_s"] == 3.0
    assert answer["options"]["num_ctx"] == 8192

    assert set(run.summary["residency"]) == {"before_embedding", "after_embedding", "after_smoke", "after_workload"}
    assert run.summary["residency_during_workload"], "sampled while the lanes ran"
    effective = {m["name"]: (m["context_length"], m["vram_pct"]) for m in run.summary["residency"]["after_workload"]["models"]}
    assert effective == {"gemma4:31b": (8192, 100.0), EMBED_MODEL: (512, 100.0)}
    assert run.summary["missing_observations"] == []
    assert run.summary["qualification"]["combined_capacity_verified"] is False
    assert run.summary["qualification"]["quality_verified"] is False


# --------------------------------------------------------------------------
# Cross-lane UNKNOWN fencing
# --------------------------------------------------------------------------


def test_an_unknown_answer_on_one_lane_fences_new_dispatch_on_both(tmp_path):
    server = FakeOllama(overlap_barrier=False, hold_s=0)
    # Lane A's C04 worker never finishes: the client deadline ends its wait
    # (UNKNOWN). Lane B's C05 answer, already sent, completes only after the
    # shared latch recorded that stop -- and B may then dispatch nothing more.
    server.hold_c05_until = lambda: any(l.stopped for l in server.shared_latches)
    run = _probe(tmp_path, server, _block_when(_is_answer("C04")))

    assert run.code == 3 and run.summary["outcome"] == "stopped"
    assert run.summary["stop_category"] == "client_deadline"
    assert run.summary["unknown_server_execution"] is True
    assert run.summary["unresolved_execution"] == [{"lane": "A", "role": "answer", "fixture": "C04",
                                                   "category": "client_deadline"}]
    assert "not a cancellation" in run.summary["unknown_note"]
    assert "verifier" not in run.server.generate_roles(), "no verifier call after the stop, on either lane"
    by_lane = {c["lane"]: c for c in _calls(run, "answer")}
    assert by_lane["A"]["state"] == CallState.UNKNOWN.value
    assert by_lane["B"]["state"] == CallState.OBSERVED_TERMINAL.value and by_lane["B"]["accepted"] is True
    assert run.summary["jobs"]["B"]["stopped"] == "client_deadline"
    assert [p.killed for p in run.launcher.kills] == [True], "only the blocked owned worker was stopped"
    assert run.summary["model_calls_dispatched"] == 4
    assert run.summary["residency"]["after_workload"] is not None, "read-only observation still allowed"


def test_an_operator_interrupt_during_the_lanes_fences_dispatch_and_waits_for_started_work(tmp_path, monkeypatch):
    original = threading.Thread.join
    raised = []

    def join(self, timeout=None):
        if self.name.startswith("probe-lane-") and not raised:
            raised.append(self.name)
            raise KeyboardInterrupt
        return original(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", join)
    run = _probe(tmp_path, FakeOllama(overlap_barrier=False, hold_s=0.05))
    assert raised and run.code == 3 and run.summary["stop_category"] == "operator_interrupt"
    assert "verifier" not in run.server.generate_roles(), "nothing new is dispatched after the interrupt"
    assert any("operator interrupt" in m for m in run.summary["missing_observations"])
    assert all(event["reaped"] for event in run.summary["worker_events"]), "no started worker is orphaned"


def test_a_terminal_but_rejected_answer_skips_only_its_own_verification(tmp_path):
    server = FakeOllama()
    server.answers["C05"] = _stream("Cut off [C1]", done_reason="length")
    run = _probe(tmp_path, server)
    assert run.summary["outcome"] == "completed" and run.summary["unknown_server_execution"] is False
    assert run.summary["jobs"]["A"]["verifier_calls"] == 2 and run.summary["jobs"]["B"]["verifier_calls"] == 0
    assert run.summary["model_calls_dispatched"] == 6


# --------------------------------------------------------------------------
# Identity and the embedding
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server_kwargs, category",
    [
        ({"version": "0.31.0"}, "runtime_version_mismatch"),
        ({"digests": {**_DIGESTS, "gemma4:31b": "77" * 32}}, "identity_mismatch"),
        ({"capabilities": ("completion",)}, "thinking_capability_missing"),
        ({"digests": {**_DIGESTS, EMBED_MODEL: "77" * 32}}, "embedding_identity_mismatch"),
        ({"digests": {k: v for k, v in _DIGESTS.items() if k != EMBED_MODEL}}, "embedding_identity_absent"),
        ({"embedding_length": 768}, "embedding_identity_dimension_mismatch"),
    ],
    ids=["runtime", "gemma-digest", "no-thinking", "embedder-digest", "embedder-absent", "embedder-dimension"],
)
def test_an_identity_mismatch_stops_before_any_model_call(tmp_path, server_kwargs, category):
    run = _probe(tmp_path, FakeOllama(**server_kwargs))
    assert run.code == 3 and run.summary["stop_category"] == category
    assert "/api/embed" not in run.server.paths() and "/api/generate" not in run.server.paths()
    assert run.summary["model_calls_dispatched"] == 0 and run.summary["unknown_server_execution"] is False


@pytest.mark.parametrize(
    "reply",
    [
        (200, json.dumps({"embeddings": [_vector(1023)]}).encode()),
        (200, json.dumps({"embeddings": [[math.nan] + _vector(1023)]}).encode()),
        (200, json.dumps({"embeddings": [_vector(value=0.0)]}).encode()),
        (200, json.dumps({"embeddings": [_vector(), _vector()]}).encode()),
        (200, json.dumps({"embeddings": [[True] + _vector(1023)]}).encode()),
        (200, json.dumps({"model": EMBED_MODEL}).encode()),
        (200, b"not json"),
        (500, json.dumps({"error": "boom /srv/secret-detail"}).encode()),
    ],
    ids=["wrong-dimension", "nan", "all-zero", "two-vectors", "boolean", "no-embeddings", "not-json", "http-500"],
)
def test_strict_embedding_validation_stops_before_any_generation(tmp_path, reply):
    server = FakeOllama()
    server.embed_reply = reply
    run = _probe(tmp_path, server)
    assert run.code == 3 and run.summary["stop_category"] == "embedding_rejected"
    assert "/api/generate" not in run.server.paths()
    validation = run.summary["embedding_validation"]
    assert validation is None or validation["accepted"] is False
    [record] = [r for r in run.records if r["record_type"] == "embedding_validation"]
    assert record["accepted"] is False and record["failure"].endswith("Error")
    assert run.summary["unknown_server_execution"] is False, "the server answered; the reply was rejected"
    assert "secret-detail" not in json.dumps(run.summary) and "secret-detail" not in run.stdout


def test_an_embedding_without_a_reply_is_unknown_and_stops_everything(tmp_path):
    run = _probe(tmp_path, FakeOllama(), _block_when(lambda request: request["path"] == "/api/embed"))
    assert run.summary["stop_category"] == "client_deadline" and run.summary["unknown_server_execution"] is True
    [embedding] = _calls(run, "embedding")
    assert embedding["state"] == CallState.UNKNOWN.value
    assert "/api/generate" not in run.server.paths()


# --------------------------------------------------------------------------
# Deadlines and limits
# --------------------------------------------------------------------------


def test_the_probe_deadline_caps_every_wait_and_stops_new_dispatch(tmp_path):
    server = FakeOllama()
    offset = [0.0]
    clock = lambda: time.monotonic() + offset[0]  # noqa: E731
    original = server.respond

    def respond(method, path, body):
        reply = original(method, path, body)
        if path == "/api/embed":
            offset[0] = 1_000.0  # the whole-probe budget is spent once the embedding returns
        return reply

    server.respond = respond
    run = _probe(tmp_path, server, clock=clock, **{"--probe-budget-seconds": "20"})
    assert run.summary["stop_category"] == "probe_deadline"
    assert "/api/generate" not in run.server.paths() and run.server.paths().count("/api/embed") == 1
    assert run.launcher.wait_timeouts and max(run.launcher.wait_timeouts) <= 20, "no wait beyond the probe budget"
    assert any("probe_deadline" in m for m in run.summary["missing_observations"]), "later views are reported missing"


def test_a_stalled_smoke_is_unknown_and_no_answer_lane_starts(tmp_path):
    run = _probe(tmp_path, FakeOllama(),
                 _block_when(lambda r: r["path"] == "/api/generate" and _role(r["body"]) == "smoke"))
    assert run.summary["stop_category"] == "client_deadline" and run.summary["unknown_server_execution"] is True
    assert run.server.generate_roles() == [] and run.summary["jobs"]["A"]["stopped"] == "lane_not_started"


def test_the_dispatcher_refuses_a_ninth_model_call_and_a_third_concurrent_generation():
    latch = probe.SharedStopLatch()
    dispatch = probe.ProbeDispatch(latch, clock=time.monotonic, origin=time.monotonic(),
                                   deadline=time.monotonic() + 60)

    class Supervisor:
        def __init__(self):
            self.calls = 0

        def call(self, **kwargs):
            self.calls += 1
            assert 0 < kwargs["budget_s"] <= 60
            return "outcome"

    supervisor = Supervisor()
    for _ in range(probe.MAX_MODEL_CALLS):
        assert dispatch.call(supervisor, {}, model_call=True, path=gq.GENERATE_PATH) == "outcome"
    with pytest.raises(gq.BatchStopped) as stopped:
        dispatch.call(supervisor, {}, model_call=True, path=gq.GENERATE_PATH)
    assert stopped.value.category == "probe_call_limit" and latch.stopped == "probe_call_limit"
    assert supervisor.calls == probe.MAX_MODEL_CALLS

    fresh = probe.ProbeDispatch(probe.SharedStopLatch(), clock=time.monotonic, origin=time.monotonic(),
                                deadline=time.monotonic() + 60)
    fresh.generations_in_flight = probe.MAX_CONCURRENT_GENERATIONS
    with pytest.raises(gq.BatchStopped) as refused:
        fresh.call(supervisor, {}, model_call=True, path=gq.GENERATE_PATH)
    assert refused.value.category == "probe_concurrency_limit"
    assert fresh.call(supervisor, {}, model_call=False, path="/api/ps") == "outcome", "observations are not limited"


def test_a_shared_latch_fences_a_supervisor_that_has_not_started_yet(tmp_path):
    latch = probe.SharedStopLatch()
    launcher = FakeLauncher(FakeOllama())
    first, second = (gq.OwnedWorkerSupervisor("http://fake-ollama:11434", budget_s=5, work_root=tmp_path,
                                              launcher=launcher, python="fake-python",
                                              worker_script=Path("fake-worker.py"), latch=latch)
                     for _ in range(2))
    first.stop("client_deadline")
    with pytest.raises(gq.BatchStopped):
        second.call(model_call=True, method="POST", path="/api/generate", body={}, read_timeout_s=1,
                    max_body_bytes=1, watch_stream=False)
    assert launcher.processes == [], "nothing was started after the stop"


# --------------------------------------------------------------------------
# Outputs, inputs and boundary
# --------------------------------------------------------------------------


def test_protected_answers_stay_in_the_result_file_and_the_summary_carries_no_text(tmp_path):
    run = _probe(tmp_path, FakeOllama())
    protected = run.out.read_text(encoding="utf-8")
    assert _C04_ANSWER in protected and _C05_ANSWER in protected
    summary_text = Path(f"{run.out}.summary.json").read_text(encoding="utf-8")
    private = [_C04_ANSWER[:30], _C05_ANSWER[:30], _THINKING, probe.SYNTHETIC_EMBEDDING_TEXT, "<evidence>",
               base.load_fixture("C04").question, base.load_fixture("C05").question, "0.01, 0.01"]
    for text in private:
        assert text not in summary_text and text not in run.stdout
    for line in run.stdout.splitlines():
        assert line.startswith(("[setup]", "[A]", "[B]", "probe ")), line


def test_the_dry_run_reports_the_plan_without_network(monkeypatch):
    import socket

    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("no network in a dry run"))
    out = io.StringIO()
    assert probe.main(["--dry-run", "--embedding-model", EMBED_MODEL], stdout=out) == 0
    report = json.loads(out.getvalue())
    assert report["planned_model_calls"] == {"embedding": 1, "smoke": 1, "answers": 2, "verifier_max": 4,
                                             "max_total": 8}
    assert report["max_concurrent_generation_calls"] == 2
    assert report["smoke_options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 32}
    assert {lane["id"]: lane["answer_payload_sha256"] for lane in report["lanes"]} == _PAYLOAD_SHAS
    assert report["profile"]["production_selectable"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"--profile": LLAMA_CONTROL},
        {"--embedding-model": "unknown-embedder:1"},
        {"--embedding-digest": "abc"},
        {"--embedding-keep-alive": "0"},
        {"--embedding-keep-alive": "-1"},
        {"--embedding-keep-alive": None},
        {"--expected-ollama-version": "latest"},
        {"--probe-budget-seconds": "0"},
        {"--probe-budget-seconds": "inf"},
        {"--client-budget-seconds": "nan"},
        {"--ollama-url": "http://user:pass@h:1"},
    ],
    ids=["llama-profile", "unknown-embedder", "short-digest", "keep-alive-zero", "keep-alive-negative",
         "keep-alive-missing", "version", "probe-budget-zero", "probe-budget-infinite", "call-budget-nan",
         "credentials"],
)
def test_inputs_are_validated_before_any_output_or_worker(tmp_path, overrides):
    out = tmp_path / "probe.jsonl"
    with pytest.raises((gq.UsageError, base.HarnessUsageError)):
        probe.main(_args(out, **overrides), stdout=io.StringIO(), registry_factory=_synthetic_registry,
                   supervisor_factory=lambda *a, **k: pytest.fail("no supervisor before inputs are valid"))
    assert not out.exists()


def test_the_probe_imports_and_endpoints_stay_inside_their_boundary():
    tree = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & {"main", "dbSetup", "psycopg2", "pgvector", "dotenv", "requests", "socket", "subprocess",
                        "signal", "psutil", "os"}
    source = _PROBE_PATH.read_text(encoding="utf-8")
    for forbidden in ("os.kill", "killpg", "taskkill", "shell=True", ".terminate(", "/api/pull", "/api/delete",
                      "/api/copy", "/api/create", "/api/push", "/api/chat", '"keep_alive": 0', "keep_alive=0",
                      "load_dotenv", "environ"):
        assert forbidden not in source
    assert gq.OBSERVATION_PATHS == {"/api/version", "/api/tags", "/api/show", "/api/ps"}
