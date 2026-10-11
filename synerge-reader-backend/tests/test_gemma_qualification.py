"""The bounded Gemma answer + verifier evaluation, exercised only against fakes.

No Ollama, database, embedding or network. Every live-path test injects a
fake launcher whose "processes" run the REAL worker code in-process against a
fake http.client connection, with synthetic expected digests. One test starts
a real child process: the inspected, model-free script below, which blocks
its own network first and then never finishes, to prove the real supervisor
stops waiting, terminates and reaps exactly that child. Output files go to
pytest's temporary directory.

The verifier is main.py's own source compiled by the harness's adapter -- a
compiled/source adapter, not a running /ask route or a database test.
"""

import ast
import dataclasses
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from citation_generation import ClaimReason, ClaimVerificationStatus
from diagnostics import gemma_qualification as gq
from diagnostics import generation_diagnostic as base
from generation_admission import CallState
from generation_profiles import (
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    SMOKE_OPTIONS,
    ModelRegistry,
    Qualification,
    _snapshot,
    default_model_registry,
)

_HARNESS_PATH = Path(gq.__file__)
_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
# The same pins as tests/test_generation_diagnostic.py: fixture identities and
# prompt bytes are unchanged by this harness.
_EVIDENCE_SHA = "0bc45a825ae0a2bce85877d30f6a6fcced4802713bf3fb66db9b13af50f9c8c7"
_FIXTURE_SHAS = {
    "C04": "6fd9f3172bb5323766aa101b619704f5c8aa0a6bff46bcf65ac31f227e05880e",
    "C05": "6b32291d9e313750e597a529f91f56a4cb2f4f2521772873c3db4130ca4f12b0",
}
_PROMPT_SHAS = {
    "C04": "492a1828a1297822c7054fb8614b6b8755aa9a95becc81407a0133a94d6cfec7",
    "C05": "fc3e74cdc22ee70227f75e446dfdca260fba1fdfc05d2bda71a5122cf2383ed6",
}
SYNTHETIC_LLAMA = "1a" * 32
SYNTHETIC_GEMMA = "9a" * 32
_SYNTHETIC_DIGESTS = {"llama3.1:8b": SYNTHETIC_LLAMA, "gemma4:31b": SYNTHETIC_GEMMA}

_THINKING = "PRIVATE-THINKING-PLAN"
_C04_ANSWER = (
    "IRCoT interleaves retrieval with chain-of-thought reasoning [C1]. "
    "Each later retrieval step uses the last generated sentence as its query — not the question [C1]. "
    "The loop continues until an answer is reached [C1]."
)
_C05_ANSWER = "IRCoT stops when the reasoning says the answer is reached [C1]. It also stops at a step limit [C1]."
_ALL_FIXTURES = ("C01", "C02", "C03", "C04", "C05", "C06")
_OTHER_ANSWERS = {
    "C01": "PaLM-2L gains 7% on MMLU Physics and 11% on MMLU Chemistry [C1]. TimeQA rises 27% and MuSiQue 7% [C1].",
    "C02": "The big Transformer reached 28.4 BLEU [C1]. Training took 3.5 days [C1].",
    "C03": "Abstraction asks a step-back question about a higher-level principle [C1]. Reasoning answers the "
           "original question grounded on it [C1].",
    "C06": "IRCoT interleaves retrieval with chain-of-thought steps [C1]. Step-Back abstracts first and then "
           "reasons [C2].",
}
# Answer payload sha256 per (profile, fixture): the default C04/C05 bytes are
# the ones sent on 9 Oct 2026 (Gemma) and recorded in the preparation dry run.
_PAYLOAD_SHAS = {
    (GEMMA_CANDIDATE, "C04"): "4cc910aabe41a0ef55501f5cad644b9d0dc96b6393feab211832202f72142350",
    (GEMMA_CANDIDATE, "C05"): "719e56ce5f630f829c6257ccec86753af24e2f7dc0afcead6b30a63c49b011ab",
    (LLAMA_CONTROL, "C04"): "168d3907d5eacd396e6141ab2e65594e7e491dbddff905605cbd8cf4ee5409a6",
    (LLAMA_CONTROL, "C05"): "53a038877519019a2e7e2f67129cdcc32b77cb6b806c8ef428500cfc61b35cf1",
}


# Gemma with only its think:false gate recorded: an explicitly unqualified
# candidate, so these tests do not depend on the released qualification.
_UNQUALIFIED_GEMMA = Qualification(think_false_verified=True, combined_capacity_verified=False,
                                   quality_verified=False, evidence_ref="synthetic unqualified candidate")


def _synthetic_profile(profile):
    fields = {"expected_digest": _SYNTHETIC_DIGESTS[profile.model_tag]}
    if profile.profile_id == GEMMA_CANDIDATE:
        fields["qualification"] = _UNQUALIFIED_GEMMA
    return dataclasses.replace(profile, **fields)


def _synthetic_registry():
    """The production profiles with synthetic expected digests; Gemma unqualified."""
    registry = default_model_registry()
    return ModelRegistry(
        profiles=tuple(_synthetic_profile(p) for p in registry.profiles),
        default_profile_id=LLAMA_CONTROL,
    )


def _ndjson(*records):
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


def _stream(text, *, thinking=None, done_reason="stop"):
    records = [{"thinking": thinking, "response": ""}] if thinking else []
    records += [{"response": text[i:i + 9], "done": False} for i in range(0, len(text), 9)]
    records.append({"response": "", "done": True, "done_reason": done_reason, "eval_count": 7,
                    "prompt_eval_count": 1210})
    return _ndjson(*records)


def _verifier_reply(word="supported", **overrides):
    body = {"response": word, "done": True, "done_reason": "stop", **overrides}
    return 200, json.dumps(body).encode("utf-8")


# --------------------------------------------------------------------------
# Fakes: an Ollama server, an http.client connection, a launcher, processes
# --------------------------------------------------------------------------


class FakeOllama:
    """Native endpoints only. Generate replies are chosen by role (num_predict
    32 smoke, 1200 answer, 16 verifier) and, for answers, by fixture."""

    def __init__(self, *, digests=None, capabilities=("completion", "thinking"), chunk_size=7):
        self.digests = dict(_SYNTHETIC_DIGESTS if digests is None else digests)
        self.capabilities = None if capabilities is None else list(capabilities)
        self.chunk_size = chunk_size
        self.requests = []
        self.answers = {"C04": _stream(_C04_ANSWER, thinking=_THINKING), "C05": _stream(_C05_ANSWER),
                        **{fid: _stream(text) for fid, text in _OTHER_ANSWERS.items()}}
        self.smoke = _stream("ready")
        self.verifier_replies = []
        self.refuse_connect = False
        self.faults = {}  # request index (1-based) -> "reset" | "timeout"

    def connection(self, scheme, host, port, timeout):
        assert (scheme, host, port) == ("http", "fake-ollama", 11434)
        return FakeConnection(self)

    def respond(self, method, path, body):
        self.requests.append((method, path, body))
        if path == "/api/version":
            return 200, b'{"version": "0.0.0-test"}'
        if path == "/api/tags":
            models = [{"name": tag, "model": tag, "digest": digest, "size": 1} for tag, digest in self.digests.items()]
            return 200, json.dumps({"models": models}).encode("utf-8")
        if path == "/api/show":
            shown = {"parameters": "top_k 40", "model_info": {"gemma.context_length": 131072},
                     "license": "LICENSE-TEXT-NOT-RECORDED"}
            if self.capabilities is not None:
                shown["capabilities"] = self.capabilities
            return 200, json.dumps(shown).encode("utf-8")
        if path == "/api/ps":
            return 200, json.dumps({"models": [{"name": "gemma4:31b", "size_vram": 5, "expires_at": "t"}]}).encode()
        assert path == "/api/generate"
        role = _role(body)
        if role == "smoke":
            return 200, self.smoke
        if role == "answer":
            reply = self.answers[_fixture_of(body)]
            return reply if isinstance(reply, tuple) else (200, reply)
        return self.verifier_replies.pop(0) if self.verifier_replies else _verifier_reply()

    @property
    def generate_bodies(self):
        return [body for _, path, body in self.requests if path == "/api/generate"]

    @property
    def sequence(self):
        return [(path if path != "/api/generate" else f"generate:{_role(body)}") for _, path, body in self.requests]


def _role(body):
    return {32: "smoke", 1200: "answer", 16: "verifier"}[body["options"]["num_predict"]]


def _fixture_of(body):
    [fixture_id] = [f for f in _ALL_FIXTURES if base.load_fixture(f).question in body["prompt"]]
    return fixture_id


class FakeConnection:
    def __init__(self, server):
        self.server = server
        self.closed = False
        self._response = None

    def connect(self):
        if self.server.refuse_connect:
            raise ConnectionRefusedError("refused /srv/secret-detail")

    def request(self, method, path, body=None, headers=None):
        payload = None if body is None else json.loads(body)
        index = len(self.server.requests) + 1
        status, data = self.server.respond(method, path, payload)
        self._response = FakeResponse(status, data, self.server.chunk_size, self.server.faults.get(index))

    def getresponse(self):
        return self._response

    def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, status, data, chunk_size, fault):
        self.status = status
        self._data = data
        self._chunk = chunk_size
        self._offset = 0
        self._fault = fault

    def read1(self, size):
        if self._fault and self._offset >= self._chunk:
            raise {"reset": ConnectionResetError("reset /srv/secret-detail"),
                   "timeout": TimeoutError("timed out")}[self._fault]
        chunk = self._data[self._offset:self._offset + min(size, self._chunk)]
        self._offset += len(chunk)
        return chunk


class FakeProcess:
    """A launched 'worker'. ``respond`` runs gq.worker_call in-process; other
    behaviours model a child that never exits (``block``), cannot be reaped,
    crashes, or returns a wrong nonce."""

    def __init__(self, launcher, argv, behaviour):
        self.launcher = launcher
        self.argv = argv
        self.behaviour = behaviour
        self.pid = 70_000 + len(launcher.processes)
        self.returncode = None
        self.killed = False

    def _run_worker(self):
        request_path, result_path, body_path = (Path(p) for p in self.argv[6:9])
        if self.behaviour == "crash":
            return 1
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = gq._new_worker_result()
        result["nonce"] = "0" * 32 if self.behaviour == "bad_nonce" else request["nonce"]
        gq.worker_call(request, body_path, result, connection_factory=self.launcher.server.connection)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        return 0

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if self.behaviour == "unreapable" or (self.behaviour == "block" and not self.killed):
            raise subprocess.TimeoutExpired(self.argv, timeout)
        self.returncode = 1 if self.killed else self._run_worker()
        return self.returncode

    def kill(self):
        self.killed = True
        self.launcher.kills.append(self)


class FakeLauncher:
    """Stands in for subprocess.Popen; nothing is executed. ``choose`` maps
    (request, launch index) to a behaviour."""

    def __init__(self, server, choose=None):
        self.server = server
        self.choose = choose or (lambda request, index: "respond")
        self.argvs, self.kwargs, self.processes, self.kills = [], [], [], []

    def __call__(self, argv, **kwargs):
        self.argvs.append(list(argv))
        self.kwargs.append(kwargs)
        request = json.loads(Path(argv[6]).read_text(encoding="utf-8"))
        behaviour = self.choose(request, len(self.argvs))
        if behaviour == "start_fails":
            raise OSError("cannot start /srv/secret-detail")
        process = FakeProcess(self, list(argv), behaviour)
        self.processes.append(process)
        return process


def _when(predicate, behaviour):
    return lambda request, index: behaviour if predicate(request, index) else "respond"


def _is_generate(role, fixture=None):
    def predicate(request, index):
        if request["path"] != "/api/generate" or _role(request["body"]) != role:
            return False
        return fixture is None or _fixture_of(request["body"]) == fixture
    return predicate


def _supervisor(launcher, tmp_path, **kwargs):
    return gq.OwnedWorkerSupervisor(
        "http://fake-ollama:11434", budget_s=kwargs.pop("budget_s", 30), work_root=tmp_path,
        launcher=launcher, python="fake-python", worker_script=Path("fake-worker.py"), **kwargs,
    )


def _live(tmp_path, server, choose=None, *, profile=GEMMA_CANDIDATE, extra=()):
    launcher = FakeLauncher(server, choose)
    out = tmp_path / "run.jsonl"
    stdout = io.StringIO()

    def factory(url, *, budget_s, work_root):
        return gq.OwnedWorkerSupervisor(url, budget_s=budget_s, work_root=work_root, launcher=launcher,
                                        python="fake-python", worker_script=Path("fake-worker.py"))

    code = gq.main(
        ["--live", "--profile", profile, "--ollama-url", "http://fake-ollama:11434",
         "--client-budget-seconds", "30", "--out", str(out), *extra],
        stdout=stdout, registry_factory=_synthetic_registry, supervisor_factory=factory,
    )
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(Path(f"{out}.summary.json").read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("gq-*")), "every per-call work directory is removed"
    return SimpleNamespace(code=code, records=records, summary=summary, stdout=stdout.getvalue(),
                           launcher=launcher, server=server, out=out)


def _calls(run):
    return [r for r in run.records if r["record_type"] == "call"]


def _gold_strings():
    anchors = json.loads(base.GRADING_PATH.read_text(encoding="utf-8"))
    for spec in anchors["fixtures"].values():
        yield from spec["required_facts"]


# --------------------------------------------------------------------------
# Import, dry run and frozen inputs
# --------------------------------------------------------------------------


def test_a_fresh_import_and_a_dry_run_open_no_socket_process_or_thread(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no network, process or thread at import or dry run")

    import socket
    import threading
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(threading.Thread, "start", refuse)
    spec = importlib.util.spec_from_file_location("gemma_qualification_fresh_import", _HARNESS_PATH)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fresh)  # dataclasses resolve their module by name
    spec.loader.exec_module(fresh)

    out = io.StringIO()
    assert fresh.main(["--dry-run"], stdout=out) == 0
    supervisor = fresh.OwnedWorkerSupervisor("http://127.0.0.1:9", budget_s=1, launcher=refuse)
    assert supervisor.stopped is None and supervisor.events == []


def test_the_dry_run_reports_hashes_options_and_counts_only():
    out = io.StringIO()
    assert gq.main(["--dry-run"], stdout=out) == 0
    report = json.loads(out.getvalue())
    rendered = out.getvalue()

    assert report["profile"]["profile_id"] == GEMMA_CANDIDATE == gq.INTENDED_FIRST_PROFILE
    assert report["profile"]["model_tag"] == "gemma4:31b"
    assert report["profile"]["expected_digest"] == default_model_registry().get(GEMMA_CANDIDATE).expected_digest
    assert report["profile"]["answer_options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
    assert report["profile"]["verifier"] == {
        "model_tag": "gemma4:31b", "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16},
        "read_timeout_s": 150.0,
    }
    assert report["smoke"]["options"] == SMOKE_OPTIONS.as_ollama_options()
    assert [(f["id"], f["revision"], f["fixture_sha256"], f["evidence_sha256"], f["prompt_sha256"])
            for f in report["fixtures"]] == [
        (fid, 2, _FIXTURE_SHAS[fid], _EVIDENCE_SHA, _PROMPT_SHAS[fid]) for fid in ("C04", "C05")
    ]
    assert report["planned_model_calls"] == {"smoke": 1, "answers": 2, "verifier_max": 4, "max_total": 7}
    assert report["verifier_adapter"]["max_claims_verified"] == 2
    assert "not the running /ask route" in report["verifier_adapter"]["label"]
    for fixture_id in ("C04", "C05"):
        fixture = base.load_fixture(fixture_id)
        assert fixture.question not in rendered and fixture.evidence["text"][:60] not in rendered
    for gold in _gold_strings():
        assert gold not in rendered
    assert "<evidence>" not in rendered and base.SMOKE_PROMPT not in rendered


def test_a_dry_run_file_is_new_and_never_overwritten(tmp_path):
    target = tmp_path / "dry.json"
    gq.main(["--dry-run", "--out", str(target)], stdout=io.StringIO())
    assert json.loads(target.read_text(encoding="utf-8"))["record_type"] == "dry_run"
    with pytest.raises(gq.UsageError):
        gq.main(["--dry-run", "--out", str(target)], stdout=io.StringIO())


def test_fixture_prompts_are_the_existing_builder_bytes_and_pinned_hashes():
    adapter = gq.load_application_verifier(gq._refuse_dispatch)
    for fixture_id in gq.FIXTURES:
        fixture = base.load_fixture(fixture_id)
        plan = gq.plan_fixture(fixture, adapter.citation_limits)
        assert plan.prompt == base.build_prompt(fixture)
        assert gq._sha(plan.prompt) == _PROMPT_SHAS[fixture_id]
        assert (fixture.revision, fixture.fixture_sha256) == (2, _FIXTURE_SHAS[fixture_id])
        assert [record.citation_id for record in plan.registry.records] == ["C1"]


# --------------------------------------------------------------------------
# The main.py verifier source adapter
# --------------------------------------------------------------------------


def _main_definitions():
    source = _MAIN_PATH.read_bytes().decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(source)
    found = {}
    for node in tree.body:
        name = gq._defined_name(node)
        if name in gq.APPLICATION_VERIFIER_DEFINITIONS:
            found[name] = node
    return source, found


def test_the_adapter_compiles_the_current_main_source_without_importing_main(monkeypatch):
    monkeypatch.delitem(sys.modules, "main", raising=False)
    adapter = gq.load_application_verifier(gq._refuse_dispatch)
    assert "main" not in sys.modules, "main.py is read as text, never imported"

    source, found = _main_definitions()
    assert adapter.template == ast.literal_eval(found["_CLAIM_VERIFIER_PROMPT"].value)
    assert adapter.words == ast.literal_eval(found["_CLAIM_VERIFIER_WORDS"].value) == (
        "supported", "partially_supported", "unsupported")
    assert adapter.citation_limits.max_claims_verified == gq.MAX_VERIFIER_CALLS_PER_ANSWER == 2
    assert adapter.citation_limits == base._ROUTE_CITATION_LIMITS
    assert adapter.source_sha256 == {
        name: gq._sha(ast.get_source_segment(source, node)) for name, node in found.items()
    }
    assert adapter.main_sha256 == gq._sha(_MAIN_PATH.read_bytes())
    assert adapter.claim_verifier_status(" **Partially-Supported.** ") == "partially_supported"
    assert adapter.claim_verifier_status("not supported") is None


def test_verifier_payload_parity_with_the_current_source():
    sent = []

    def post_ollama(endpoint, payload, *, stream=False, timeout=60):
        sent.append((endpoint, payload, stream, timeout))
        return gq._WorkerHttpResponse(*_verifier_reply("Supported"))

    adapter = gq.load_application_verifier(post_ollama)
    profile = _synthetic_registry().get(GEMMA_CANDIDATE)
    snapshot = _snapshot(profile, "admin_override", SYNTHETIC_GEMMA)
    plan = gq.plan_fixture(base.load_fixture("C04"), adapter.citation_limits)
    records = plan.registry.records
    claim = "  IRCoT interleaves retrieval with reasoning [C1].  "

    assert adapter.ollama_claim_verifier(claim, records, snapshot.verifier) == "supported"
    [(endpoint, payload, stream, timeout)] = sent
    _, found = _main_definitions()
    template = ast.literal_eval(found["_CLAIM_VERIFIER_PROMPT"].value)
    evidence = "\n\n".join(f"[{record.citation_id}] {record.evidence_text}" for record in records)
    assert (endpoint, stream, timeout) == ("/api/generate", False, snapshot.verifier.read_timeout_s)
    assert payload == snapshot.verifier.ollama_payload(template.format(evidence=evidence, claim=claim.strip()))
    assert payload["model"] == profile.model_tag and payload["think"] is False
    assert payload["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 16}


@pytest.mark.parametrize("change", ["new-dependency", "missing", "duplicate"])
def test_the_adapter_fails_closed_when_main_changes(tmp_path, change):
    source = _MAIN_PATH.read_bytes().decode("utf-8")
    if change == "new-dependency":
        source = source.replace(
            "    prompt = _CLAIM_VERIFIER_PROMPT.format(",
            "    connect_to_postgres()\n    prompt = _CLAIM_VERIFIER_PROMPT.format(", 1)
    elif change == "missing":
        source = source.replace("def _claim_verifier_status(", "def _renamed_claim_verifier_status(", 1)
    else:
        source += "\n_CLAIM_VERIFIER_WORDS = ('supported',)\n"
    changed = tmp_path / "main.py"
    changed.write_text(source, encoding="utf-8")
    with pytest.raises(gq.AdapterError):
        gq.load_application_verifier(gq._refuse_dispatch, changed)


@pytest.mark.parametrize(
    "reply, status, reason",
    [
        (_verifier_reply("Unsupported."), ClaimVerificationStatus.UNSUPPORTED, ClaimReason.VERIFIED),
        (_verifier_reply("maybe"), ClaimVerificationStatus.UNVERIFIED, ClaimReason.VERIFIER_MALFORMED),
        (_verifier_reply("supported", done_reason="length"), ClaimVerificationStatus.UNVERIFIED,
         ClaimReason.VERIFIER_MALFORMED),
        (_verifier_reply("supported", error="runner /srv/secret-detail"), ClaimVerificationStatus.UNVERIFIED,
         ClaimReason.VERIFIER_UNAVAILABLE),
        ((500, b'{"error": "boom /srv/secret-detail"}'), ClaimVerificationStatus.UNVERIFIED,
         ClaimReason.VERIFIER_UNAVAILABLE),
        ((200, b"not json"), ClaimVerificationStatus.UNVERIFIED, ClaimReason.VERIFIER_UNAVAILABLE),
    ],
    ids=["unsupported", "malformed-word", "length", "error-body", "http-500", "not-json"],
)
def test_the_original_verifier_status_and_error_mapping_is_unchanged(reply, status, reason):
    adapter = gq.load_application_verifier(lambda endpoint, payload, **kwargs: gq._WorkerHttpResponse(*reply))
    plan = gq.plan_fixture(base.load_fixture("C04"), adapter.citation_limits)
    snapshot = _snapshot(_synthetic_registry().get(GEMMA_CANDIDATE), "admin_override", SYNTHETIC_GEMMA)
    result = gq.run_verification("IRCoT interleaves retrieval with reasoning [C1].", plan, snapshot, adapter,
                                 SimpleNamespace(context={}))
    [claim] = result["claims"]
    assert (claim["status"], claim["reason"]) == (status.value, reason.value)
    assert "secret-detail" not in json.dumps(result)


# --------------------------------------------------------------------------
# The worker (child-side code) against a fake connection
# --------------------------------------------------------------------------


def _worker(tmp_path, server, *, body=None, path="/api/generate", watch=True, limit=1 << 20):
    request = {"protocol": gq.WORKER_PROTOCOL, "nonce": "n", "base_url": "http://fake-ollama:11434",
               "method": "POST" if body is not None else "GET", "path": path, "body": body,
               "read_timeout_s": 5.0, "max_body_bytes": limit, "watch_stream": watch}
    result = gq._new_worker_result()
    gq.worker_call(request, tmp_path / "body.bin", result, connection_factory=server.connection)
    return result


def test_the_worker_reads_to_end_of_body_and_times_the_first_answer_token_not_thinking(tmp_path):
    server = FakeOllama(chunk_size=5)
    body = {"model": "gemma4:31b", "prompt": base.load_fixture("C04").question, "options": {"num_predict": 1200}}
    result = _worker(tmp_path, server, body=body)
    assert result["category"] == "end_of_body" and result["http_status"] == 200
    assert (tmp_path / "body.bin").read_bytes() == server.answers["C04"]
    timing = result["timing"]
    assert timing["first_response_token_s"] is not None and timing["done_record_s"] is not None
    assert timing["first_response_token_s"] <= timing["done_record_s"] <= timing["end_of_body_s"]


@pytest.mark.parametrize(
    "setup, category, phase",
    [
        ("refuse", "not_sent_connect_failed", "connect"),
        ("reset", "transport_error_after_send", "read"),
        ("timeout", "read_timeout_after_send", "read"),
        ("limit", "body_limit_exceeded", "read"),
    ],
)
def test_worker_failures_are_fixed_categories_without_messages(tmp_path, setup, category, phase):
    server = FakeOllama()
    if setup == "refuse":
        server.refuse_connect = True
    elif setup in ("reset", "timeout"):
        server.faults[1] = setup
    body = {"model": "m", "prompt": "p", "options": {"num_predict": 32}}
    result = _worker(tmp_path, server, body=body, limit=10 if setup == "limit" else 1 << 20)
    assert (result["category"], result["phase"]) == (category, phase)
    assert "secret-detail" not in json.dumps(result)
    assert set(result) == {"protocol", "nonce", "phase", "category", "error_type", "http_status",
                           "body_bytes", "timing"}


# --------------------------------------------------------------------------
# A full batch with fake workers
# --------------------------------------------------------------------------


def test_a_completed_batch_runs_in_order_with_one_profile_and_at_most_two_verifier_calls(tmp_path):
    run = _live(tmp_path, FakeOllama())
    assert run.code == 0 and run.summary["outcome"] == "completed" and run.summary["unknown_server_execution"] is False
    assert run.server.sequence == [
        "/api/version", "/api/tags", "/api/show", "/api/ps", "generate:smoke", "/api/ps",
        "generate:answer", "generate:verifier", "generate:verifier",
        "generate:answer", "generate:verifier", "generate:verifier", "/api/ps",
    ]
    assert {body["model"] for body in run.server.generate_bodies} == {"gemma4:31b"}, "never another tag"

    profile = _synthetic_registry().get(GEMMA_CANDIDATE)
    snapshot = _snapshot(profile, "admin_override", SYNTHETIC_GEMMA)
    smoke, c04, *rest = run.server.generate_bodies
    assert smoke == base.request_body(profile, base.SMOKE_PROMPT, think=False, options=SMOKE_OPTIONS)
    for fixture_id, body in (("C04", c04), ("C05", rest[2])):
        prompt = base.build_prompt(base.load_fixture(fixture_id))
        assert body == snapshot.ollama_payload(prompt, stream=True) == base.request_body(profile, prompt, think=False)
        assert gq._sha(body["prompt"]) == _PROMPT_SHAS[fixture_id]
    for body in rest[:2] + rest[3:]:
        assert body["options"] == snapshot.verifier.options.as_ollama_options() and body["stream"] is False
        assert (body["model"], body["keep_alive"], body["think"]) == ("gemma4:31b", profile.keep_alive, False)
    states = [c["state"] for c in _calls(run)]
    assert states == [CallState.OBSERVED_TERMINAL.value] * 7
    verification = {v["fixture"]: v for v in run.summary["verification"]}
    assert verification["C04"]["verifier_calls"] == 2 and verification["C05"]["verifier_calls"] == 2
    assert sum(verification["C04"]["claim_status_counts"].values()) == 3, "the third claim is left unverified"


def test_expected_and_observed_digests_are_recorded_separately(tmp_path):
    run = _live(tmp_path, FakeOllama())
    identity = run.summary["identity"]
    assert identity["check"] == "match"
    assert identity["expected_digest"] == identity["observed_digest"] == SYNTHETIC_GEMMA
    assert identity["reports_thinking"] is True and identity["think_sent"] is False
    [record] = [r for r in run.records if r["record_type"] == "identity"]
    assert "license" not in json.dumps(record) and "digest" not in record["identity"]


@pytest.mark.parametrize(
    "server_kwargs, category",
    [
        (dict(digests={"llama3.1:8b": SYNTHETIC_LLAMA}), "identity_absent"),
        (dict(digests={**_SYNTHETIC_DIGESTS, "gemma4:31b": "77" * 32}), "identity_mismatch"),
        (dict(digests={**_SYNTHETIC_DIGESTS, "gemma4:31b": "garbled"}), "identity_unreadable"),
        (dict(capabilities=("completion",)), "thinking_capability_missing"),
        (dict(capabilities=None), "thinking_capability_missing"),
    ],
    ids=["absent", "mismatch", "malformed-digest", "no-thinking-capability", "capabilities-missing"],
)
def test_identity_or_thinking_evidence_failures_stop_before_any_generate(tmp_path, server_kwargs, category):
    run = _live(tmp_path, FakeOllama(**server_kwargs))
    assert run.server.generate_bodies == [], "nothing is dispatched without confirmed identity and thinking evidence"
    assert (run.summary["outcome"], run.summary["stop_category"]) == ("stopped", category)
    assert run.summary["unknown_server_execution"] is False and run.code == 3
    assert run.summary["identity"]["expected_digest"] == SYNTHETIC_GEMMA
    if category == "identity_mismatch":
        assert run.summary["identity"]["observed_digest"] == "77" * 32


def test_strict_completion_keeps_thinking_out_and_only_accepted_answers_are_verified(tmp_path):
    server = FakeOllama()
    server.answers["C04"] = _stream("Cut off at the limit [C1]", done_reason="length")
    run = _live(tmp_path, server)
    c04, c05 = [c for c in _calls(run) if c["role"] == "answer"]
    assert (c04["state"], c04["accepted"], c04["failure"]) == (CallState.OBSERVED_TERMINAL.value, False, "output_limit")
    assert (c05["state"], c05["accepted"], c05["failure"]) == (CallState.OBSERVED_TERMINAL.value, True, None)
    assert run.server.sequence.count("generate:verifier") == 2, "only C05 is verified"
    assert {v["fixture"]: v.get("skipped") for v in run.summary["verification"]} == {"C04": "answer_not_accepted",
                                                                                      "C05": None}

    second = tmp_path / "with-thinking"
    second.mkdir()
    thinking = _live(second, FakeOllama())
    answer = [c for c in _calls(thinking) if c["role"] == "answer"][0]
    assert answer["text"] == _C04_ANSWER and _THINKING not in answer["text"]
    assert answer["thinking_chars"] == len(_THINKING)


@pytest.mark.parametrize(
    "smoke, failure, thinking_chars",
    [
        (_stream("ready", done_reason="length"), "output_limit", 0),
        (_stream(""), None, 0),
        (_stream("   \n\t "), None, 0),
        (_stream("", thinking="ready"), None, 5),
        (_stream("ready") + b"{not json\n", "malformed", 0),
    ],
    ids=["length", "blank", "whitespace", "thinking-only", "malformed-after-final-record"],
)
def test_a_terminal_but_rejected_smoke_stops_before_any_later_call(tmp_path, smoke, failure, thinking_chars):
    server = FakeOllama()
    server.smoke = smoke
    run = _live(tmp_path, server)

    assert run.code == 3
    assert (run.summary["outcome"], run.summary["stop_category"]) == ("stopped", "smoke_rejected")
    assert run.summary["unknown_server_execution"] is False, "the server finished: this is not UNKNOWN"
    [call] = _calls(run)
    assert (call["role"], call["state"], call["accepted"]) == ("smoke", CallState.OBSERVED_TERMINAL.value, False)
    assert (call["failure"], call["thinking_chars"]) == (failure, thinking_chars)
    assert run.server.sequence == ["/api/version", "/api/tags", "/api/show", "/api/ps", "generate:smoke"], (
        "no later dispatch: no residency read, answer or verifier")
    assert len(run.launcher.processes) == 5 and run.launcher.kills == []
    [end] = [r for r in run.records if r["record_type"] == "end"]
    assert (end["outcome"], end["stop_category"], end["unknown_server_execution"], end["note"]) == (
        "stopped", "smoke_rejected", False, None)
    assert "smoke -: category=end_of_body state=observed_terminal accepted=False" in run.stdout


def test_an_accepted_smoke_ignores_thinking_and_the_batch_continues(tmp_path):
    server = FakeOllama()
    server.smoke = _stream("ready", thinking="PLAN")
    run = _live(tmp_path, server)
    smoke = _calls(run)[0]
    assert (smoke["state"], smoke["accepted"], smoke["text"], smoke["thinking_chars"]) == (
        CallState.OBSERVED_TERMINAL.value, True, "ready", 4)
    assert run.code == 0 and run.summary["outcome"] == "completed" and "generate:answer" in run.server.sequence


def test_a_smoke_without_terminal_evidence_keeps_the_unknown_behaviour(tmp_path):
    server = FakeOllama()
    server.smoke = _ndjson({"response": "rea"}, {"error": "runner died /srv/secret-detail"})
    run = _live(tmp_path, server)
    [call] = _calls(run)
    assert (call["state"], call["accepted"]) == (CallState.UNKNOWN.value, False)
    assert (run.code, run.summary["stop_category"], run.summary["unknown_server_execution"]) == (
        3, "no_terminal_evidence", True)
    assert run.server.sequence[-1] == "generate:smoke"


def test_the_unknown_note_ends_the_window_without_retry_or_reset():
    note = gq.UNKNOWN_NOTE
    assert "not a cancellation and not a completion" in note
    assert "ends this batch and its window" in note
    assert "no automatic retry, unload, restart or reset" in note
    assert "Keep-alive expiry, an empty /api/ps or idle GPUs do not prove the work ended" in note
    assert "expire naturally" not in note and "stop is really needed" not in note


@pytest.mark.parametrize(
    "reply",
    [
        _ndjson({"response": "partial [C1]"}, {"error": "runner died /srv/secret-detail"}),
        _ndjson({"response": "no final record [C1]"}),
        b'{"response": "a"}\n{bad\n' + _ndjson({"response": "", "done": True, "done_reason": "stop"}),
        (500, b'{"error": "boom"}'),
    ],
    ids=["error-line", "end-of-body-without-final-record", "malformed-line", "http-500"],
)
def test_an_answer_without_terminal_evidence_is_unknown_and_stops_everything(tmp_path, reply):
    server = FakeOllama()
    server.answers["C04"] = reply
    run = _live(tmp_path, server)
    [answer] = [c for c in _calls(run) if c["role"] == "answer"]
    assert answer["state"] == CallState.UNKNOWN.value
    assert run.summary["stop_category"] == "no_terminal_evidence" and run.summary["unknown_server_execution"] is True
    assert run.server.sequence[-1] == "generate:answer", "no verifier, no C05, no later observation"
    assert "secret-detail" not in json.dumps(run.summary) and "secret-detail" not in run.stdout


@pytest.mark.parametrize("target", ["smoke", "C04-answer", "C04-first-verifier", "C05-answer"])
def test_a_client_deadline_after_possible_dispatch_is_unknown_and_stops_all_later_dispatch(tmp_path, target):
    verifier_seen = []

    def choose(request, index):
        if request["path"] != "/api/generate":
            return "respond"
        role = _role(request["body"])
        if role == "verifier":
            verifier_seen.append(index)
        fixture = _fixture_of(request["body"]) if role == "answer" else None
        hit = {"smoke": role == "smoke", "C04-answer": role == "answer" and fixture == "C04",
               "C04-first-verifier": role == "verifier" and len(verifier_seen) == 1,
               "C05-answer": role == "answer" and fixture == "C05"}[target]
        return "block" if hit else "respond"

    run = _live(tmp_path, FakeOllama(), choose)
    blocked = [p for p in run.launcher.processes if p.behaviour == "block"]
    assert len(blocked) == 1 and run.launcher.kills == blocked, "only the owned, blocked worker is killed"
    assert run.launcher.processes[-1] is blocked[0], "nothing is launched after the deadline"
    last = _calls(run)[-1]
    assert (last["category"], last["state"]) == ("client_deadline", CallState.UNKNOWN.value)
    assert run.summary["unknown_server_execution"] is True and run.summary["stop_category"] == "client_deadline"
    end = [r for r in run.records if r["record_type"] == "end"][0]
    assert "not a cancellation and not a completion" in end["note"]


def test_a_non_terminal_verifier_reply_keeps_its_mapping_but_no_further_call_is_made(tmp_path):
    server = FakeOllama()
    server.verifier_replies = [_verifier_reply("supported", error="runner /srv/secret-detail")]
    run = _live(tmp_path, server)
    assert run.server.sequence[-1] == "generate:verifier" and run.server.sequence.count("generate:verifier") == 1
    assert run.summary["stop_category"] == "no_terminal_evidence" and run.summary["unknown_server_execution"] is True
    [verifier] = [c for c in _calls(run) if c["role"] == "verifier"]
    assert verifier["state"] == CallState.UNKNOWN.value


@pytest.mark.parametrize(
    "behaviour, category, state",
    [
        ("start_fails", "worker_start_failed", CallState.NEVER_DISPATCHED.value),
        ("crash", "worker_exit_nonzero", CallState.UNKNOWN.value),
        ("bad_nonce", "ipc_invalid", CallState.UNKNOWN.value),
        ("unreapable", "client_deadline", CallState.UNKNOWN.value),
    ],
)
def test_worker_start_exit_and_ipc_failures_are_fixed_categories_that_stop(tmp_path, behaviour, category, state):
    run = _live(tmp_path, FakeOllama(), _when(_is_generate("smoke"), behaviour))
    [smoke] = _calls(run)
    assert (smoke["category"], smoke["state"]) == (category, state)
    assert run.summary["stop_category"] == category and run.server.sequence[-1] in ("/api/ps", "generate:smoke")
    assert "generate:answer" not in run.server.sequence
    if behaviour == "unreapable":
        assert smoke["worker_reaped"] is False and len(run.launcher.kills) == 1


def test_a_refused_connection_is_not_sent_and_still_stops(tmp_path):
    server = FakeOllama()
    launcher = FakeLauncher(server)
    supervisor = _supervisor(launcher, tmp_path)
    server.refuse_connect = True
    outcome = supervisor.call(model_call=True, method="POST", path="/api/generate", body={"options": {}},
                              read_timeout_s=5, max_body_bytes=100, watch_stream=True)
    assert (outcome.category, outcome.state) == ("not_sent_connect_failed", CallState.NEVER_DISPATCHED.value)
    with pytest.raises(gq.BatchStopped):
        supervisor.call(model_call=False, method="GET", path="/api/ps", body=None, read_timeout_s=5,
                        max_body_bytes=100, watch_stream=False)
    assert len(launcher.processes) == 1


def test_an_observation_failure_stops_before_any_generate(tmp_path):
    run = _live(tmp_path, FakeOllama(), _when(lambda request, index: request["path"] == "/api/version", "block"))
    assert run.server.generate_bodies == [] and run.summary["stop_category"] == "client_deadline"
    assert run.summary["unknown_server_execution"] is False, "an observation is not model work"


def test_the_supervisor_runs_one_worker_at_a_time_and_only_permitted_endpoints(tmp_path):
    server = FakeOllama()
    supervisor = _supervisor(FakeLauncher(server), tmp_path)
    for path in ("/api/pull", "/api/delete", "/api/chat", "/v1/chat/completions"):
        with pytest.raises(gq.AdapterError):
            supervisor.call(model_call=True, method="POST", path=path, body={}, read_timeout_s=1,
                            max_body_bytes=1, watch_stream=False)
    with pytest.raises(gq.AdapterError):
        supervisor.call(model_call=False, method="POST", path="/api/generate", body={}, read_timeout_s=1,
                        max_body_bytes=1, watch_stream=False)
    supervisor._active = True
    with pytest.raises(gq.AdapterError):
        supervisor.call(model_call=False, method="GET", path="/api/ps", body=None, read_timeout_s=1,
                        max_body_bytes=1, watch_stream=False)


def test_the_worker_gets_an_isolated_command_and_an_environment_with_systemroot_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_CONNECTION_STRING", "postgresql://secret")
    monkeypatch.setenv("SYNERGE_MIGRATION_TEST_DSN", "postgresql://secret")
    monkeypatch.setenv("PGPASSWORD", "secret")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://elsewhere:1")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy:1")
    run = _live(tmp_path, FakeOllama())
    for argv, kwargs in zip(run.launcher.argvs, run.launcher.kwargs):
        assert argv[:6] == ["fake-python", "-I", "-S", "-B", "fake-worker.py", "--worker"] and len(argv) == 9
        assert set(kwargs["env"]) <= {"SYSTEMROOT", "SYSTEMDRIVE"}
        assert not set(kwargs["env"].values()) & {"postgresql://secret", "secret", "http://elsewhere:1"}
        assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
        assert not kwargs.get("shell") and not kwargs.get("start_new_session") and not kwargs.get("creationflags")
    assert gq.worker_python() == (getattr(sys, "_base_executable", None) or sys.executable)


# --------------------------------------------------------------------------
# Outputs, CLI and invariants
# --------------------------------------------------------------------------


def test_protected_answers_stay_in_the_result_file_and_the_summary_is_shareable(tmp_path):
    run = _live(tmp_path, FakeOllama())
    protected = run.out.read_text(encoding="utf-8")
    assert _C04_ANSWER in protected and _C05_ANSWER in protected

    summary_text = Path(f"{run.out}.summary.json").read_text(encoding="utf-8")
    question = base.load_fixture("C04").question
    for private in (_C04_ANSWER[:40], _C05_ANSWER[:40], _THINKING, question, "<evidence>", "IRCoT interleaves"):
        assert private not in summary_text and private not in run.stdout
    for gold in _gold_strings():
        assert gold not in summary_text
    assert run.summary["scope_note"].startswith("C04/C05 use frozen manual evidence and native Ollama inference")
    assert "C01-C06 baseline is not represented" in run.summary["baseline_note"]
    assert set(run.summary["timing"]) >= {"smoke_first_token_s", "answer_first_token_s", "answer_s", "verifier_s",
                                         "total_diagnostic_s"}
    assert [a["fixture"] for a in run.summary["anchors"]] == ["C04", "C05"]
    assert all(line.split()[0] in ("smoke", "answer", "verifier", "batch") for line in run.stdout.splitlines())


def test_grading_is_read_only_after_the_last_network_call(tmp_path, monkeypatch):
    seen = {}
    original = gq.grade_answers

    def recording(answers, profile, grading_path):
        seen["requests_at_grading"] = len(server.requests)
        return original(answers, profile, grading_path)

    monkeypatch.setattr(gq, "grade_answers", recording)
    server = FakeOllama()
    run = _live(tmp_path, server)
    assert seen["requests_at_grading"] == len(run.server.requests)
    for body in run.server.generate_bodies:
        for gold in _gold_strings():
            assert gold not in body["prompt"]


def test_identity_and_qualification_flags_are_unchanged(tmp_path):
    before = default_model_registry()
    run = _live(tmp_path, FakeOllama())
    after = default_model_registry()
    assert before == after
    # The run used the explicitly unqualified candidate; it stays unqualified.
    gemma = _synthetic_registry().get(GEMMA_CANDIDATE)
    assert (gemma.selectable, gemma.requires_qualification) == (False, True)
    assert run.summary["profile"] == {
        "profile_id": GEMMA_CANDIDATE, "model_tag": "gemma4:31b", "production_selectable": False,
        "requires_qualification": True, "think_false_verified": True, "combined_capacity_verified": False,
        "quality_verified": False, "automatic_qualification": False,
    }


def test_results_and_summaries_are_never_overwritten(tmp_path):
    out = tmp_path / "run.jsonl"
    out.write_text("keep", encoding="utf-8")
    args = ["--live", "--profile", GEMMA_CANDIDATE, "--ollama-url", "http://fake-ollama:11434",
            "--client-budget-seconds", "30", "--out", str(out)]
    with pytest.raises(gq.UsageError):
        gq.main(args, stdout=io.StringIO(), registry_factory=_synthetic_registry)
    assert out.read_text(encoding="utf-8") == "keep"
    other = tmp_path / "other.jsonl"
    Path(f"{other}.summary.json").write_text("keep", encoding="utf-8")
    with pytest.raises(gq.UsageError):
        gq.main(args[:-1] + [str(other)], stdout=io.StringIO(), registry_factory=_synthetic_registry)
    assert not other.exists()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--dry-run", "--live"],
        ["--live", "--ollama-url", "http://h:1", "--client-budget-seconds", "5", "--out", "X"],
        ["--live", "--profile", "gemma_candidate,llama_control", "--ollama-url", "http://h:1",
         "--client-budget-seconds", "5", "--out", "X"],
        ["--live", "--profile", "gemma_candidate", "--ollama-url", "http://user:pass@h:1",
         "--client-budget-seconds", "5", "--out", "X"],
        ["--live", "--profile", "gemma_candidate", "--ollama-url", "http://h:1,http://g:2",
         "--client-budget-seconds", "5", "--out", "X"],
        ["--live", "--profile", "gemma_candidate", "--ollama-url", "http://h:1/api",
         "--client-budget-seconds", "5", "--out", "X"],
        ["--live", "--profile", "gemma_candidate", "--client-budget-seconds", "5", "--out", "X"],
        ["--live", "--profile", "unknown_profile", "--ollama-url", "http://h:1", "--client-budget-seconds", "5",
         "--out", "X"],
    ],
    ids=["nothing", "both-modes", "no-profile", "profile-list", "credentials", "url-list", "url-path",
         "no-url-even-with-environment", "unknown-profile"],
)
def test_a_live_run_needs_explicit_single_inputs(tmp_path, monkeypatch, args):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-environment:11434")
    args = [str(tmp_path / "out.jsonl") if a == "X" else a for a in args]
    with pytest.raises((gq.UsageError, base.HarnessUsageError)):
        gq.main(args, stdout=io.StringIO(), registry_factory=_synthetic_registry,
                supervisor_factory=lambda *a, **k: pytest.fail("no supervisor before inputs are valid"))
    assert not (tmp_path / "out.jsonl").exists()


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf", "3601"])
def test_the_client_budget_must_be_positive_and_finite(tmp_path, budget):
    with pytest.raises(gq.UsageError):
        gq.main(["--live", "--profile", GEMMA_CANDIDATE, "--ollama-url", "http://h:1", "--client-budget-seconds",
                 budget, "--out", str(tmp_path / "o.jsonl")], stdout=io.StringIO(),
                registry_factory=_synthetic_registry)
    for bad in (True, None, "5", float("nan"), float("inf"), 0, -2):
        with pytest.raises(gq.UsageError):
            gq.require_budget(bad)
    assert gq.require_budget(600) == 600.0


def test_llama_control_uses_the_same_harness_as_a_single_profile(tmp_path):
    run = _live(tmp_path, FakeOllama(), profile=LLAMA_CONTROL)
    assert {body["model"] for body in run.server.generate_bodies} == {"llama3.1:8b"}
    assert all("think" not in body for body in run.server.generate_bodies)
    assert run.summary["identity"]["observed_digest"] == SYNTHETIC_LLAMA


# --------------------------------------------------------------------------
# Fixture lists, expanded runs and grading separation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [GEMMA_CANDIDATE, LLAMA_CONTROL])
def test_the_default_run_is_v1_with_unchanged_prompts_and_payloads_for_both_profiles(tmp_path, profile):
    run = _live(tmp_path, FakeOllama(), profile=profile)
    assert run.summary["harness_revision"] == gq.HARNESS_REVISION == "gemma-qualification-v1"
    assert run.summary["scope_note"] == gq.SCOPE_NOTE and run.summary["baseline_note"] == gq.BASELINE_NOTE
    answers = {c["fixture"]: c["payload_sha256"] for c in _calls(run) if c["role"] == "answer"}
    assert answers == {fid: _PAYLOAD_SHAS[(profile, fid)] for fid in ("C04", "C05")}
    assert [(f["id"], f["fixture_sha256"], f["prompt_sha256"]) for f in run.summary["fixtures"]] == [
        (fid, _FIXTURE_SHAS[fid], _PROMPT_SHAS[fid]) for fid in ("C04", "C05")]
    assert len(_calls(run)) <= 7
    (tmp_path / "explicit").mkdir()
    explicit = _live(tmp_path / "explicit", FakeOllama(), profile=profile, extra=("--fixtures", "C04,C05"))
    assert explicit.summary["harness_revision"] == "gemma-qualification-v1"
    assert [c["payload_sha256"] for c in _calls(explicit)] == [c["payload_sha256"] for c in _calls(run)]


def test_an_expanded_run_reports_its_revision_case_list_and_at_most_1_plus_3n_calls(tmp_path):
    fixtures = ("C01", "C02", "C03", "C04", "C05", "C06")
    run = _live(tmp_path, FakeOllama(), extra=("--fixtures", ",".join(fixtures)))
    assert run.code == 0 and run.summary["outcome"] == "completed"
    assert run.summary["harness_revision"] == gq.EXPANDED_HARNESS_REVISION == "gemma-qualification-v2"
    assert [f["id"] for f in run.summary["fixtures"]] == list(fixtures)
    assert "C01, C02, C03, C04, C05, C06" in run.summary["scope_note"]
    assert "not reproductions of the historical website attempts" in run.summary["baseline_note"]
    assert "three-document" in run.summary["baseline_note"] and "L01 is not represented" in run.summary["baseline_note"]
    assert [c["fixture"] for c in _calls(run) if c["role"] == "answer"] == list(fixtures)
    assert len(_calls(run)) <= 1 + 3 * len(fixtures)
    c06 = next(f for f in run.summary["fixtures"] if f["id"] == "C06")
    assert c06["source_text_sha256s"][0] == _EVIDENCE_SHA and len(c06["source_text_sha256s"]) == 2
    assert all("source_text_sha256s" not in f for f in run.summary["fixtures"] if f["id"] != "C06")
    grading = {a["fixture"]: a for a in run.summary["anchors"]}
    assert grading["C06"]["citation_ids_present"] == {"C1": True, "C2": True}
    assert "citation_ids_present" not in grading["C04"]
    out = io.StringIO()
    gq.main(["--dry-run", "--fixtures", ",".join(fixtures)], stdout=out)
    report = json.loads(out.getvalue())
    assert report["harness_revision"] == "gemma-qualification-v2"
    assert report["planned_model_calls"] == {"smoke": 1, "answers": 6, "verifier_max": 12, "max_total": 19}


@pytest.mark.parametrize("value", ["C07", "C04,C04", "c04", "C04, C05", "", "C04,,C05", "../c04", "C4"])
def test_fixture_lists_are_validated_before_any_supervisor_output_or_network(tmp_path, value):
    out = tmp_path / "out.jsonl"
    with pytest.raises((gq.UsageError, base.FixtureError)):
        gq.main(["--live", "--profile", GEMMA_CANDIDATE, "--ollama-url", "http://fake-ollama:11434",
                 "--client-budget-seconds", "30", "--out", str(out), "--fixtures", value],
                stdout=io.StringIO(), registry_factory=_synthetic_registry,
                supervisor_factory=lambda *a, **k: pytest.fail("no supervisor before the fixtures are valid"))
    assert not out.exists() and not list(tmp_path.iterdir())


def test_grading_files_are_read_only_after_the_last_network_call_of_an_expanded_run(tmp_path, monkeypatch):
    server = FakeOllama()
    reads = []
    anchors = {base.GRADING_PATH.resolve(), base.BASELINE_GRADING_PATH.resolve()}
    original = Path.read_text

    def recording(self, *args, **kwargs):
        if self.resolve() in anchors:
            reads.append((self.name, len(server.requests)))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording)
    run = _live(tmp_path, server, extra=("--fixtures", "C01,C02,C03,C04,C05,C06"))
    total = len(run.server.requests)
    assert sorted(name for name, _ in reads) == ["baseline_gold_anchors.json", "ircot_gold_anchors.json"]
    assert all(at == total for _, at in reads), "no grading file is read before the last network call"
    gold = []
    for path in sorted(anchors):
        for spec in json.loads(original(path, encoding="utf-8"))["fixtures"].values():
            gold += spec["required_facts"] + spec.get("supplementary_facts", [])
    for body in run.server.generate_bodies:
        rendered = json.dumps(body)
        for fact in gold:
            assert fact not in body["prompt"] and fact not in rendered


def test_embedding_is_refused_unless_a_supervisor_explicitly_allows_it(tmp_path):
    server = FakeOllama()
    default = _supervisor(FakeLauncher(server), tmp_path)
    with pytest.raises(gq.AdapterError):
        default.call(model_call=True, method="POST", path=gq.EMBED_PATH, body={}, read_timeout_s=1,
                     max_body_bytes=1, watch_stream=False)
    allowed = _supervisor(FakeLauncher(server), tmp_path, allow_embedding=True)
    with pytest.raises(gq.AdapterError):
        allowed.call(model_call=False, method="POST", path=gq.EMBED_PATH, body={}, read_timeout_s=1,
                     max_body_bytes=1, watch_stream=False)
    assert default.events == [] and allowed.events == []
    assert gq.OBSERVATION_PATHS == {"/api/version", "/api/tags", "/api/show", "/api/ps"}


def test_a_per_call_budget_may_only_shorten_the_client_wait(tmp_path):
    waits = []

    class RecordingLauncher(FakeLauncher):
        def __call__(self, argv, **kwargs):
            process = super().__call__(argv, **kwargs)
            original_wait = process.wait

            def wait(timeout=None):
                waits.append(timeout)
                return original_wait(timeout)

            process.wait = wait
            return process

    supervisor = _supervisor(RecordingLauncher(FakeOllama()), tmp_path, budget_s=30)
    for budget, expected in ((5, 5.0), (90, 30.0), (None, 30.0)):
        supervisor.call(model_call=False, method="GET", path="/api/version", body=None, read_timeout_s=60,
                        max_body_bytes=1024, watch_stream=False, budget_s=budget)
        assert waits[-1] == expected
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(gq.UsageError):
            supervisor.call(model_call=False, method="GET", path="/api/version", body=None, read_timeout_s=1,
                            max_body_bytes=1024, watch_stream=False, budget_s=bad)


def test_the_harness_imports_and_endpoints_stay_inside_their_boundary():
    tree = ast.parse(_HARNESS_PATH.read_text(encoding="utf-8"))
    boundary = next(index for index, node in enumerate(tree.body)
                    if isinstance(node, ast.If) and "WORKER_FLAG" in ast.unparse(node.test))
    worker_roots, all_roots = set(), set()
    for index, node in enumerate(tree.body):
        for child in ast.walk(node):
            names = []
            if isinstance(child, ast.Import):
                names = [alias.name.split(".")[0] for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and child.module:
                names = [child.module.split(".")[0]]
            all_roots.update(names)
            if index < boundary:
                worker_roots.update(names)
    assert worker_roots <= {
        "__future__", "argparse", "ast", "builtins", "datetime", "hashlib", "http", "json", "math", "os", "re",
        "subprocess", "sys", "tempfile", "time", "urllib", "uuid", "dataclasses", "pathlib", "typing",
    }, "the worker half is standard library only"
    assert not all_roots & {"main", "dbSetup", "psycopg2", "pgvector", "dotenv", "requests", "socket", "signal",
                            "psutil", "threading"}

    source = _HARNESS_PATH.read_text(encoding="utf-8")
    for forbidden in ("os.kill", "killpg", "taskkill", "CREATE_NEW_PROCESS_GROUP", "start_new_session=True",
                      "shell=True", ".terminate(", "/api/pull", "/api/delete", "/api/copy", "/api/create",
                      "/api/push", "/api/chat", '"keep_alive": 0', "keep_alive=0", "os.environ.get(\"OLLAMA",
                      "load_dotenv"):
        assert forbidden not in source
    assert gq.OBSERVATION_PATHS == {"/api/version", "/api/tags", "/api/show", "/api/ps"}


# --------------------------------------------------------------------------
# The real supervisor against one inspected, model-free child process
# --------------------------------------------------------------------------

# Inspected test child. It blocks its own network before anything else, writes
# a probe (its pid, whether a loopback connect was refused, its environment
# NAMES and interpreter flags) and then never finishes: silent ("blocked":
# nothing before the first byte) or an endless slow drip into its body file.
_OWNED_TEST_CHILD = '''\
import json
import os
import socket
import sys
import time
from pathlib import Path


def _refuse(*args, **kwargs):
    raise OSError("network is blocked in this owned test child")


socket.socket.connect = _refuse
socket.socket.connect_ex = _refuse
socket.create_connection = _refuse
socket.getaddrinfo = _refuse

here = Path(__file__)
try:
    socket.create_connection(("127.0.0.1", 9), timeout=1)
    network_blocked = False
except OSError:
    network_blocked = True
probe = {
    "pid": os.getpid(),
    "network_blocked": network_blocked,
    "environment": sorted(os.environ),
    "flags": [sys.flags.isolated, sys.flags.no_site, sys.flags.dont_write_bytecode],
    "argv_tail": sys.argv[1:2],
}
here.with_suffix(".probe").write_text(json.dumps(probe), encoding="utf-8")
mode = here.with_suffix(".mode").read_text(encoding="utf-8").strip()
body = Path(sys.argv[-1])
while True:
    if mode == "drip":
        with open(body, "ab") as sink:
            sink.write(b"{")
    time.sleep(0.05)
'''


@pytest.mark.parametrize("mode", ["blocked", "drip"], ids=["blocked-before-first-byte", "endless-slow-drip"])
def test_the_real_supervisor_stops_waiting_kills_and_reaps_only_its_owned_child(tmp_path, mode):
    guard = sys.modules.get("sitecustomize")
    if getattr(guard, "SynergeGuardBlocked", None) is not None and not getattr(guard, "OWNED_CHILD_SHA256S", None):
        pytest.skip("this guard forbids every child process; run under the owned-child guard")
    child = tmp_path / "owned_child.py"
    child.write_text(_OWNED_TEST_CHILD, encoding="utf-8", newline="\n")
    child.with_suffix(".mode").write_text(mode, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    budget = 4.0
    supervisor = gq.OwnedWorkerSupervisor("http://127.0.0.1:9", budget_s=budget, work_root=work,
                                          worker_script=child)

    began = time.monotonic()
    outcome = supervisor.call(model_call=True, method="POST", path="/api/generate", body={"model": "none"},
                              read_timeout_s=60, max_body_bytes=1024, watch_stream=True)
    elapsed = time.monotonic() - began

    assert (outcome.category, outcome.state) == ("client_deadline", CallState.UNKNOWN.value)
    assert outcome.reaped is True and outcome.body is None
    assert budget <= elapsed < budget + gq.REAP_SECONDS + 5
    [event] = supervisor.events
    assert event["deadline"] is True and event["killed"] is True and event["reaped"] is True
    probe = json.loads(child.with_suffix(".probe").read_text(encoding="utf-8"))
    assert probe["pid"] == outcome.pid == event["pid"], "the killed process is the worker itself, not a launcher"
    assert probe["network_blocked"] is True
    # Only SYSTEMROOT and SYSTEMDRIVE are passed (asserted on the launch
    # arguments elsewhere); the Microsoft Store interpreter package also
    # injects PYTHONUSERBASE at process creation, which -I ignores.
    if sys.platform == "linux":
        # CPython adds LC_CTYPE inside the isolated Linux child.
        # The launch-shape test separately checks the supplied environment.
        assert set(probe["environment"]) == {"LC_CTYPE"}
    else:
        assert set(probe["environment"]) <= {
            "SYSTEMROOT", "SYSTEMDRIVE", "PYTHONUSERBASE"
        }
    assert probe["flags"] == [1, 1, 1] and probe["argv_tail"] == ["--worker"]
    assert list(work.iterdir()) == [], "the per-call work directory is removed after the reap"

    with pytest.raises(gq.BatchStopped):
        supervisor.call(model_call=True, method="POST", path="/api/generate", body={}, read_timeout_s=1,
                        max_body_bytes=1, watch_stream=False)
    assert len(supervisor.events) == 1, "nothing is started after UNKNOWN server execution"
