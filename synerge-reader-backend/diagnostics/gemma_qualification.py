#!/usr/bin/env python3
"""First bounded Gemma answer + citation-verifier evaluation (operator-run).

One pinned registry profile -- intended first: ``gemma_candidate`` /
``gemma4:31b`` -- for the whole batch:

1. Identity and thinking gates (the existing diagnostic's ``model_identity``
   and ``thinking_decision``): the installed digest must equal the profile's
   expected digest, and a ``think: false`` profile needs the server to report
   thinking capability. Anything else stops with no generation dispatched.
   Expected and observed digests are recorded separately.
2. One streamed smoke at the existing SMOKE_OPTIONS. It must be accepted
   exactly like an answer -- exact ``stop``, non-blank response text,
   thinking excluded, no error or malformed line -- or the batch stops
   (``smoke_rejected``, not UNKNOWN) before any later call.
3. C04, then C05: the frozen fixtures, the application's prompt builder and
   the profile's own answer settings, sent as the B1 answer snapshot sends
   them. An answer is accepted only on an exact ``stop`` completion with
   non-blank response text; thinking is never answer text. A rejected answer
   is not verified.
4. For each accepted answer, the existing citation pipeline
   (``citation_generation.generate_citations``) with the same profile's B1
   verifier: at most two sequential calls, never another model tag, with
   main.py's own verifier prompt, evidence assembly and status parser. Those
   are compiled from main.py's SOURCE (a source adapter, never an import of
   main.py and never the running /ask route).

Every network call runs in an owned, isolated worker process -- one at a
time, stdlib only, empty environment except SYSTEMROOT and SYSTEMDRIVE -- that this parent
supervises with a hard client wait budget. The parent stops waiting when the
budget expires even if the worker is blocked before the first byte or is
receiving an endless slow drip; it then terminates and reaps that one child
and nothing else. A model call whose client wait ended without the server's
terminal record is UNKNOWN server execution -- not a cancellation, not a
completion -- and stops the whole batch at once: no verifier, no next
fixture, no rerun, no other model, no reset.

Scope: C04/C05 use frozen manual evidence and native Ollama inference only.
No retrieval, embedding, database, chat history, rating, SynergeReader API,
login or upload is involved, and nothing here qualifies a model for
production; heuristic anchors and a green verifier status are not question
completeness, legal truth or qualification. By default only C04 and C05 run,
with their frozen identities (revision ``gemma-qualification-v1``).
``--fixtures`` runs another list of frozen manual-evidence fixtures in order
(revision ``gemma-qualification-v2``, at most 1 + 3n model calls); those are
newly prepared fixtures, not reproductions of historical website attempts.

Never: pull, delete, copy, create, push, unload or stop a model; signal any
process other than the worker this parent started; read a .env file,
credentials or an endpoint from the environment.

Usage (operator, inside an agreed single-profile resource window):

    python diagnostics/gemma_qualification.py --dry-run [--profile gemma_candidate]
    python diagnostics/gemma_qualification.py --live --profile gemma_candidate \\
        --ollama-url http://127.0.0.1:11434 --client-budget-seconds 600 \\
        --out C:/protected/gemma_c04_c05.jsonl
    python diagnostics/gemma_qualification.py --live --profile gemma_candidate \\
        --fixtures C01,C02,C03,C04,C05,C06 --ollama-url http://127.0.0.1:11434 \\
        --client-budget-seconds 600 --out C:/protected/gemma_c01_c06.jsonl

``--out`` receives protected raw answers; ``<out>.summary.json`` (or
``--summary-out``) receives a shareable summary of counts, hashes, safe
categories and timings only. Neither file is ever overwritten.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import datetime
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ==========================================================================
# Part 1 -- the owned worker. Runs in the CHILD process, started as
# ``python -I -S -B gemma_qualification.py --worker REQUEST RESULT BODY``,
# before any application module is imported (standard library only).
# ==========================================================================

WORKER_FLAG = "--worker"
WORKER_PROTOCOL = "gemma-qualification-worker-v1"
RESULT_LIMIT_BYTES = 64 * 1024
_READ_SIZE = 4096

# What a worker can report. Anything else is an IPC failure.
WORKER_CATEGORIES = frozenset({
    "end_of_body",                 # the response body was read to its end
    "not_sent_connect_failed",     # no connection, so no request byte left
    "worker_error",                # failed in the phase it reports
    "read_timeout_after_send",     # a socket read timed out after sending
    "transport_error_after_send",  # any other transport failure after sending
    "body_limit_exceeded",         # stopped reading at the byte bound
})
_BEFORE_SEND_PHASES = frozenset({"load", "prepare", "connect"})
_TIMING_KEYS = frozenset({"response_headers_s", "first_response_token_s", "done_record_s", "end_of_body_s"})


class _StreamWatch:
    """Child-side timing only: when the first non-empty ``response`` text and
    the first ``done`` record arrived. It never decides completion or
    acceptance -- the parent re-reads the whole body with the application's
    parser -- and thinking text never counts as an answer token."""

    def __init__(self, started: float, clock: Callable[[], float]) -> None:
        self._started = started
        self._clock = clock
        self._buffer = b""
        self.first_response_token_s = None
        self.done_record_s = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._line(line)

    def _line(self, line: bytes) -> None:
        try:
            record = json.loads(line)
        except ValueError:
            return
        if not isinstance(record, dict):
            return
        now = round(self._clock() - self._started, 3)
        text = record.get("response")
        if self.first_response_token_s is None and isinstance(text, str) and text:
            self.first_response_token_s = now
        if self.done_record_s is None and record.get("done") is True:
            self.done_record_s = now

    def timing(self) -> dict:
        return {"first_response_token_s": self.first_response_token_s, "done_record_s": self.done_record_s}


def _http_connection(scheme: str, host: str, port: Optional[int], timeout: float):
    cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    return cls(host, port, timeout=timeout)


def worker_call(request: dict, body_path: Path, result: dict, *,
                connection_factory: Callable = _http_connection,
                clock: Callable[[], float] = time.monotonic) -> dict:
    """One HTTP request to the Ollama root named in ``request``, recorded in
    ``result`` as it progresses (so a failure keeps the phase it reached).

    Only fixed categories, the HTTP status, byte counts, exception TYPE names
    and child-side timings are recorded -- never an exception message or a
    header. ``phase`` turns to ``send`` immediately before the request is
    written; from then on the request may have reached the server.
    """
    result["phase"] = "prepare"
    parsed = urllib.parse.urlsplit(request["base_url"])
    body = None if request.get("body") is None else json.dumps(request["body"]).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    limit = int(request["max_body_bytes"])
    connection = connection_factory(parsed.scheme, parsed.hostname, parsed.port, float(request["read_timeout_s"]))
    watch = None
    try:
        result["phase"] = "connect"
        try:
            connection.connect()
        except Exception as exc:
            result.update(category="not_sent_connect_failed", error_type=type(exc).__name__)
            return result
        result["phase"] = "send"
        started = clock()
        watch = _StreamWatch(started, clock) if request.get("watch_stream") else None
        try:
            connection.request(request["method"], request["path"], body=body, headers=headers)
            response = connection.getresponse()
            result["http_status"] = response.status
            result["timing"]["response_headers_s"] = round(clock() - started, 3)
            result["phase"] = "read"
            total = 0
            with open(body_path, "xb") as sink:
                while True:
                    chunk = response.read1(_READ_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        result.update(category="body_limit_exceeded", body_bytes=total)
                        return result
                    sink.write(chunk)
                    if watch is not None:
                        watch.feed(chunk)
            result["timing"]["end_of_body_s"] = round(clock() - started, 3)
            result.update(category="end_of_body", body_bytes=total)
        except TimeoutError:
            result.update(category="read_timeout_after_send")
        except Exception as exc:
            result.update(category="transport_error_after_send", error_type=type(exc).__name__)
        return result
    finally:
        if watch is not None:
            result["timing"].update(watch.timing())
        try:
            connection.close()
        except Exception:
            pass


def _new_worker_result() -> dict:
    return {
        "protocol": WORKER_PROTOCOL, "nonce": None, "phase": "load", "category": "worker_error",
        "error_type": None, "http_status": None, "body_bytes": 0, "timing": {},
    }


def _worker_main(argv) -> int:
    if len(argv) != 3:
        return 2
    request_path, result_path, body_path = (Path(value) for value in argv)
    result = _new_worker_result()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict) or request.get("protocol") != WORKER_PROTOCOL:
            raise ValueError("unexpected worker request")
        result["nonce"] = request.get("nonce")
        worker_call(request, body_path, result)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    with open(result_path, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result))
    return 0


if __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == WORKER_FLAG:
    sys.exit(_worker_main(sys.argv[2:]))


# ==========================================================================
# Part 2 -- the parent: plan, supervise, evaluate, report.
# ==========================================================================

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from citation_generation import (  # noqa: E402
    CitationLimits,
    CitationRegistry,
    build_generation_prompt,
    generate_citations,
)
from diagnostics import generation_diagnostic as base  # noqa: E402
from generation_admission import TERMINAL_DONE_REASONS, CallState, is_terminal_generate_reply  # noqa: E402
from generation_profiles import (  # noqa: E402
    GEMMA_CANDIDATE,
    SMOKE_OPTIONS,
    AnswerGenerationSnapshot,
    ModelProfile,
    ModelRegistry,
    _snapshot,
    default_model_registry,
)
from ollama_generation import OllamaGenerateStream, completed_generate_response  # noqa: E402

HARNESS_REVISION = "gemma-qualification-v1"
# Any fixture list other than the default C04, C05 is an expanded run.
EXPANDED_HARNESS_REVISION = "gemma-qualification-v2"
FIXTURES = ("C04", "C05")
INTENDED_FIRST_PROFILE = GEMMA_CANDIDATE
MAX_VERIFIER_CALLS_PER_ANSWER = 2
MAX_BUDGET_SECONDS = 3600.0
OBSERVATION_READ_TIMEOUT_S = 30.0
STREAM_BODY_LIMIT_BYTES = 8 * 1024 * 1024
JSON_BODY_LIMIT_BYTES = 4 * 1024 * 1024
REAP_SECONDS = 10.0
MAIN_PATH = _BACKEND_DIR / "main.py"
OBSERVATION_PATHS = frozenset({"/api/version", "/api/tags", "/api/show", "/api/ps"})
GENERATE_PATH = "/api/generate"
# Permitted only by a supervisor constructed with allow_embedding=True (the
# capacity probe); an embedding request is a model call like a generation.
EMBED_PATH = "/api/embed"

# main.py definitions the verifier adapter compiles, and the only names they
# may read besides each other and builtins.
APPLICATION_VERIFIER_DEFINITIONS = (
    "_CITATION_LIMITS",
    "_CLAIM_VERIFIER_PROMPT",
    "_CLAIM_VERIFIER_WORDS",
    "_claim_verifier_status",
    "_ollama_claim_verifier",
)
_ADAPTER_GLOBALS = frozenset({"re", "Optional", "CitationLimits", "completed_generate_response", "post_ollama"})
ADAPTER_LABEL = (
    "main.py source-compiled verifier adapter (_ollama_claim_verifier, _claim_verifier_status, "
    "_CLAIM_VERIFIER_PROMPT, _CLAIM_VERIFIER_WORDS, _CITATION_LIMITS); not the running /ask route, "
    "no database"
)
SCOPE_NOTE = (
    "C04/C05 use frozen manual evidence and native Ollama inference only: no retrieval, embedding, "
    "database, chat history, rating, login, upload or SynergeReader API. Not a production "
    "qualification; anchors and verifier statuses are not question completeness or legal truth."
)
BASELINE_NOTE = (
    "Only C04 and C05 run here, with their frozen identities (revision 2; pinned fixture, evidence "
    "and prompt hashes). The wider C01-C06 baseline is not represented: C01-C03 and C06 have no "
    "results from this diagnostic."
)
UNKNOWN_NOTE = (
    "UNKNOWN server execution: the client stopped waiting after the request may have reached the "
    "server. That is not a cancellation and not a completion. It ends this batch and its window: "
    "no automatic retry, unload, restart or reset. Keep-alive expiry, an empty /api/ps or idle GPUs "
    "do not prove the work ended. This harness never stops a model or a service."
)
RESIDENCY_LIMITS = (
    "GET /api/ps is a point-in-time view of what this Ollama server reports as loaded; it does not "
    "prove exclusive GPU use, other processes' memory, or that a model stopped working."
)

UsageError = base.HarnessUsageError


class BatchStopped(BaseException):
    """Ends the batch at once. A BaseException on purpose: the reused
    application code (``verify_claims``, ``model_identity``) catches
    ``Exception``, and nothing may absorb a stop and dispatch another call."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


class ObservationFailed(Exception):
    pass


class VerifierHttpError(Exception):
    pass


class AdapterError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha(data) -> str:
    return base.sha256_hex(data)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def require_budget(value) -> float:
    """A per-call client wait budget: a real, finite number of seconds in
    (0, MAX_BUDGET_SECONDS]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageError("--client-budget-seconds must be a number")
    if not math.isfinite(value) or not 0 < value <= MAX_BUDGET_SECONDS:
        raise UsageError(f"--client-budget-seconds must be finite and in (0, {MAX_BUDGET_SECONDS:g}]")
    return float(value)


def require_root_url(url: str) -> str:
    """Exactly one plain Ollama server root; never a list, never credentials."""
    if not isinstance(url, str) or "," in url or any(ch.isspace() for ch in url):
        raise UsageError("--ollama-url must be exactly one server root")
    return base.validate_base_url(url)


def require_profile(profile_id: str, registry: ModelRegistry) -> ModelProfile:
    if not isinstance(profile_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", profile_id):
        raise UsageError("--profile must be exactly one registry profile id")
    [profile] = base.load_profiles([profile_id], registry)
    return profile


@dataclass(frozen=True)
class FixturePlan:
    fixture: base.Fixture
    prompt: str
    registry: CitationRegistry

    @property
    def fixture_id(self) -> str:
        return self.fixture.fixture_id


def require_fixtures(value: Optional[str]) -> tuple:
    """The fixture ids of one batch, in order: the default C04, C05, or an
    explicit comma-separated list of distinct ids (each loaded and its pinned
    hashes checked before any network contact)."""
    if value is None:
        return FIXTURES
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise UsageError("--fixtures must be a comma-separated list such as C01,C05")
    ids = tuple(value.split(","))
    if any(not re.fullmatch(r"C\d{2}", fixture_id) for fixture_id in ids):
        raise UsageError("--fixtures takes fixture ids such as C01")
    if len(set(ids)) != len(ids):
        raise UsageError("--fixtures must not repeat a fixture")
    return ids


@dataclass(frozen=True)
class RunSettings:
    """What a batch reports about itself: the default C04/C05 batch keeps the
    first revision and notes; any other fixture list is an expanded run."""

    harness_revision: str
    scope_note: str
    baseline_note: str


def run_settings(fixture_ids) -> RunSettings:
    fixture_ids = tuple(fixture_ids)
    if fixture_ids == FIXTURES:
        return RunSettings(HARNESS_REVISION, SCOPE_NOTE, BASELINE_NOTE)
    listed = ", ".join(fixture_ids)
    return RunSettings(
        EXPANDED_HARNESS_REVISION,
        f"{listed}: frozen manual evidence and native Ollama inference only: no retrieval, embedding, database, "
        "chat history, rating, login, upload or SynergeReader API. Not a production qualification; anchors and "
        "verifier statuses are not question completeness or legal truth.",
        f"Manual-evidence fixtures {listed}. These are not reproductions of the historical website attempts, "
        "which used retrieval over uploaded documents; C06 here is the controlled two-paper scope (IRCoT and "
        "Step-Back), not the historical three-document attempt. L01 is not represented.",
    )


def plan_fixture(fixture: base.Fixture, limits: CitationLimits) -> FixturePlan:
    """The fixture's prompt and citation registry, built from the existing
    diagnostic's shared evidence builder; the prompt bytes are checked against
    that diagnostic's ``build_prompt``."""
    registry = CitationRegistry.from_bundle(base.evidence_bundle(fixture), limits)
    prompt = build_generation_prompt(fixture.question, registry)
    if prompt != base.build_prompt(fixture):
        raise AdapterError("fixture prompt differs from the existing diagnostic's prompt")
    return FixturePlan(fixture=fixture, prompt=prompt, registry=registry)


# --------------------------------------------------------------------------
# main.py verifier source adapter (compiled from source; main is never imported)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplicationVerifier:
    template: str
    words: tuple
    citation_limits: CitationLimits
    claim_verifier_status: Callable
    ollama_claim_verifier: Callable
    source_sha256: dict
    main_sha256: str
    label: str = ADAPTER_LABEL


def _defined_name(node) -> Optional[str]:
    if isinstance(node, ast.FunctionDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _free_names(node) -> set:
    loaded, bound = set(), set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            (loaded if isinstance(child.ctx, ast.Load) else bound).add(child.id)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node:
            bound.add(child.name)
    return loaded - bound - set(dir(builtins))


def load_application_verifier(post_ollama: Callable, main_path: Path = MAIN_PATH) -> ApplicationVerifier:
    """Compile exactly the five verifier definitions out of main.py's source.

    main.py itself is never imported or executed. Each definition must appear
    once at module level, and together they may read only each other,
    builtins and ``_ADAPTER_GLOBALS``; ``post_ollama`` is the supplied
    transport. Anything else -- a renamed definition, a new dependency such
    as a database call -- fails closed here.
    """
    raw = Path(main_path).read_bytes()
    source = raw.decode("utf-8").replace("\r\n", "\n")
    tree = ast.parse(source, filename=str(main_path))
    found = {}
    for node in tree.body:
        name = _defined_name(node)
        if name in APPLICATION_VERIFIER_DEFINITIONS:
            if name in found:
                raise AdapterError(f"{name} is defined more than once in main.py")
            found[name] = node
    missing = [name for name in APPLICATION_VERIFIER_DEFINITIONS if name not in found]
    if missing:
        raise AdapterError(f"main.py no longer defines {missing}")
    allowed = _ADAPTER_GLOBALS | set(APPLICATION_VERIFIER_DEFINITIONS)
    for name, node in found.items():
        extra = _free_names(node) - allowed
        if extra:
            raise AdapterError(f"{name} reads names outside the adapter boundary: {sorted(extra)}")
    ordered = [found[name] for name in APPLICATION_VERIFIER_DEFINITIONS]
    namespace = {
        "__builtins__": builtins,
        "__name__": "synerge_main_verifier_adapter",
        "re": re,
        "Optional": Optional,
        "CitationLimits": CitationLimits,
        "completed_generate_response": completed_generate_response,
        "post_ollama": post_ollama,
    }
    exec(compile(ast.Module(body=ordered, type_ignores=[]), str(main_path), "exec"), namespace)  # noqa: S102
    limits = namespace["_CITATION_LIMITS"]
    if not isinstance(limits, CitationLimits):
        raise AdapterError("_CITATION_LIMITS is not CitationLimits")
    return ApplicationVerifier(
        template=namespace["_CLAIM_VERIFIER_PROMPT"],
        words=tuple(namespace["_CLAIM_VERIFIER_WORDS"]),
        citation_limits=limits,
        claim_verifier_status=namespace["_claim_verifier_status"],
        ollama_claim_verifier=namespace["_ollama_claim_verifier"],
        source_sha256={name: _sha(ast.get_source_segment(source, found[name])) for name in APPLICATION_VERIFIER_DEFINITIONS},
        main_sha256=_sha(raw),
    )


def _refuse_dispatch(*args, **kwargs):
    raise BatchStopped("dispatch_not_permitted")


# --------------------------------------------------------------------------
# The owned-worker supervisor
# --------------------------------------------------------------------------


CHILD_ENVIRONMENT_NAMES = ("SYSTEMROOT", "SYSTEMDRIVE")


class StopLatch:
    """The one stop decision a supervisor consults before it starts a worker.

    Each supervisor owns one by default. A caller that runs several
    supervisors at once (the capacity probe) passes them one shared latch
    whose ``__enter__``/``__exit__`` hold a lock, so a stop recorded by any of
    them fences every later start at once."""

    def __init__(self) -> None:
        self.stopped: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def stop(self, category: str) -> None:
        with self:
            if self.stopped is None:
                self.stopped = category


def child_environment() -> dict:
    """The worker's whole environment: SYSTEMROOT and SYSTEMDRIVE only.
    Windows networking needs the first; without the second, Windows expands
    ``%SystemDrive%`` in its own cache paths literally and creates that
    folder in the worker's working directory. No proxy, DSN, credential, PG*
    or PYTHON* variable reaches the worker."""
    return {name: os.environ[name] for name in CHILD_ENVIRONMENT_NAMES if os.environ.get(name)}


def worker_python() -> str:
    """The base interpreter, not a virtual-environment redirector: on Windows
    ``venv\\Scripts\\python.exe`` starts the real interpreter as its own
    child, and terminating the redirector need not stop that grandchild. The
    worker needs only the standard library (it runs with -I -S)."""
    return getattr(sys, "_base_executable", None) or sys.executable


@dataclass(frozen=True)
class WorkerOutcome:
    category: str
    state: Optional[str]          # model calls: a CallState value, or None until evaluated
    phase: Optional[str]
    http_status: Optional[int]
    body: Optional[bytes]
    timing: dict
    wall_s: float
    pid: Optional[int]
    reaped: bool
    error_type: Optional[str] = None


class OwnedWorkerSupervisor:
    """Runs each network call in one owned child process, one at a time.

    The parent waits at most ``budget_s`` for the child to exit, whatever the
    child is doing. On expiry it terminates and reaps that child -- the
    process object it started, never a pid, process group, service or other
    process -- and the call's outcome is ``client_deadline``. IPC is bounded:
    the child writes a small result file and a body file the parent reads up
    to fixed byte limits; its standard streams are discarded.

    Any outcome other than a body read to its end latches ``stopped``; every
    later call then raises BatchStopped before a process is started.

    Model endpoints are ``/api/generate`` only, unless the supervisor is
    constructed with ``allow_embedding=True``, which also permits
    ``/api/embed`` as a model call (the capacity probe's one embedding).
    """

    def __init__(self, base_url: str, *, budget_s: float, work_root: Optional[Path] = None,
                 launcher: Callable = subprocess.Popen, python: Optional[str] = None,
                 worker_script: Optional[Path] = None, clock: Callable[[], float] = time.monotonic,
                 reap_s: float = REAP_SECONDS, latch: Optional[StopLatch] = None,
                 allow_embedding: bool = False) -> None:
        self.base_url = require_root_url(base_url)
        self.budget_s = require_budget(budget_s)
        self._work_root = None if work_root is None else str(work_root)
        self._launcher = launcher
        self._python = python or worker_python()
        self._worker_script = Path(worker_script) if worker_script is not None else Path(__file__).resolve()
        self._clock = clock
        self._reap_s = reap_s
        self._active = False
        self._latch = latch if latch is not None else StopLatch()
        self._model_paths = frozenset({GENERATE_PATH, EMBED_PATH} if allow_embedding is True else {GENERATE_PATH})
        self.events = []  # process-level log: pid, deadline, kill, reap; no content

    @property
    def stopped(self) -> Optional[str]:
        return self._latch.stopped

    def stop(self, category: str) -> None:
        self._latch.stop(category)

    def call(self, *, model_call: bool, method: str, path: str, body: Optional[dict],
             read_timeout_s: float, max_body_bytes: int, watch_stream: bool,
             budget_s: Optional[float] = None) -> WorkerOutcome:
        """One worker call. ``budget_s`` may only shorten this supervisor's
        client wait for this one call (a caller's whole-run deadline)."""
        if budget_s is not None:
            budget_s = min(require_budget(budget_s), self.budget_s)
        with self._latch:
            # Checked and claimed under the latch, so a stop recorded by any
            # supervisor sharing it fences every later start.
            if self._latch.stopped is not None:
                raise BatchStopped(self._latch.stopped)
            if path not in self._model_paths and path not in OBSERVATION_PATHS:
                raise AdapterError("only native observation endpoints and /api/generate are permitted")
            if path in self._model_paths and not model_call:
                raise AdapterError(f"{path} is a model call")
            if self._active:
                raise AdapterError("one worker at a time")
            self._active = True
        try:
            outcome = self._run(model_call, method, path, body, read_timeout_s, max_body_bytes, watch_stream,
                                self.budget_s if budget_s is None else budget_s)
        finally:
            self._active = False
        if outcome.category != "end_of_body":
            self.stop(outcome.category)
        return outcome

    def _run(self, model_call, method, path, body, read_timeout_s, max_body_bytes, watch_stream,
             budget_s) -> WorkerOutcome:
        nonce = uuid.uuid4().hex
        workdir = Path(tempfile.mkdtemp(prefix="gq-", dir=self._work_root))
        files = (workdir / "request.json", workdir / "result.json", workdir / "body.bin")
        request_path, result_path, body_path = files
        request = {
            "protocol": WORKER_PROTOCOL, "nonce": nonce, "base_url": self.base_url, "method": method,
            "path": path, "body": body, "read_timeout_s": min(float(read_timeout_s), budget_s),
            "max_body_bytes": int(max_body_bytes), "watch_stream": bool(watch_stream),
        }
        started = self._clock()
        event = {"path": path, "model_call": model_call, "pid": None, "deadline": False, "killed": False, "reaped": None}
        self.events.append(event)

        def outcome(category, *, phase=None, status=None, payload=None, timing=None, reaped=True, error_type=None):
            state = None
            if model_call and category != "end_of_body":
                never_sent = category in ("worker_start_failed", "not_sent_connect_failed") or (
                    category == "worker_error" and phase in _BEFORE_SEND_PHASES
                )
                state = (CallState.NEVER_DISPATCHED if never_sent else CallState.UNKNOWN).value
            return WorkerOutcome(
                category=category, state=state, phase=phase, http_status=status, body=payload,
                timing=dict(timing or {}), wall_s=round(self._clock() - started, 3), pid=event["pid"],
                reaped=reaped, error_type=error_type,
            )

        try:
            with open(request_path, "x", encoding="utf-8") as handle:
                handle.write(json.dumps(request))
            argv = [self._python, "-I", "-S", "-B", str(self._worker_script), WORKER_FLAG,
                    str(request_path), str(result_path), str(body_path)]
            try:
                process = self._launcher(
                    argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env=child_environment(), cwd=str(workdir), close_fds=True,
                )
            except Exception as exc:
                return outcome("worker_start_failed", phase="not_started", error_type=type(exc).__name__)
            event["pid"] = getattr(process, "pid", None)
            try:
                process.wait(timeout=budget_s)
            except subprocess.TimeoutExpired:
                event["deadline"] = True
                event["reaped"] = self._stop_owned(process, event)
                return outcome("client_deadline", reaped=event["reaped"])
            except BaseException:
                # An operator interrupt: stop this one child, then re-raise.
                event["reaped"] = self._stop_owned(process, event)
                raise
            event["reaped"] = True
            if process.returncode != 0:
                return outcome("worker_exit_nonzero")
            return self._read_result(outcome, nonce, result_path, body_path, max_body_bytes)
        finally:
            self._cleanup(workdir, files, event)

    def _stop_owned(self, process, event) -> bool:
        """Terminate and reap the one child this supervisor started."""
        try:
            process.kill()
            event["killed"] = True
        except OSError:
            pass
        try:
            process.wait(timeout=self._reap_s)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _read_result(self, outcome, nonce, result_path, body_path, max_body_bytes) -> WorkerOutcome:
        raw = _read_bounded(result_path, RESULT_LIMIT_BYTES)
        if raw is None:
            return outcome("ipc_invalid")
        if raw is _OVERSIZE:
            return outcome("ipc_oversize")
        try:
            result = json.loads(raw.decode("utf-8"))
        except ValueError:
            return outcome("ipc_invalid")
        if (not isinstance(result, dict) or result.get("protocol") != WORKER_PROTOCOL
                or result.get("nonce") != nonce or result.get("category") not in WORKER_CATEGORIES):
            return outcome("ipc_invalid")
        status = result.get("http_status")
        timing = result.get("timing")
        error_type = result.get("error_type")
        if (status is not None and type(status) is not int) or not isinstance(timing, dict) \
                or set(timing) - _TIMING_KEYS \
                or any(v is not None and type(v) not in (int, float) for v in timing.values()) \
                or (error_type is not None and not (isinstance(error_type, str) and error_type.isidentifier())):
            return outcome("ipc_invalid")
        category, phase = result["category"], result.get("phase")
        if category != "end_of_body":
            return outcome(category, phase=phase, status=status, timing=timing, error_type=error_type)
        payload = _read_bounded(body_path, max_body_bytes)
        if payload is _OVERSIZE:
            return outcome("ipc_oversize", phase=phase, status=status, timing=timing)
        payload = payload if payload is not None else b""
        if len(payload) != result.get("body_bytes"):
            return outcome("ipc_invalid", phase=phase, status=status, timing=timing)
        return outcome("end_of_body", phase=phase, status=status, payload=payload, timing=timing)

    def _cleanup(self, workdir: Path, files, event) -> None:
        try:
            for path in files:
                path.unlink(missing_ok=True)
            workdir.rmdir()
        except OSError:
            event["workdir_retained"] = True


_OVERSIZE = object()


def _read_bounded(path: Path, limit: int):
    try:
        with open(path, "rb") as handle:
            data = handle.read(limit + 1)
    except FileNotFoundError:
        return None
    return _OVERSIZE if len(data) > limit else data


class SupervisedHttp:
    """The existing diagnostic's ``http`` interface (get_json / post_json)
    over the supervisor, for its identity and residency observations."""

    def __init__(self, supervisor: OwnedWorkerSupervisor) -> None:
        self.supervisor = supervisor
        self.base_url = supervisor.base_url

    def get_json(self, path: str) -> dict:
        return self._json("GET", path, None)

    def post_json(self, path: str, body: dict) -> dict:
        return self._json("POST", path, body)

    def _json(self, method: str, path: str, body: Optional[dict]) -> dict:
        if path not in OBSERVATION_PATHS:
            raise AdapterError("not an observation endpoint")
        outcome = self.supervisor.call(
            model_call=False, method=method, path=path, body=body,
            read_timeout_s=OBSERVATION_READ_TIMEOUT_S, max_body_bytes=JSON_BODY_LIMIT_BYTES, watch_stream=False,
        )
        if outcome.category != "end_of_body" or outcome.http_status != 200:
            self.supervisor.stop("observation_failed")
            raise ObservationFailed(outcome.category if outcome.category != "end_of_body" else "http_status")
        try:
            parsed = json.loads(outcome.body.decode("utf-8"))
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            self.supervisor.stop("observation_failed")
            raise ObservationFailed("unreadable_json")
        return parsed


# --------------------------------------------------------------------------
# Model calls and their evaluation
# --------------------------------------------------------------------------


def _call_record(seq: int, role: str, fixture_id: Optional[str], payload: dict, outcome: WorkerOutcome) -> dict:
    return {
        "seq": seq,
        "role": role,
        "fixture": fixture_id,
        "model": payload.get("model"),
        "payload_sha256": _sha(base._canonical(payload)),
        "options": payload.get("options"),
        "think": payload.get("think", "omitted"),
        "stream": payload.get("stream"),
        "category": outcome.category,
        "state": outcome.state,
        "http_status": outcome.http_status,
        "error_type": outcome.error_type,
        "body_bytes": None if outcome.body is None else len(outcome.body),
        "body_sha256": None if outcome.body is None else _sha(outcome.body),
        "timing": outcome.timing,
        "wall_s": outcome.wall_s,
        "worker_reaped": outcome.reaped,
    }


class BatchTransport:
    """Every model call of one batch, in order, through one supervisor.
    ``emit`` receives each call record once it is settled."""

    def __init__(self, supervisor: OwnedWorkerSupervisor, emit: Callable[[dict], None] = lambda record: None) -> None:
        self.supervisor = supervisor
        self.emit = emit
        self.calls = []
        self.context = {"fixture": None}

    def _settle(self, record: dict, terminal: bool) -> None:
        record["state"] = (CallState.OBSERVED_TERMINAL if terminal else CallState.UNKNOWN).value
        if not terminal:
            # The server ended the response without its terminal record: the
            # model may still be working. Nothing else is dispatched.
            self.supervisor.stop("no_terminal_evidence")

    def stream_generate(self, role: str, fixture_id: Optional[str], payload: dict, read_timeout_s: float,
                        **extra) -> dict:
        outcome = self.supervisor.call(
            model_call=True, method="POST", path=GENERATE_PATH, body=payload, read_timeout_s=read_timeout_s,
            max_body_bytes=STREAM_BODY_LIMIT_BYTES, watch_stream=True,
        )
        record = _call_record(len(self.calls) + 1, role, fixture_id, payload, outcome)
        record.update(extra)
        self.calls.append(record)
        if outcome.category == "end_of_body":
            reader = OllamaGenerateStream()
            text = "".join(reader.feed(outcome.body) + reader.close())
            # Native terminal evidence: the final record (done:true with "stop"
            # or "length") was parsed before any error or malformed line.
            terminal = (outcome.http_status == 200 and reader.done
                        and reader.done_reason in TERMINAL_DONE_REASONS and not reader.model_error)
            self._settle(record, terminal)
            record.update(
                done_reason=reader.done_reason,
                failure=reader.failure,
                # Strict acceptance: exactly "stop", no error or malformed line,
                # and non-blank response text. Thinking is never answer text.
                accepted=terminal and reader.failure is None and bool(text.strip()),
                response_chars=len(text),
                response_sha256=_sha(text),
                thinking_chars=reader.thinking_chars,
                server_metrics=dict(reader.metrics),
                text=text,
            )
        self.emit(record)
        return record

    def post_ollama(self, endpoint: str, payload: dict, *, stream: bool = False, timeout: float = 60):
        """main.py's ``post_ollama`` signature, for the compiled verifier."""
        if endpoint != GENERATE_PATH or stream:
            raise BatchStopped("verifier_call_shape")
        outcome = self.supervisor.call(
            model_call=True, method="POST", path=GENERATE_PATH, body=payload, read_timeout_s=timeout,
            max_body_bytes=JSON_BODY_LIMIT_BYTES, watch_stream=False,
        )
        record = _call_record(len(self.calls) + 1, "verifier", self.context["fixture"], payload, outcome)
        self.calls.append(record)
        if outcome.category != "end_of_body":
            self.emit(record)
            raise BatchStopped(self.supervisor.stopped or outcome.category)
        try:
            body = json.loads(outcome.body.decode("utf-8"))
        except ValueError:
            body = None
        self._settle(record, outcome.http_status == 200 and is_terminal_generate_reply(body))
        record["done_reason"] = body.get("done_reason") if isinstance(body, dict) else None
        self.emit(record)
        return _WorkerHttpResponse(outcome.http_status, outcome.body)


class _WorkerHttpResponse:
    """What main.py's verifier reads from ``post_ollama``'s return value."""

    def __init__(self, status_code: Optional[int], body: bytes) -> None:
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code is None or self.status_code >= 400:
            raise VerifierHttpError(f"HTTP status {self.status_code}")

    def json(self):
        return json.loads(self._body.decode("utf-8"))


def run_verification(answer_text: str, plan: FixturePlan, snapshot: AnswerGenerationSnapshot,
                     adapter: ApplicationVerifier, transport: BatchTransport) -> dict:
    """The existing citation pipeline with this answer's own B1 verifier.

    Statuses, reasons and the exception mapping are ``verify_claims``'s own;
    this wrapper only counts calls and records what main.py's verifier
    returned or which exception TYPE it raised.
    """
    transport.context["fixture"] = plan.fixture_id
    calls = []

    def verify(claim, records):
        if len(calls) >= MAX_VERIFIER_CALLS_PER_ANSWER:
            raise BatchStopped("verifier_call_limit")
        entry = {"index": len(calls) + 1, "claim_sha256": _sha(claim),
                 "citation_ids": [record.citation_id for record in records]}
        calls.append(entry)
        try:
            word = adapter.ollama_claim_verifier(claim, records, snapshot.verifier)
        except Exception as exc:
            entry["raised"] = type(exc).__name__
            raise
        entry["status_word"] = word
        return word

    result = generate_citations(answer_text, plan.registry, verify, adapter.citation_limits)
    return {
        "fixture": plan.fixture_id,
        "claims": [claim.to_dict() for claim in result.claims],
        "used_citation_ids": list(result.used_citation_ids),
        "invalid_citation_ids": list(result.invalid_citation_ids),
        "verifier_calls": calls,
    }


# --------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------


def _qualification_flags(profile: ModelProfile) -> dict:
    qualification = profile.qualification
    return {
        "production_selectable": profile.selectable,
        "requires_qualification": profile.requires_qualification,
        "think_false_verified": getattr(qualification, "think_false_verified", None),
        "combined_capacity_verified": getattr(qualification, "combined_capacity_verified", None),
        "quality_verified": getattr(qualification, "quality_verified", None),
        "automatic_qualification": False,
    }


class _Sink:
    def __init__(self, handle, stdout) -> None:
        self.handle = handle
        self.stdout = stdout

    def write(self, record: dict) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def say(self, line: str) -> None:
        self.stdout.write(line + "\n")


def _status(record: dict) -> str:
    """A console line with categories and timings only -- never text."""
    return (f"{record['role']} {record['fixture'] or '-'}: category={record['category']} "
            f"state={record['state']} accepted={record.get('accepted')} "
            f"first_token_s={record['timing'].get('first_response_token_s')} "
            f"end_of_body_s={record['timing'].get('end_of_body_s')} wall_s={record['wall_s']}")


def run_batch(*, profile: ModelProfile, supervisor: OwnedWorkerSupervisor, sink: _Sink,
              plans, adapter_factory: Callable[[Callable], ApplicationVerifier] = load_application_verifier,
              clock: Callable[[], float] = time.monotonic, now: Callable[[], str] = _utc_now,
              grading_path: Optional[Path] = None, settings: Optional[RunSettings] = None) -> dict:
    """Identity, smoke, then each fixture in order -- sequentially, one model
    call at a time, each only after the previous one showed native terminal
    evidence, and the answers only after the smoke was accepted (exact "stop",
    non-blank text)."""
    settings = settings or run_settings(p.fixture_id for p in plans)

    def emit(record: dict) -> None:
        sink.write({"record_type": "call", **record})
        sink.say(_status(record))

    transport = BatchTransport(supervisor, emit)
    adapter = adapter_factory(transport.post_ollama)
    http = SupervisedHttp(supervisor)
    run_id = f"gq-{uuid.uuid4().hex}"
    sink.write({
        "record_type": "run", "run_id": run_id, "harness_revision": settings.harness_revision, "started_at": now(),
        "ollama_url": supervisor.base_url, "profile_id": profile.profile_id, "model_tag": profile.model_tag,
        "client_budget_seconds": supervisor.budget_s, "fixtures": [p.fixture_id for p in plans],
        "verifier_adapter": adapter.label, "verifier_source_sha256": adapter.source_sha256,
        "main_sha256": adapter.main_sha256, "scope_note": settings.scope_note,
        "baseline_note": settings.baseline_note,
        "persists_answers_in_application": False, "qualification": _qualification_flags(profile),
    })
    started = clock()
    identity, decision, residency = None, None, {}
    answers, verifications = {}, {}
    outcome, stop_category = "completed", None
    try:
        identity = base.model_identity(http, profile)
        decision = base.thinking_decision("profile", profile, identity)
        sink.write({"record_type": "identity", "identity": identity, "thinking_policy": decision})
        if identity["check"] != "match":
            supervisor.stop(f"identity_{identity['check']}")
            raise BatchStopped(supervisor.stopped)
        if decision["refused"]:
            supervisor.stop("thinking_capability_missing")
            raise BatchStopped(supervisor.stopped)
        if decision["send"] != profile.think:
            supervisor.stop("think_parity")
            raise BatchStopped(supervisor.stopped)
        snapshot = _snapshot(profile, "admin_override", identity["observed_digest"])
        residency["before_smoke"] = base.residency(http)

        transport.context["fixture"] = None
        smoke = transport.stream_generate(
            "smoke", None, base.request_body(profile, base.SMOKE_PROMPT, think=decision["send"], options=SMOKE_OPTIONS),
            profile.answer_read_timeout_s, prompt_sha256=_sha(base.SMOKE_PROMPT),
        )
        if smoke["state"] != CallState.OBSERVED_TERMINAL.value:
            raise BatchStopped(supervisor.stopped)
        if not smoke["accepted"]:
            # Terminal, so the server finished this call (not UNKNOWN), but it
            # is not a clean finished reply: "length", blank, whitespace or
            # thinking-only text, or an error or malformed line. The smoke must
            # be accepted exactly like an answer before anything else is sent.
            supervisor.stop("smoke_rejected")
            raise BatchStopped(supervisor.stopped)
        residency["after_smoke"] = base.residency(http)

        for plan in plans:
            transport.context["fixture"] = plan.fixture_id
            answer = transport.stream_generate(
                "answer", plan.fixture_id, snapshot.ollama_payload(plan.prompt, stream=True),
                snapshot.answer_read_timeout_s, prompt_sha256=_sha(plan.prompt),
            )
            answers[plan.fixture_id] = answer
            if answer["state"] != CallState.OBSERVED_TERMINAL.value:
                raise BatchStopped(supervisor.stopped)
            if not answer["accepted"]:
                # Terminal but not a finished answer (length, empty, error or
                # malformed): nothing to verify; the next fixture may run.
                verifications[plan.fixture_id] = {"fixture": plan.fixture_id, "skipped": "answer_not_accepted"}
                sink.write({"record_type": "verification", **verifications[plan.fixture_id]})
                continue
            verification = run_verification(answer["text"], plan, snapshot, adapter, transport)
            verifications[plan.fixture_id] = verification
            sink.write({"record_type": "verification", **verification})
        residency["after_batch"] = base.residency(http)
    except BatchStopped as stop:
        outcome, stop_category = "stopped", stop.category
    if outcome == "completed" and supervisor.stopped is not None:
        # Every model call finished; only the closing residency read failed.
        stop_category = supervisor.stopped
    total_s = round(clock() - started, 3)

    unknown = any(c["state"] == CallState.UNKNOWN.value for c in transport.calls)
    # Grading happens only now, after the last network call.
    grading = grade_answers(answers, profile, grading_path)
    sink.write({"record_type": "post_hoc_grading", "grading": grading,
                "note": "Heuristic anchors only; not question completeness, legal truth or qualification."})
    sink.write({"record_type": "manual_assessment", "fixtures": [
        {"fixture": fid, "assessor": None, "answers_every_part": None, "citations_appropriate": None,
         "unsupported_statements": None, "notes": None}
        for fid in answers if answers[fid].get("accepted")
    ]})
    summary = build_summary(
        run_id=run_id, profile=profile, supervisor=supervisor, identity=identity, decision=decision,
        transport=transport, answers=answers, verifications=verifications, grading=grading,
        residency=residency, outcome=outcome, stop_category=stop_category, unknown=unknown, total_s=total_s,
        adapter=adapter, plans=plans, settings=settings,
    )
    sink.write({"record_type": "end", "finished_at": now(), "outcome": outcome, "stop_category": stop_category,
                "unknown_server_execution": unknown, "note": UNKNOWN_NOTE if unknown else None})
    sink.say(f"batch {outcome} stop={stop_category} unknown_server_execution={unknown} total_s={total_s}")
    return summary


def grade_answers(answers: dict, profile: ModelProfile, grading_path: Optional[Path]) -> list:
    """Heuristic anchors, read only now -- after the last network call. The
    default is every gold-anchor file merged; ``grading_path`` names one."""
    if not answers:
        return []
    anchors = base.load_anchors() if grading_path is None else base.load_anchors((Path(grading_path),))
    graded = []
    for fixture_id, answer in answers.items():
        if not answer.get("accepted"):
            continue
        graded.append(base.grade_record({
            "record_type": "generation", "generated": True, "evaluation_id": f"{fixture_id}-{answer['seq']}",
            "fixture": {"id": fixture_id}, "profile": {"profile_id": profile.profile_id},
            "response": {"text": answer["text"]}, "stream": {"failure": answer["failure"]}, "error": None,
        }, anchors))
    return [g for g in graded if g is not None]


def _counts(values) -> dict:
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _fixture_entry(plan: FixturePlan) -> dict:
    """A fixture's identity for reports: hashes only. A multi-source fixture
    also lists each excerpt's text hash in citation order."""
    fixture = plan.fixture
    entry = {"id": fixture.fixture_id, "revision": fixture.revision, "fixture_sha256": fixture.fixture_sha256,
             "evidence_sha256": fixture.evidence_sha256, "prompt_sha256": _sha(plan.prompt)}
    if fixture.evidence is None:
        entry["source_text_sha256s"] = list(fixture.source_text_sha256s)
    return entry


def build_summary(*, run_id, profile, supervisor, identity, decision, transport, answers, verifications,
                  grading, residency, outcome, stop_category, unknown, total_s, adapter, plans,
                  settings: Optional[RunSettings] = None) -> dict:
    """Shareable: counts, hashes, safe categories and timings only."""
    settings = settings or run_settings(p.fixture_id for p in plans)
    safe_calls = [
        {k: c.get(k) for k in ("seq", "role", "fixture", "model", "payload_sha256", "category", "state",
                                "http_status", "error_type", "body_bytes", "body_sha256", "timing", "wall_s",
                                "worker_reaped", "done_reason", "failure", "accepted", "response_chars",
                                "response_sha256", "thinking_chars")}
        for c in transport.calls
    ]
    verifier_seconds = {}
    for c in transport.calls:
        if c["role"] == "verifier":
            verifier_seconds.setdefault(c["fixture"], []).append(c["timing"].get("end_of_body_s"))
    smoke = next((c for c in transport.calls if c["role"] == "smoke"), None)
    return {
        "record_type": "summary",
        "run_id": run_id,
        "harness_revision": settings.harness_revision,
        "scope_note": settings.scope_note,
        "baseline_note": settings.baseline_note,
        "profile": {"profile_id": profile.profile_id, "model_tag": profile.model_tag,
                    **_qualification_flags(profile)},
        "identity": None if identity is None else {
            "check": identity.get("check"),
            "expected_digest": identity.get("expected_digest"),
            "observed_digest": identity.get("observed_digest"),
            "runtime_version": identity.get("runtime_version"),
            "reports_thinking": None if decision is None else decision.get("server_reports_thinking"),
            "think_sent": None if decision is None else decision.get("send"),
            "observation_errors": sorted((identity.get("errors") or {}).keys()),
        },
        "outcome": outcome,
        "stop_category": stop_category,
        "unknown_server_execution": unknown,
        "unknown_note": UNKNOWN_NOTE if unknown else None,
        "fixtures": [_fixture_entry(p) for p in plans],
        "calls": safe_calls,
        "verification": [
            {"fixture": fid, "skipped": v.get("skipped"),
             "claim_status_counts": _counts(c["status"] for c in v.get("claims", [])),
             "claim_reason_counts": _counts(c["reason"] for c in v.get("claims", [])),
             "verifier_calls": len(v.get("verifier_calls", [])),
             "verifier_application_exceptions": [e.get("raised") for e in v.get("verifier_calls", []) if e.get("raised")]}
            for fid, v in verifications.items()
        ],
        "anchors": [{"fixture": g["fixture"], "groups_hit": sum(1 for x in g["anchor_groups"] if x["hit"]),
                     "groups_total": len(g["anchor_groups"]), "cites_supplied_page": g["cites_supplied_page"],
                     **({"citation_ids_present": g["citation_ids_present"]} if "citation_ids_present" in g else {})}
                    for g in grading],
        "timing": {
            "smoke_first_token_s": None if smoke is None else smoke["timing"].get("first_response_token_s"),
            "smoke_end_of_body_s": None if smoke is None else smoke["timing"].get("end_of_body_s"),
            "answer_first_token_s": {f: a["timing"].get("first_response_token_s") for f, a in answers.items()},
            "answer_s": {f: a["timing"].get("end_of_body_s") for f, a in answers.items()},
            "verifier_s": verifier_seconds,
            "total_diagnostic_s": total_s,
            "note": "Child-side times run from sending the request; wall_s adds worker start and reap.",
        },
        "residency": residency,
        "residency_limits": RESIDENCY_LIMITS,
        "verifier_adapter": {"label": adapter.label, "source_sha256": adapter.source_sha256,
                             "template_sha256": _sha(adapter.template), "max_calls_per_answer": MAX_VERIFIER_CALLS_PER_ANSWER},
        "worker_events": [{k: e.get(k) for k in ("path", "model_call", "deadline", "killed", "reaped", "workdir_retained")}
                          for e in supervisor.events],
        "manual_assessment": "pending: the operator grades each accepted answer in the protected result file",
    }


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------


def dry_run_report(profile: ModelProfile, plans, adapter: ApplicationVerifier,
                   settings: Optional[RunSettings] = None) -> dict:
    """Hashes, options and counts only: never a prompt, excerpt or answer."""
    settings = settings or run_settings(p.fixture_id for p in plans)
    snapshot = _snapshot(profile, "admin_override", None)
    return {
        "record_type": "dry_run",
        "harness_revision": settings.harness_revision,
        "scope_note": settings.scope_note,
        "baseline_note": settings.baseline_note,
        "profile": {
            "profile_id": profile.profile_id, "model_tag": profile.model_tag,
            "expected_digest": profile.expected_digest, "think": profile.think, "keep_alive": profile.keep_alive,
            "answer_options": profile.answer_options.as_ollama_options(),
            "answer_read_timeout_s": profile.answer_read_timeout_s,
            "verifier": {"model_tag": snapshot.verifier.model_tag,
                         "options": snapshot.verifier.options.as_ollama_options(),
                         "read_timeout_s": snapshot.verifier.read_timeout_s},
            **_qualification_flags(profile),
        },
        "smoke": {"prompt_sha256": _sha(base.SMOKE_PROMPT), "options": SMOKE_OPTIONS.as_ollama_options()},
        "fixtures": [{**_fixture_entry(p), "prompt_chars": len(p.prompt), "answer_payload_sha256":
                      _sha(base._canonical(snapshot.ollama_payload(p.prompt, stream=True)))} for p in plans],
        "verifier_adapter": {"label": adapter.label, "source_sha256": adapter.source_sha256,
                             "template_sha256": _sha(adapter.template),
                             "max_claims_verified": adapter.citation_limits.max_claims_verified,
                             "max_calls_per_answer": MAX_VERIFIER_CALLS_PER_ANSWER},
        "planned_model_calls": {"smoke": 1, "answers": len(plans),
                                "verifier_max": MAX_VERIFIER_CALLS_PER_ANSWER * len(plans),
                                "max_total": 1 + len(plans) * (1 + MAX_VERIFIER_CALLS_PER_ANSWER)},
        "observation_endpoints": sorted(OBSERVATION_PATHS),
        "model_endpoint": GENERATE_PATH,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse(argv):
    parser = argparse.ArgumentParser(description="Bounded single-profile answer + verifier evaluation "
                                                 "(default fixtures C04, C05)")
    parser.add_argument("--dry-run", action="store_true", help="print hashes, options and counts; no network")
    parser.add_argument("--live", action="store_true", help="explicit opt-in to contact the given Ollama root")
    parser.add_argument("--profile", help="exactly one registry profile id (live: required)")
    parser.add_argument("--fixtures", help="comma-separated frozen fixture ids in run order, e.g. "
                                           "C01,C02,C03,C04,C05,C06 (default C04,C05)")
    parser.add_argument("--ollama-url", help="the Ollama server root, e.g. http://127.0.0.1:11434 (live: required)")
    parser.add_argument("--client-budget-seconds", type=float,
                        help="hard client wait per worker call, finite, in (0, 3600] (live: required)")
    parser.add_argument("--out", help="new protected JSONL result file (live: required; never overwritten)")
    parser.add_argument("--summary-out", help="new shareable summary file (default: <out>.summary.json)")
    return parser.parse_args(argv)


def _new_output(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        raise UsageError(f"{path} already exists; results are never overwritten")
    if not path.parent.is_dir():
        raise UsageError(f"{path.parent} is not an existing directory")
    return path


def main(argv=None, *, stdout=sys.stdout, registry_factory: Callable[[], ModelRegistry] = default_model_registry,
         supervisor_factory: Callable[..., OwnedWorkerSupervisor] = OwnedWorkerSupervisor,
         adapter_factory: Callable[[Callable], ApplicationVerifier] = load_application_verifier,
         clock: Callable[[], float] = time.monotonic, now: Callable[[], str] = _utc_now) -> int:
    args = _parse(argv)
    if args.dry_run == args.live:
        raise UsageError("choose exactly one of --dry-run or --live")
    fixture_ids = require_fixtures(args.fixtures)
    # Every fixture is loaded and its pinned hashes checked before any output
    # file, supervisor or network contact.
    fixtures = [base.load_fixture(f) for f in fixture_ids]
    settings = run_settings(fixture_ids)
    registry = registry_factory()

    if args.dry_run:
        profile = require_profile(args.profile or INTENDED_FIRST_PROFILE, registry)
        adapter = adapter_factory(_refuse_dispatch)
        plans = [plan_fixture(fixture, adapter.citation_limits) for fixture in fixtures]
        report = dry_run_report(profile, plans, adapter, settings)
        stdout.write(json.dumps(report, sort_keys=True) + "\n")
        if args.out:
            with _new_output(args.out).open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(report, sort_keys=True) + "\n")
        return 0

    missing = [flag for flag, value in (("--profile", args.profile), ("--ollama-url", args.ollama_url),
                                         ("--client-budget-seconds", args.client_budget_seconds),
                                         ("--out", args.out)) if value is None]
    if missing:
        raise UsageError(f"a live run needs {', '.join(missing)}")
    profile = require_profile(args.profile, registry)
    url = require_root_url(args.ollama_url)
    budget = require_budget(args.client_budget_seconds)
    out_path = _new_output(args.out)
    summary_path = _new_output(args.summary_out or f"{args.out}.summary.json")
    if summary_path.resolve() == out_path.resolve():
        raise UsageError("--summary-out must differ from --out")
    # Everything that cannot touch the network is prepared before it is.
    check_adapter = adapter_factory(_refuse_dispatch)
    plans = [plan_fixture(fixture, check_adapter.citation_limits) for fixture in fixtures]
    supervisor = supervisor_factory(url, budget_s=budget, work_root=out_path.parent)

    with out_path.open("x", encoding="utf-8") as out, summary_path.open("x", encoding="utf-8") as summary_file:
        summary = run_batch(profile=profile, supervisor=supervisor, sink=_Sink(out, stdout), plans=plans,
                            adapter_factory=adapter_factory, clock=clock, now=now, settings=settings)
        summary_file.write(json.dumps(summary, ensure_ascii=True, sort_keys=True) + "\n")
    return 0 if summary["outcome"] == "completed" else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (UsageError, base.FixtureError, AdapterError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(2)
