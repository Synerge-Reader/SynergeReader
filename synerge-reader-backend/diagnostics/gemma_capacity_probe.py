#!/usr/bin/env python3
"""Bounded Gemma capacity probe (operator-run): the answer model and the
configured embedder resident together, and two overlapping answer jobs.

One pinned Gemma profile answers and verifies. In order:

1. Read-only identity, before any model call: the Ollama runtime version
   against the operator's expected version; Gemma's installed digest against
   the profile's pin (expected and observed recorded separately) and its
   thinking capability; the embedder's installed digest against the
   operator-supplied digest and its embedding length against the supplied
   embedding profile. Any mismatch stops the probe.
2. One synthetic embedding request through the application's own adapter
   (``OllamaEmbeddingProvider.embed_query``): the supplied embedding profile's
   model and query prefix, ``truncate: false`` and an explicit keep_alive. The
   adapter validates the reply strictly (one vector, the profile's dimension,
   real, finite, not all zero). It completes before any answer lane starts.
3. One smoke at the production answer context (the profile's ``num_ctx``,
   8192), 32 output tokens, accepted only like an answer.
4. Two lanes start together: lane A answers C04 and lane B answers C05 with
   the application's exact answer payloads, and each accepted answer is then
   checked by main.py's compiled verifier -- at most two sequential calls.

At most 8 model calls (1 embedding, 1 smoke, 2 answers, 4 verifier calls) and
at most 2 generation calls in flight. Every call runs in an owned worker
(diagnostics/gemma_qualification.py) under a finite per-call client budget,
further capped by the whole-probe budget. The first outcome without terminal
evidence on any lane -- UNKNOWN server execution -- stops new dispatch on both
lanes at once through one shared stop latch; a worker that had already started
is observed and reaped under its own bound. An operator interrupt while the
lanes run fences both lanes the same way and still waits for started calls.
Client termination does not prove that server inference stopped. Never:
retry, fallback, unload, warm-up loop or any service operation.

Recorded: expected and observed identities; every call's interval on one
probe clock, its requested settings and the server's reported metrics; each
job's answer-plus-verifier duration and whether it exceeds the application's
whole-job budget; ``/api/ps`` residency and effective context before, between,
during and after the workload; and every missing observation or unresolved
execution. This measures joint residency and two short-fixture generation
jobs: not full-length answers, not concurrent upload throughput and not the
application's admission ledger. It sets no thresholds and extrapolates
nothing.

Usage (operator, inside an agreed window with no competing Ollama clients):

    python diagnostics/gemma_capacity_probe.py --dry-run --embedding-model mxbai-embed-large:335m
    python diagnostics/gemma_capacity_probe.py --live --profile gemma_candidate \\
        --ollama-url http://127.0.0.1:11434 --expected-ollama-version 0.30.8 \\
        --embedding-model mxbai-embed-large:335m --embedding-digest <64 hex> \\
        --embedding-keep-alive 30m --client-budget-seconds 600 \\
        --probe-budget-seconds 1800 --out C:/protected/gemma_capacity.jsonl

``--out`` receives protected raw answers; ``<out>.summary.json`` (or
``--summary-out``) receives a shareable summary without any text. Neither file
is ever overwritten.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from diagnostics import gemma_qualification as gq  # noqa: E402
from diagnostics import generation_diagnostic as base  # noqa: E402
from generation_admission import AdmissionSettings, CallState  # noqa: E402
from generation_profiles import (  # noqa: E402
    GEMMA_CANDIDATE,
    SMOKE_OPTIONS,
    GenerationOptions,
    ModelProfile,
    ModelRegistry,
    _snapshot,
    default_model_registry,
    normalize_digest,
)
from ollama_embedding_provider import EmbeddingProviderError, OllamaEmbeddingProvider  # noqa: E402
from rag_model_profiles import EmbeddingProfile, ProfileError, known_embedding_profile  # noqa: E402

PROBE_REVISION = "gemma-capacity-probe-v1"
LANES = (("A", "C04"), ("B", "C05"))
MAX_MODEL_CALLS = 8
MAX_CONCURRENT_GENERATIONS = 2
# The application's provisional whole-job budget for one admitted answer.
APP_JOB_BUDGET_S = AdmissionSettings().answer_job_budget_s
SYNTHETIC_EMBEDDING_TEXT = "Synthetic capacity probe sentence; it contains no document or user text."
RESIDENCY_INTERVAL_S = 10.0
MAX_RESIDENCY_SAMPLES = 30
_KEEP_ALIVE_PATTERN = re.compile(r"^[1-9][0-9]{0,3}[smh]$")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
_METRIC_KEYS = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count",
                "eval_duration")

SCOPE_NOTE = (
    "Joint residency of the pinned Gemma profile and the supplied embedder, and two overlapping short-fixture "
    "answer jobs (C04, C05) with at most two verifier calls each. Not full-length answers, not concurrent upload "
    "throughput, not the application's admission ledger. No thresholds and no extrapolation; the application "
    "job budget comparison is for the measured jobs only."
)

UsageError = gq.UsageError


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class SharedStopLatch(gq.StopLatch):
    """One stop decision for every lane: a stop recorded by any supervisor
    sharing this latch fences every later worker start, at once."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc) -> bool:
        self._lock.release()
        return False


class ProbeDispatch:
    """Every worker call of the probe passes here: the whole-probe deadline
    caps each call's client wait, and model calls are counted against
    MAX_MODEL_CALLS and MAX_CONCURRENT_GENERATIONS under one lock."""

    def __init__(self, latch: SharedStopLatch, *, clock: Callable[[], float], origin: float,
                 deadline: float) -> None:
        self.latch = latch
        self.clock = clock
        self.origin = origin
        self.deadline = deadline
        self._lock = threading.Lock()
        self.model_calls = 0
        self.generations_in_flight = 0
        self.peak_generations_in_flight = 0

    def now(self) -> float:
        return round(self.clock() - self.origin, 3)

    def call(self, supervisor: gq.OwnedWorkerSupervisor, interval: dict, *, model_call: bool, path: str,
             **kwargs) -> gq.WorkerOutcome:
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            if model_call:
                self.latch.stop("probe_deadline")
            raise gq.BatchStopped("probe_deadline")
        generation = path == gq.GENERATE_PATH
        if model_call:
            with self._lock:
                if self.latch.stopped is not None:
                    raise gq.BatchStopped(self.latch.stopped)
                if self.model_calls >= MAX_MODEL_CALLS:
                    self.latch.stop("probe_call_limit")
                    raise gq.BatchStopped("probe_call_limit")
                if generation and self.generations_in_flight >= MAX_CONCURRENT_GENERATIONS:
                    self.latch.stop("probe_concurrency_limit")
                    raise gq.BatchStopped("probe_concurrency_limit")
                self.model_calls += 1
                if generation:
                    self.generations_in_flight += 1
                    self.peak_generations_in_flight = max(self.peak_generations_in_flight,
                                                          self.generations_in_flight)
        interval["started_s"] = self.now()
        try:
            return supervisor.call(model_call=model_call, path=path, budget_s=remaining, **kwargs)
        finally:
            interval["ended_s"] = self.now()
            if model_call and generation:
                with self._lock:
                    self.generations_in_flight -= 1


class LaneSupervisor:
    """The supervisor interface gq.BatchTransport and gq.SupervisedHttp use,
    routed through the probe dispatcher. ``interval`` and ``outcome`` describe
    the last call."""

    def __init__(self, inner: gq.OwnedWorkerSupervisor, dispatch: ProbeDispatch) -> None:
        self.inner = inner
        self.dispatch = dispatch
        self.base_url = inner.base_url
        self.budget_s = inner.budget_s
        self.interval: dict = {}
        self.outcome: Optional[gq.WorkerOutcome] = None

    @property
    def stopped(self) -> Optional[str]:
        return self.inner.stopped

    @property
    def events(self) -> list:
        return self.inner.events

    def stop(self, category: str) -> None:
        self.inner.stop(category)

    def call(self, **kwargs) -> gq.WorkerOutcome:
        self.interval, self.outcome = {}, None
        self.outcome = self.dispatch.call(self.inner, self.interval, **kwargs)
        return self.outcome


class LockedSink:
    def __init__(self, handle, stdout) -> None:
        self._sink = gq._Sink(handle, stdout)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        with self._lock:
            self._sink.write(record)

    def say(self, line: str) -> None:
        with self._lock:
            self._sink.say(line)


def _metrics_from_body(body: Optional[bytes]) -> Optional[dict]:
    try:
        parsed = json.loads(body.decode("utf-8")) if body else None
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {key: parsed[key] for key in _METRIC_KEYS if isinstance(parsed.get(key), (int, float))}


def _lane_emitter(sink: LockedSink, lane: str, proxy: LaneSupervisor) -> Callable[[dict], None]:
    def emit(record: dict) -> None:
        record["lane"] = lane
        record.update(proxy.interval)
        if "server_metrics" not in record and proxy.outcome is not None and proxy.outcome.category == "end_of_body":
            metrics = _metrics_from_body(proxy.outcome.body)
            if metrics is not None:
                record["server_metrics"] = metrics
        sink.write({"record_type": "call", **record})
        sink.say(f"[{lane}] {gq._status(record)}")
    return emit


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def require_embedding_profile(model: object) -> EmbeddingProfile:
    if not isinstance(model, str) or not model:
        raise UsageError("--embedding-model must name exactly one known embedding model")
    try:
        return known_embedding_profile("ollama", model)
    except ProfileError:
        raise UsageError("--embedding-model is not a known Ollama embedding model") from None


def require_digest(value: object) -> str:
    digest = normalize_digest(value)
    if digest is None:
        raise UsageError("--embedding-digest must be a full sha256 manifest digest")
    return digest


def require_keep_alive(value: object) -> str:
    if not isinstance(value, str) or not _KEEP_ALIVE_PATTERN.match(value):
        raise UsageError("--embedding-keep-alive must be an explicit duration such as 30m")
    return value


def require_version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION_PATTERN.match(value):
        raise UsageError("--expected-ollama-version must be a version such as 0.30.8")
    return value


def require_gemma_profile(profile_id: object, registry: ModelRegistry) -> ModelProfile:
    profile = gq.require_profile(profile_id, registry)
    if profile.profile_id != GEMMA_CANDIDATE:
        raise UsageError(f"the capacity probe runs only the pinned {GEMMA_CANDIDATE} profile")
    return profile


def smoke_options(profile: ModelProfile) -> GenerationOptions:
    """The smoke at the production answer context: only the output limit is
    the smoke's, so the runner is not reloaded for the answers."""
    return GenerationOptions(temperature=SMOKE_OPTIONS.temperature, num_ctx=profile.answer_options.num_ctx,
                             num_predict=SMOKE_OPTIONS.num_predict)


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def embedder_identity(http, embedding: EmbeddingProfile, expected_digest: str) -> dict:
    """Expected identity (operator-supplied digest, profile dimension) beside
    what the server reports for the embedding model's exact tag."""
    identity = {
        "model": embedding.model, "expected_digest": expected_digest, "observed_digest": None,
        "expected_dimension": embedding.dimension, "observed_embedding_length": None,
        "observed_context_length": None, "check": "unreadable",
    }
    listed = [m for m in http.get_json("/api/tags").get("models", [])
              if isinstance(m, dict) and embedding.model in (m.get("name"), m.get("model"))]
    if not listed:
        identity["check"] = "absent"
    else:
        observed = normalize_digest(listed[0].get("digest"))
        identity["observed_digest"] = observed
        identity["check"] = "unreadable" if observed is None else (
            "match" if observed == expected_digest else "mismatch")
    info = http.post_json("/api/show", {"model": embedding.model}).get("model_info") or {}
    lengths = [v for k, v in info.items() if k.endswith("embedding_length")]
    contexts = [v for k, v in info.items() if k.endswith("context_length")]
    identity["observed_embedding_length"] = lengths[0] if len(lengths) == 1 else None
    identity["observed_context_length"] = contexts[0] if len(contexts) == 1 else None
    if identity["check"] == "match" and identity["observed_embedding_length"] != embedding.dimension:
        identity["check"] = ("dimension_unreadable" if identity["observed_embedding_length"] is None
                             else "dimension_mismatch")
    return identity


# --------------------------------------------------------------------------
# The probe
# --------------------------------------------------------------------------


class Probe:
    def __init__(self, *, profile: ModelProfile, embedding: EmbeddingProfile, embedding_digest: str,
                 embedding_keep_alive: str, expected_version: str, url: str, call_budget_s: float,
                 probe_budget_s: float, work_root: Path, sink: LockedSink,
                 supervisor_factory: Callable[..., gq.OwnedWorkerSupervisor],
                 adapter_factory: Callable[[Callable], gq.ApplicationVerifier],
                 clock: Callable[[], float], now: Callable[[], str],
                 sample_interval_s: float = RESIDENCY_INTERVAL_S) -> None:
        self.profile = profile
        self.embedding = embedding
        self.embedding_digest = embedding_digest
        self.embedding_keep_alive = embedding_keep_alive
        self.expected_version = expected_version
        self.url = url
        self.call_budget_s = call_budget_s
        self.probe_budget_s = probe_budget_s
        self.work_root = work_root
        self.sink = sink
        self.supervisor_factory = supervisor_factory
        self.adapter_factory = adapter_factory
        self.clock = clock
        self.now = now
        self.sample_interval_s = sample_interval_s
        self.latch = SharedStopLatch()
        origin = clock()
        self.dispatch = ProbeDispatch(self.latch, clock=clock, origin=origin, deadline=origin + probe_budget_s)
        self.supervisors: list = []
        self.calls: list = []
        self.residency: dict = {}
        self.samples: list = []
        self.missing: list = []
        self._calls_lock = threading.Lock()

    # --- plumbing ----------------------------------------------------------------

    def _proxy(self, *, model_work: bool, allow_embedding: bool = False) -> LaneSupervisor:
        inner = self.supervisor_factory(self.url, budget_s=self.call_budget_s, work_root=self.work_root,
                                        latch=self.latch if model_work else gq.StopLatch(),
                                        allow_embedding=allow_embedding)
        self.supervisors.append(inner)
        return LaneSupervisor(inner, self.dispatch)

    def _record(self, record: dict) -> None:
        with self._calls_lock:
            self.calls.append(record)

    def _snapshot(self, label: str) -> Optional[dict]:
        """A read-only ``/api/ps`` view through a fresh observer, so a failed
        observation or a stopped model latch never blocks later views."""
        http = gq.SupervisedHttp(self._proxy(model_work=False))
        try:
            snap = base.residency(http)
        except gq.BatchStopped as stop:
            self.missing.append(f"residency {label}: {stop.category}")
            return None
        if "error" in snap:
            self.missing.append(f"residency {label}: observation failed ({snap['error'].get('kind')})")
            return None
        return {"at_s": self.dispatch.now(), **snap}

    # --- phases ------------------------------------------------------------------

    def identities(self, http) -> tuple:
        identity = base.model_identity(http, self.profile)
        decision = base.thinking_decision("profile", self.profile, identity)
        runtime = {"expected": self.expected_version, "observed": identity.get("runtime_version")}
        embedder = embedder_identity(http, self.embedding, self.embedding_digest)
        record = {"record_type": "identity", "runtime": runtime, "generation": identity,
                  "thinking_policy": decision, "embedding": embedder}
        self.sink.write(record)
        if runtime["observed"] is None:
            self.missing.append("runtime version")
        if runtime["observed"] != runtime["expected"]:
            self.latch.stop("runtime_version_mismatch")
        elif identity["check"] != "match":
            self.latch.stop(f"identity_{identity['check']}")
        elif decision["refused"]:
            self.latch.stop("thinking_capability_missing")
        elif decision["send"] != self.profile.think:
            self.latch.stop("think_parity")
        elif embedder["check"] != "match":
            self.latch.stop(f"embedding_identity_{embedder['check']}")
        if self.latch.stopped is not None:
            raise gq.BatchStopped(self.latch.stopped)
        return identity, decision, runtime, embedder

    def embed(self, proxy: LaneSupervisor) -> dict:
        result = {"accepted": False, "vectors": None, "dimension": None, "failure": None}

        def post(endpoint: str, payload: dict):
            if endpoint != gq.EMBED_PATH:
                raise gq.BatchStopped("embedding_call_shape")
            outcome = proxy.call(model_call=True, method="POST", path=gq.EMBED_PATH, body=payload,
                                 read_timeout_s=self.call_budget_s, max_body_bytes=gq.JSON_BODY_LIMIT_BYTES,
                                 watch_stream=False)
            record = gq._call_record(len(self.calls) + 1, "embedding", None, payload, outcome)
            record.update(proxy.interval, lane="setup", requested={
                "model": payload.get("model"), "inputs": len(payload.get("input") or []),
                "truncate": payload.get("truncate"), "keep_alive": payload.get("keep_alive")})
            if outcome.category == "end_of_body":
                # A complete HTTP response was read: the server finished this call.
                record["state"] = CallState.OBSERVED_TERMINAL.value
                record["server_metrics"] = _metrics_from_body(outcome.body)
            self._record(record)
            self.sink.write({"record_type": "call", **record})
            self.sink.say(f"[setup] embedding: category={record['category']} state={record['state']} "
                          f"http_status={record['http_status']} wall_s={record['wall_s']}")
            if outcome.category != "end_of_body":
                raise gq.BatchStopped(proxy.stopped or outcome.category)
            return gq._WorkerHttpResponse(outcome.http_status, outcome.body)

        provider = OllamaEmbeddingProvider(profile=self.embedding, http_post=post,
                                           keep_alive=self.embedding_keep_alive)
        try:
            vector = provider.embed_query(SYNTHETIC_EMBEDDING_TEXT)
        except EmbeddingProviderError as exc:
            result["failure"] = type(exc).__name__
            self.sink.write({"record_type": "embedding_validation", **result})
            self.latch.stop("embedding_rejected")
            raise gq.BatchStopped("embedding_rejected") from None
        result.update(accepted=True, vectors=1, dimension=len(vector))
        self.sink.write({"record_type": "embedding_validation", **result})
        return result

    def lane(self, name: str, plan: gq.FixturePlan, snapshot, transport: gq.BatchTransport,
             adapter: gq.ApplicationVerifier, start: threading.Event) -> dict:
        result = {"lane": name, "fixture": plan.fixture_id, "stopped": None}
        start.wait()
        try:
            answer = transport.stream_generate(
                "answer", plan.fixture_id, snapshot.ollama_payload(plan.prompt, stream=True),
                snapshot.answer_read_timeout_s, prompt_sha256=gq._sha(plan.prompt),
            )
            result["answer_accepted"] = answer.get("accepted")
            if answer["state"] != CallState.OBSERVED_TERMINAL.value:
                raise gq.BatchStopped(transport.supervisor.stopped or "no_terminal_evidence")
            if answer["accepted"]:
                verification = gq.run_verification(answer["text"], plan, snapshot, adapter, transport)
            else:
                verification = {"fixture": plan.fixture_id, "skipped": "answer_not_accepted"}
            self.sink.write({"record_type": "verification", "lane": name, **verification})
        except gq.BatchStopped as stop:
            result["stopped"] = stop.category
        except Exception as exc:
            # Not expected: fence both lanes rather than let one fail silently.
            self.latch.stop("lane_error")
            result["stopped"] = f"lane_error:{type(exc).__name__}"
        finally:
            for record in transport.calls:
                self._record(record)
        return result

    def run(self, plans: dict) -> dict:
        started = time.monotonic()
        outcome, identity, decision, runtime, embedder, embedding, smoke = "completed", None, None, None, None, None, None
        lanes: dict = {}
        setup = self._proxy(model_work=True, allow_embedding=True)
        http = gq.SupervisedHttp(setup)
        try:
            identity, decision, runtime, embedder = self.identities(http)
            self.residency["before_embedding"] = self._snapshot("before_embedding")
            embedding = self.embed(setup)
            self.residency["after_embedding"] = self._snapshot("after_embedding")
            setup_transport = gq.BatchTransport(setup, _lane_emitter(self.sink, "setup", setup))
            smoke = setup_transport.stream_generate(
                "smoke", None,
                base.request_body(self.profile, base.SMOKE_PROMPT, think=decision["send"],
                                  options=smoke_options(self.profile)),
                self.profile.answer_read_timeout_s, prompt_sha256=gq._sha(base.SMOKE_PROMPT),
            )
            for record in setup_transport.calls:
                self._record(record)
            if smoke["state"] != CallState.OBSERVED_TERMINAL.value:
                raise gq.BatchStopped(setup.stopped or "no_terminal_evidence")
            if not smoke["accepted"]:
                self.latch.stop("smoke_rejected")
                raise gq.BatchStopped("smoke_rejected")
            self.residency["after_smoke"] = self._snapshot("after_smoke")
            lanes = self.run_lanes(plans, identity)
        except gq.BatchStopped:
            pass
        except gq.ObservationFailed:
            self.latch.stop("observation_failed")
        self.residency["after_workload"] = self._snapshot("after_workload")
        if self.latch.stopped is not None:
            outcome = "stopped"
        return {"outcome": outcome, "stop_category": self.latch.stopped, "identity": identity,
                "decision": decision, "runtime": runtime, "embedder": embedder, "embedding": embedding,
                "lanes": lanes, "total_wall_s": round(time.monotonic() - started, 3)}

    def run_lanes(self, plans: dict, identity: dict) -> dict:
        snapshot = _snapshot(self.profile, "admin_override", identity["observed_digest"])
        start, done = threading.Event(), threading.Event()
        prepared, results, threads = [], {}, []
        for name, fixture_id in LANES:
            proxy = self._proxy(model_work=True)
            transport = gq.BatchTransport(proxy, _lane_emitter(self.sink, name, proxy))
            prepared.append((name, plans[fixture_id], transport, self.adapter_factory(transport.post_ollama)))

        def run_one(name, plan, transport, adapter):
            results[name] = self.lane(name, plan, snapshot, transport, adapter, start)

        def sample() -> None:
            wait = min(2.0, self.sample_interval_s)
            while not done.wait(wait) and len(self.samples) < MAX_RESIDENCY_SAMPLES:
                snap = self._snapshot("during_workload")
                if snap is not None:
                    self.samples.append(snap)
                    self.sink.write({"record_type": "residency_sample", **snap})
                wait = self.sample_interval_s

        for name, plan, transport, adapter in prepared:
            threads.append(threading.Thread(target=run_one, args=(name, plan, transport, adapter),
                                            name=f"probe-lane-{name}", daemon=True))
        sampler = threading.Thread(target=sample, name="probe-residency", daemon=True)
        for thread in threads:
            thread.start()
        sampler.start()
        start.set()
        join_s = self.probe_budget_s + 2 * gq.REAP_SECONDS + 5
        for thread in threads:
            while True:
                try:
                    thread.join(join_s)
                    break
                except KeyboardInterrupt:
                    # An operator interrupt reaches only this thread. Fence
                    # both lanes, then keep waiting: a worker already started
                    # is reaped by its own lane under its own bound, never
                    # orphaned.
                    if self.latch.stopped is None:
                        self.missing.append("operator interrupt during the lanes; started calls were left to "
                                            "finish under their own bounds")
                    self.latch.stop("operator_interrupt")
            if thread.is_alive():
                self.missing.append(f"{thread.name} did not finish within the probe bound")
        done.set()
        sampler.join(gq.OBSERVATION_READ_TIMEOUT_S + gq.REAP_SECONDS + 5)
        if sampler.is_alive():
            self.missing.append("residency sampler did not finish within its bound")
        for name, _ in LANES:
            self.sink.write({"record_type": "lane", **results.get(name, {"lane": name, "stopped": "lane_unjoined"})})
        return results


# --------------------------------------------------------------------------
# Summary (shareable: no text)
# --------------------------------------------------------------------------


def _seconds(value) -> Optional[float]:
    return None if not isinstance(value, (int, float)) else round(value / 1e9, 3)


def _rate(count, duration_ns) -> Optional[float]:
    if not isinstance(count, (int, float)) or not isinstance(duration_ns, (int, float)) or duration_ns <= 0:
        return None
    return round(count / (duration_ns / 1e9), 2)


def _observed(record: dict) -> Optional[dict]:
    metrics = record.get("server_metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "prompt_eval_count": metrics.get("prompt_eval_count"), "eval_count": metrics.get("eval_count"),
        "load_s": _seconds(metrics.get("load_duration")), "prompt_eval_s": _seconds(metrics.get("prompt_eval_duration")),
        "eval_s": _seconds(metrics.get("eval_duration")), "total_s": _seconds(metrics.get("total_duration")),
        "observed_prompt_tokens_per_s": _rate(metrics.get("prompt_eval_count"), metrics.get("prompt_eval_duration")),
        "observed_output_tokens_per_s": _rate(metrics.get("eval_count"), metrics.get("eval_duration")),
    }


def _overlap(first: Optional[tuple], second: Optional[tuple]) -> Optional[float]:
    if first is None or second is None:
        return None
    return round(max(0.0, min(first[1], second[1]) - max(first[0], second[0])), 3)


def _span(records) -> Optional[tuple]:
    spans = [(r["started_s"], r["ended_s"]) for r in records if "started_s" in r and "ended_s" in r]
    return (min(s for s, _ in spans), max(e for _, e in spans)) if spans else None


def _residency_view(snap: Optional[dict]) -> Optional[dict]:
    if snap is None:
        return None
    models = []
    for model in snap.get("models", []):
        size, vram = model.get("size"), model.get("size_vram")
        models.append({**model, "vram_pct": round(100 * vram / size, 1)
                       if isinstance(size, (int, float)) and size and isinstance(vram, (int, float)) else None})
    return {"at_s": snap.get("at_s"), "models": models}


def _max_overlapping(records) -> int:
    edges = sorted([(r["started_s"], 1) for r in records] + [(r["ended_s"], -1) for r in records],
                   key=lambda edge: (edge[0], edge[1]))
    current = peak = 0
    for _, step in edges:
        current += step
        peak = max(peak, current)
    return peak


def build_summary(*, run_id: str, probe: Probe, result: dict, plans: dict, adapter: gq.ApplicationVerifier) -> dict:
    calls = sorted(probe.calls, key=lambda r: (r.get("started_s") is None, r.get("started_s") or 0))
    safe_keys = ("lane", "role", "fixture", "model", "payload_sha256", "category", "state", "http_status",
                 "error_type", "body_bytes", "body_sha256", "timing", "wall_s", "worker_reaped", "done_reason",
                 "failure", "accepted", "response_chars", "response_sha256", "thinking_chars", "started_s",
                 "ended_s", "options", "think", "requested")
    safe_calls = [{**{k: c.get(k) for k in safe_keys if k in c}, "observed": _observed(c)} for c in calls]
    generation = [c for c in calls if c.get("role") in ("smoke", "answer", "verifier")
                  and "started_s" in c and "ended_s" in c]
    jobs = {}
    for name, fixture_id in LANES:
        lane_calls = [c for c in calls if c.get("lane") == name]
        answer = next((c for c in lane_calls if c.get("role") == "answer"), None)
        span = _span(lane_calls)
        duration = None if span is None else round(span[1] - span[0], 3)
        lane_result = result["lanes"].get(name, {})
        jobs[name] = {
            "fixture": fixture_id,
            "answer_interval_s": None if answer is None or "ended_s" not in answer
            else [answer["started_s"], answer["ended_s"]],
            "verifier_calls": sum(1 for c in lane_calls if c.get("role") == "verifier"),
            "answer_plus_verifier_s": duration,
            "exceeds_app_job_budget": None if duration is None else duration > APP_JOB_BUDGET_S,
            "stopped": lane_result.get("stopped") if lane_result else "lane_not_started",
        }
    answers = {c.get("lane"): c for c in calls if c.get("role") == "answer"}

    def interval(lane):
        record = answers.get(lane)
        return None if record is None or "ended_s" not in record else (record["started_s"], record["ended_s"])

    unresolved = [{"lane": c.get("lane"), "role": c.get("role"), "fixture": c.get("fixture"),
                   "category": c.get("category")} for c in calls if c.get("state") == CallState.UNKNOWN.value]
    missing = list(probe.missing)
    for c in calls:
        if c.get("category") == "end_of_body" and _observed(c) is None:
            missing.append(f"server metrics for {c.get('lane')} {c.get('role')} {c.get('fixture') or ''}".strip())
    if not probe.samples and result["lanes"]:
        missing.append("no residency sample during the workload")
    snapshot = _snapshot(probe.profile, "admin_override", None)
    events = [e for s in probe.supervisors for e in s.events]
    return {
        "record_type": "summary",
        "probe_revision": PROBE_REVISION,
        "run_id": run_id,
        "scope_note": SCOPE_NOTE,
        "outcome": result["outcome"],
        "stop_category": result["stop_category"],
        "unknown_server_execution": bool(unresolved),
        "unresolved_execution": unresolved,
        "unknown_note": gq.UNKNOWN_NOTE if unresolved else None,
        "identity": {
            "runtime": result["runtime"],
            "generation": None if result["identity"] is None else {
                "model_tag": probe.profile.model_tag,
                "expected_digest": result["identity"].get("expected_digest"),
                "observed_digest": result["identity"].get("observed_digest"),
                "check": result["identity"].get("check"),
                "reports_thinking": None if result["decision"] is None else result["decision"].get(
                    "server_reports_thinking"),
                "think_sent": None if result["decision"] is None else result["decision"].get("send"),
            },
            "embedding": result["embedder"],
        },
        "requested": {
            "profile_id": probe.profile.profile_id, "model_tag": probe.profile.model_tag,
            "keep_alive": probe.profile.keep_alive, "think": probe.profile.think,
            "smoke_options": smoke_options(probe.profile).as_ollama_options(),
            "answer_options": probe.profile.answer_options.as_ollama_options(),
            "verifier_options": snapshot.verifier.options.as_ollama_options(),
            "embedding": {"model": probe.embedding.model, "dimension": probe.embedding.dimension,
                          "query_prefix_applied": bool(probe.embedding.query_prefix), "truncate": False,
                          "keep_alive": probe.embedding_keep_alive},
            "client_budget_seconds": probe.call_budget_s, "probe_budget_seconds": probe.probe_budget_s,
        },
        "fixtures": [gq._fixture_entry(plans[fixture_id]) for _, fixture_id in LANES],
        "embedding_validation": result["embedding"],
        "calls": safe_calls,
        "model_calls_dispatched": sum(1 for e in events if e.get("model_call")),
        "max_model_calls": MAX_MODEL_CALLS,
        "peak_generation_calls_in_flight": probe.dispatch.peak_generations_in_flight,
        "max_overlapping_generation_intervals": _max_overlapping(generation),
        "max_concurrent_generation_calls": MAX_CONCURRENT_GENERATIONS,
        "jobs": jobs,
        "answer_overlap_s": _overlap(interval("A"), interval("B")),
        "application_job_budget_s": APP_JOB_BUDGET_S,
        "job_budget_note": "Measured short-fixture jobs only; not a full-length answer and not a projection.",
        "residency": {label: _residency_view(snap) for label, snap in probe.residency.items()},
        "residency_during_workload": [_residency_view(s) for s in probe.samples],
        "residency_limits": gq.RESIDENCY_LIMITS,
        "missing_observations": missing,
        "verifier_adapter": {"label": adapter.label, "source_sha256": adapter.source_sha256,
                             "template_sha256": gq._sha(adapter.template),
                             "max_calls_per_answer": gq.MAX_VERIFIER_CALLS_PER_ANSWER},
        "worker_events": [{k: e.get(k) for k in ("path", "model_call", "deadline", "killed", "reaped",
                                                    "workdir_retained")} for e in events],
        "total_wall_s": result["total_wall_s"],
        "qualification": gq._qualification_flags(probe.profile),
    }


# --------------------------------------------------------------------------
# Dry run and CLI
# --------------------------------------------------------------------------


def dry_run_report(profile: ModelProfile, embedding: EmbeddingProfile, plans: dict,
                   adapter: gq.ApplicationVerifier) -> dict:
    """Hashes, settings and counts only: never a prompt, excerpt or answer."""
    snapshot = _snapshot(profile, "admin_override", None)
    return {
        "record_type": "dry_run",
        "probe_revision": PROBE_REVISION,
        "scope_note": SCOPE_NOTE,
        "profile": {"profile_id": profile.profile_id, "model_tag": profile.model_tag,
                    "expected_digest": profile.expected_digest, "think": profile.think,
                    "keep_alive": profile.keep_alive, **gq._qualification_flags(profile)},
        "smoke_options": smoke_options(profile).as_ollama_options(),
        "answer_options": profile.answer_options.as_ollama_options(),
        "verifier_options": snapshot.verifier.options.as_ollama_options(),
        "embedding": {"model": embedding.model, "dimension": embedding.dimension,
                      "query_prefix_applied": bool(embedding.query_prefix), "truncate": False,
                      "keep_alive": "required on --live"},
        "lanes": [{"lane": name, **gq._fixture_entry(plans[fixture_id]),
                   "answer_payload_sha256": gq._sha(base._canonical(snapshot.ollama_payload(
                       plans[fixture_id].prompt, stream=True)))} for name, fixture_id in LANES],
        "planned_model_calls": {"embedding": 1, "smoke": 1, "answers": len(LANES),
                                "verifier_max": gq.MAX_VERIFIER_CALLS_PER_ANSWER * len(LANES),
                                "max_total": MAX_MODEL_CALLS},
        "max_concurrent_generation_calls": MAX_CONCURRENT_GENERATIONS,
        "application_job_budget_s": APP_JOB_BUDGET_S,
        "verifier_adapter": {"label": adapter.label, "source_sha256": adapter.source_sha256},
        "endpoints": sorted(gq.OBSERVATION_PATHS | {gq.GENERATE_PATH, gq.EMBED_PATH}),
    }


def _parse(argv):
    parser = argparse.ArgumentParser(description="Bounded Gemma + embedder capacity probe (two answer lanes)")
    parser.add_argument("--dry-run", action="store_true", help="print hashes, settings and counts; no network")
    parser.add_argument("--live", action="store_true", help="explicit opt-in to contact the given Ollama root")
    parser.add_argument("--profile", help=f"must be {GEMMA_CANDIDATE} (live: required)")
    parser.add_argument("--ollama-url", help="the Ollama server root (live: required)")
    parser.add_argument("--expected-ollama-version", help="e.g. 0.30.8 (live: required)")
    parser.add_argument("--embedding-model", help="the configured embedding model, e.g. mxbai-embed-large:335m")
    parser.add_argument("--embedding-digest", help="its expected full installed digest (live: required)")
    parser.add_argument("--embedding-keep-alive", help="explicit keep_alive for the one embedding (live: required)")
    parser.add_argument("--client-budget-seconds", type=float, help="hard client wait per call, in (0, 3600]")
    parser.add_argument("--probe-budget-seconds", type=float, help="hard client deadline for the whole probe, "
                                                                     "in (0, 3600]")
    parser.add_argument("--out", help="new protected JSONL result file (live: required; never overwritten)")
    parser.add_argument("--summary-out", help="new shareable summary file (default: <out>.summary.json)")
    return parser.parse_args(argv)


def _plans(adapter: gq.ApplicationVerifier) -> dict:
    return {fixture_id: gq.plan_fixture(base.load_fixture(fixture_id), adapter.citation_limits)
            for _, fixture_id in LANES}


def main(argv=None, *, stdout=sys.stdout, registry_factory: Callable[[], ModelRegistry] = default_model_registry,
         supervisor_factory: Callable[..., gq.OwnedWorkerSupervisor] = gq.OwnedWorkerSupervisor,
         adapter_factory: Callable[[Callable], gq.ApplicationVerifier] = gq.load_application_verifier,
         clock: Callable[[], float] = time.monotonic, now: Callable[[], str] = _utc_now,
         sample_interval_s: float = RESIDENCY_INTERVAL_S) -> int:
    args = _parse(argv)
    if args.dry_run == args.live:
        raise UsageError("choose exactly one of --dry-run or --live")
    registry = registry_factory()
    embedding = require_embedding_profile(args.embedding_model)

    if args.dry_run:
        profile = require_gemma_profile(args.profile or GEMMA_CANDIDATE, registry)
        adapter = adapter_factory(gq._refuse_dispatch)
        stdout.write(json.dumps(dry_run_report(profile, embedding, _plans(adapter), adapter), sort_keys=True) + "\n")
        return 0

    missing = [flag for flag, value in (
        ("--profile", args.profile), ("--ollama-url", args.ollama_url),
        ("--expected-ollama-version", args.expected_ollama_version), ("--embedding-digest", args.embedding_digest),
        ("--embedding-keep-alive", args.embedding_keep_alive),
        ("--client-budget-seconds", args.client_budget_seconds),
        ("--probe-budget-seconds", args.probe_budget_seconds), ("--out", args.out)) if value is None]
    if missing:
        raise UsageError(f"a live probe needs {', '.join(missing)}")
    profile = require_gemma_profile(args.profile, registry)
    url = gq.require_root_url(args.ollama_url)
    version = require_version(args.expected_ollama_version)
    digest = require_digest(args.embedding_digest)
    keep_alive = require_keep_alive(args.embedding_keep_alive)
    call_budget = gq.require_budget(args.client_budget_seconds)
    probe_budget = gq.require_budget(args.probe_budget_seconds)
    out_path = gq._new_output(args.out)
    summary_path = gq._new_output(args.summary_out or f"{args.out}.summary.json")
    if summary_path.resolve() == out_path.resolve():
        raise UsageError("--summary-out must differ from --out")
    # Everything that cannot touch the network is prepared before it is.
    check_adapter = adapter_factory(gq._refuse_dispatch)
    plans = _plans(check_adapter)

    with out_path.open("x", encoding="utf-8") as out, summary_path.open("x", encoding="utf-8") as summary_file:
        sink = LockedSink(out, stdout)
        probe = Probe(profile=profile, embedding=embedding, embedding_digest=digest,
                      embedding_keep_alive=keep_alive, expected_version=version, url=url,
                      call_budget_s=call_budget, probe_budget_s=probe_budget, work_root=out_path.parent,
                      sink=sink, supervisor_factory=supervisor_factory, adapter_factory=adapter_factory,
                      clock=clock, now=now, sample_interval_s=sample_interval_s)
        run_id = f"gcp-{uuid.uuid4().hex}"
        sink.write({"record_type": "run", "run_id": run_id, "probe_revision": PROBE_REVISION, "started_at": now(),
                    "ollama_url": url, "profile_id": profile.profile_id, "model_tag": profile.model_tag,
                    "embedding_model": embedding.model, "client_budget_seconds": call_budget,
                    "probe_budget_seconds": probe_budget, "lanes": [list(lane) for lane in LANES],
                    "verifier_source_sha256": check_adapter.source_sha256, "main_sha256": check_adapter.main_sha256,
                    "scope_note": SCOPE_NOTE, "persists_answers_in_application": False,
                    "qualification": gq._qualification_flags(profile)})
        result = probe.run(plans)
        summary = build_summary(run_id=run_id, probe=probe, result=result, plans=plans, adapter=check_adapter)
        sink.write({"record_type": "end", "finished_at": now(), "outcome": result["outcome"],
                    "stop_category": result["stop_category"],
                    "unknown_server_execution": summary["unknown_server_execution"],
                    "note": gq.UNKNOWN_NOTE if summary["unknown_server_execution"] else None})
        sink.say(f"probe {result['outcome']} stop={result['stop_category']} "
                 f"unknown_server_execution={summary['unknown_server_execution']} "
                 f"model_calls={summary['model_calls_dispatched']} "
                 f"peak_generation_in_flight={summary['peak_generation_calls_in_flight']}")
        summary_file.write(json.dumps(summary, ensure_ascii=True, sort_keys=True) + "\n")
    return 0 if result["outcome"] == "completed" else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (UsageError, base.FixtureError, gq.AdapterError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(2)
