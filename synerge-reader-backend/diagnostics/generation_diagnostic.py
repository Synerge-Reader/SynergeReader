#!/usr/bin/env python3
"""Operator-run C04/C05 answer-generation diagnostic over Ollama's native API.

Each fixture's frozen question and frozen evidence are serialized with the
application's own prompt builder (citation_generation.build_generation_prompt)
and sent to each model profile with the registry's answer settings, so every
model receives byte-identical prompts and the same options. The result records
what the server reported, not just what was requested.

Before any generation -- question or smoke -- the installed tag's identity is
read and its full digest compared with the profile's expected digest. An
absent, unreadable or different artifact is never generated with: the run is
recorded as skipped. Nothing is ever pulled, and no other tag is tried.

What this never does: open a database, read credentials or a .env file, call
the SynergeReader website or API, persist an answer or entry_id, pull, delete,
stop or unload a model, or restart anything. Its only network peer is the
Ollama URL the operator passes explicitly; ``--dry-run`` and ``--grade`` use
none.

Gold grading anchors live in grading/ and are read only by ``--grade``, after
generation. They are never part of a request.

Usage (operator, inside an authorized resource window; not run by CI or by
the test suite):

    python diagnostics/generation_diagnostic.py --dry-run
    python diagnostics/generation_diagnostic.py --ollama-url http://127.0.0.1:11434 \\
        --out c04_c05_run.jsonl
    python diagnostics/generation_diagnostic.py --grade c04_c05_run.jsonl
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from answer_evidence import EvidenceBundle, EvidenceItem, EvidenceMode  # noqa: E402
from citation_generation import (  # noqa: E402
    CitationLimits,
    CitationRegistry,
    build_generation_prompt,
)
from generation_profiles import (  # noqa: E402
    GEMMA_CANDIDATE,
    LLAMA_CONTROL,
    SMOKE_OPTIONS,
    GenerationOptions,
    ModelProfile,
    ModelRegistry,
    build_generate_payload,
    default_model_registry,
    normalize_digest,
)
from ollama_generation import OllamaGenerateStream  # noqa: E402

HARNESS_REVISION = "c1-generation-diagnostic-v2"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
GRADING_PATH = Path(__file__).resolve().parent / "grading" / "ircot_gold_anchors.json"
DEFAULT_FIXTURES = ("C04", "C05")
DEFAULT_PROFILES = (LLAMA_CONTROL, GEMMA_CANDIDATE)
# The same cap the /ask route applies; the prompt builder reads it only for
# its evidence budget, so the serialization matches production.
_ROUTE_CITATION_LIMITS = CitationLimits(max_claims_verified=2)
SMOKE_PROMPT = "Reply with the single word: ready."

THINK_POLICIES = ("profile", "false", "omit")

CLEANUP_NOTE = (
    "The client stopped waiting; that does not prove the server stopped "
    "generating. Inspect residency (`ollama ps` or GET /api/ps) and prefer "
    "letting the model expire naturally through its keep_alive. If a stop is "
    "really needed, stop only the identified test runner, and only during an "
    "operator-authorized resource window. This harness never stops a model or "
    "a service."
)


class FixtureError(ValueError):
    pass


class HarnessUsageError(ValueError):
    pass


class _WallClockExceeded(Exception):
    pass


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_hex(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Fixtures and prompt serialization
# --------------------------------------------------------------------------

_FIXTURE_KEYS = {
    "fixture_id", "fixture_revision", "question", "task_prefix", "evidence_supply",
    "prompt_evidence_mode", "evidence_file", "evidence_text_sha256",
}
_EVIDENCE_KEYS = {
    "evidence_id", "filename", "page_start", "page_end", "source_kind",
    "provenance_note", "excerpt_scope_note", "text",
}


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    revision: int
    question: str
    evidence: dict
    fixture_sha256: str
    evidence_sha256: str


def load_fixture(fixture_id: str, fixture_dir: Path = FIXTURE_DIR) -> Fixture:
    """A frozen fixture, or FixtureError. Hashes are over canonical JSON, so a
    CRLF checkout of the same files hashes identically."""
    if not re.fullmatch(r"C\d{2}", fixture_id):
        raise FixtureError(f"unknown fixture id {fixture_id!r}")
    path = Path(fixture_dir) / f"{fixture_id.lower()}.json"
    if not path.is_file():
        raise FixtureError(f"no fixture file for {fixture_id}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if set(spec) != _FIXTURE_KEYS or spec["fixture_id"] != fixture_id:
        raise FixtureError(f"{path.name} does not have the frozen fixture shape")
    evidence_path = Path(fixture_dir) / spec["evidence_file"]
    if evidence_path.parent != Path(fixture_dir) or not evidence_path.is_file():
        raise FixtureError("fixture evidence must be a file in the fixture directory")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if set(evidence) != _EVIDENCE_KEYS:
        raise FixtureError(f"{evidence_path.name} does not have the frozen evidence shape")
    evidence_sha = sha256_hex(evidence["text"])
    if evidence_sha != spec["evidence_text_sha256"]:
        raise FixtureError("evidence text does not match its pinned hash; the fixture is not frozen")
    return Fixture(
        fixture_id=fixture_id,
        revision=spec["fixture_revision"],
        question=spec["question"],
        evidence=evidence,
        fixture_sha256=sha256_hex(_canonical({"fixture": spec, "evidence": evidence})),
        evidence_sha256=evidence_sha,
    )


def build_prompt(fixture: Fixture) -> str:
    """The application's prompt for this fixture, as /ask would build it for
    retrieved passages. The evidence carries only the document text, its file
    name and page -- no question, no answer."""
    evidence = fixture.evidence
    pages = list(range(evidence["page_start"], evidence["page_end"] + 1))
    item = EvidenceItem(
        text=evidence["text"],
        source_type="document_chunk",
        filename=evidence["filename"],
        chunk_id=f"diagnostic:{evidence['evidence_id']}",
        page_start=evidence["page_start"],
        page_end=evidence["page_end"],
        locator_json={"locator_type": "pdf_pages", "page_numbers": pages},
    )
    bundle = EvidenceBundle(mode=EvidenceMode.HYBRID_RETRIEVAL, items=(item,))
    registry = CitationRegistry.from_bundle(bundle, _ROUTE_CITATION_LIMITS)
    return build_generation_prompt(fixture.question, registry)


def load_profiles(profile_ids: Iterable[str], registry: Optional[ModelRegistry] = None) -> list:
    registry = registry or default_model_registry()
    profiles = []
    for profile_id in profile_ids:
        profile = registry.get(profile_id)
        if profile is None:
            raise HarnessUsageError(f"unknown profile {profile_id!r}")
        if profile.expected_digest is None:
            raise HarnessUsageError(f"profile {profile_id!r} has no expected digest to verify")
        profiles.append(profile)
    return profiles


def request_body(profile: ModelProfile, prompt: str, *, think: Optional[bool] = None,
                 options: Optional[GenerationOptions] = None) -> dict:
    return build_generate_payload(
        profile.model_tag, prompt, options or profile.answer_options, profile.keep_alive,
        stream=True, think=think,
    )


# --------------------------------------------------------------------------
# Transport (stdlib only; injectable for tests)
# --------------------------------------------------------------------------


def validate_base_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url or "")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HarnessUsageError("--ollama-url must be an http(s) URL with a host")
    if parsed.username or parsed.password:
        raise HarnessUsageError("--ollama-url must not carry credentials")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise HarnessUsageError("--ollama-url must be the server root, e.g. http://127.0.0.1:11434")
    return f"{parsed.scheme}://{parsed.netloc}"


class OllamaHttp:
    """Native Ollama endpoints only. Read-only calls plus /api/generate."""

    def __init__(self, base_url: str, *, read_timeout: float) -> None:
        self.base_url = validate_base_url(base_url)
        self.read_timeout = read_timeout

    def _request(self, path: str, body: Optional[dict]):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method="GET" if body is None else "POST",
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=self.read_timeout)

    def get_json(self, path: str) -> dict:
        with self._request(path, None) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_json(self, path: str, body: dict) -> dict:
        with self._request(path, body) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_stream(self, path: str, body: dict) -> Iterator[bytes]:
        with self._request(path, body) as response:
            while True:
                chunk = response.read1(512) if hasattr(response, "read1") else response.read(512)
                if not chunk:
                    return
                yield chunk


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


def _error_summary(exc: BaseException) -> dict:
    """Type and status only: server bodies can carry paths or internals."""
    summary = {"kind": "error", "type": type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        summary.update(kind="http", status=exc.code)
    elif isinstance(exc, (socket.timeout, TimeoutError)) or (
        isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (socket.timeout, TimeoutError))
    ):
        summary["kind"] = "timeout"
    elif isinstance(exc, (urllib.error.URLError, OSError)):
        summary["kind"] = "connection"
    return summary


def residency(http) -> dict:
    try:
        body = http.get_json("/api/ps")
    except Exception as exc:
        return {"error": _error_summary(exc)}
    keep = ("name", "model", "digest", "size", "size_vram", "context_length", "expires_at")
    return {"models": [{k: m.get(k) for k in keep if k in m} for m in body.get("models", [])]}


def model_identity(http, profile: ModelProfile) -> dict:
    """Expected identity (the registry pin) beside what the server reports.

    ``check`` is ``match`` only when the exact tag is listed with a full digest
    equal to the expected one; otherwise ``absent``, ``unreadable`` or
    ``mismatch``, and no generation may follow.
    """
    identity = {
        "model_tag": profile.model_tag,
        "expected_digest": profile.expected_digest,
        "observed_digest": None,
        "check": "unreadable",
        "errors": {},
    }
    try:
        identity["runtime_version"] = http.get_json("/api/version").get("version")
    except Exception as exc:
        identity["errors"]["version"] = _error_summary(exc)
    try:
        listed = [
            m for m in http.get_json("/api/tags").get("models", [])
            if m.get("name") == profile.model_tag or m.get("model") == profile.model_tag
        ]
    except Exception as exc:
        identity["errors"]["tags"] = _error_summary(exc)
    else:
        identity["installed"] = bool(listed)
        if not listed:
            identity["check"] = "absent"
        else:
            entry = listed[0]
            observed = normalize_digest(entry.get("digest"))
            identity.update(
                observed_digest=observed,
                size=entry.get("size"),
                modified_at=entry.get("modified_at"),
                details=entry.get("details"),
            )
            if observed is None:
                identity["check"] = "unreadable"
            elif observed != profile.expected_digest:
                identity["check"] = "mismatch"
            else:
                identity["check"] = "match"
    try:
        shown = http.post_json("/api/show", {"model": profile.model_tag})
        identity["capabilities"] = shown.get("capabilities")
        # The Modelfile's own parameters (e.g. top_k / top_p / stop): these
        # are what a request inherits when it does not send them.
        identity["modelfile_parameters"] = shown.get("parameters")
        info = shown.get("model_info") or {}
        identity["trained_context_length"] = {
            key: value for key, value in info.items() if key.endswith("context_length")
        }
    except Exception as exc:
        identity["errors"]["show"] = _error_summary(exc)
    return identity


def thinking_decision(policy: str, profile: ModelProfile, identity: dict) -> dict:
    """The effective think field for one call, and why.

    * ``profile`` (default): follow the registry. A profile whose policy is
      ``think: false`` sends it only when the server reports thinking
      capability, and otherwise refuses to generate rather than risk an
      unverified thinking mode. A profile with no think policy sends nothing.
    * ``false``: send ``think: false`` where the server reports thinking
      capability; elsewhere omit it and say so.
    * ``omit``: never send a think field.
    """
    if policy not in THINK_POLICIES:
        raise HarnessUsageError(f"unknown thinking policy {policy!r}")
    capabilities = identity.get("capabilities")
    reports_thinking = isinstance(capabilities, list) and "thinking" in capabilities
    decision = {
        "policy": policy,
        "profile_think": profile.think,
        "server_reports_thinking": reports_thinking if capabilities is not None else None,
        "send": None,
        "refused": False,
    }
    wants_false = policy == "false" or (policy == "profile" and profile.think is False)
    if policy == "omit":
        decision["outcome"] = "omitted: --think omit"
    elif not wants_false:
        decision["outcome"] = "omitted: the profile sends no think field"
    elif reports_thinking:
        decision.update(send=False, outcome="think:false selected: the server reports thinking capability")
    elif policy == "profile":
        decision.update(
            refused=True,
            outcome="refused: the profile requires think:false but the server does not report thinking capability",
        )
    else:
        decision["outcome"] = "requested false but omitted: the server does not report thinking capability"
    return decision


def _generate(http, body: dict, *, max_seconds: float, clock: Callable[[], float]) -> dict:
    reader = OllamaGenerateStream()
    parts = []
    error = None
    first_token = None
    started = clock()
    try:
        for chunk in http.post_stream("/api/generate", body):
            tokens = reader.feed(chunk)
            if tokens and first_token is None:
                first_token = clock() - started
            parts.extend(tokens)
            if clock() - started > max_seconds:
                raise _WallClockExceeded()
        tail = reader.close()
        if tail and first_token is None:
            first_token = clock() - started
        parts.extend(tail)
    except _WallClockExceeded:
        error = {"kind": "timeout", "type": "wall_clock", "max_seconds": max_seconds}
    except Exception as exc:
        error = _error_summary(exc)
    text = "".join(parts)
    return {
        "generated": True,
        "skip_reason": None,
        "response": {"text": text, "present": bool(text.strip()), "chars": len(text)},
        "thinking": {"present": reader.thinking_chars > 0, "chars": reader.thinking_chars},
        "stream": {
            "failure": reader.failure,
            "done_reason": reader.done_reason,
            "malformed_lines": reader.malformed_lines,
        },
        "server_metrics": dict(reader.metrics),
        # Client-side wall clock: to the first non-empty answer token (None if
        # none arrived) and to the end of the stream.
        "client_first_token_seconds": None if first_token is None else round(first_token, 3),
        "client_elapsed_seconds": round(clock() - started, 3),
        "error": error,
        "cleanup_note": CLEANUP_NOTE if error and error["kind"] == "timeout" else None,
    }


def _skipped(reason: str) -> dict:
    return {
        "generated": False,
        "skip_reason": reason,
        "response": None,
        "thinking": None,
        "stream": None,
        "server_metrics": {},
        "client_first_token_seconds": None,
        "client_elapsed_seconds": None,
        "error": None,
        "cleanup_note": None,
    }


def _attempt(http, record: dict, profile: ModelProfile, prompt: str, options: GenerationOptions,
             *, think_policy: str, max_seconds: float, clock: Callable[[], float]) -> None:
    """Identity and thinking gates, then at most one /api/generate call."""
    identity = model_identity(http, profile)
    decision = thinking_decision(think_policy, profile, identity)
    record["identity"] = identity
    record["thinking_policy"] = decision
    body = request_body(profile, prompt, think=decision["send"], options=options)
    record["requested"] = {
        "options": body["options"],
        "keep_alive": body["keep_alive"],
        "stream": body["stream"],
        "think": decision["send"] if decision["send"] is not None else "omitted",
        # Whether this request was actually sent to /api/generate.
        "sent": False,
        "note": "requested settings; server limits may differ (see server_metrics, residency)",
    }
    if identity["check"] != "match":
        record.update(_skipped(
            f"identity {identity['check']}: generation refused; expected digest "
            "not confirmed as installed (nothing is pulled and no other tag is tried)"
        ))
        return
    if decision["refused"]:
        record.update(_skipped(f"thinking policy {decision['outcome']}"))
        return
    record["requested"]["sent"] = True
    record["residency_before"] = residency(http)
    record.update(_generate(http, body, max_seconds=max_seconds, clock=clock))
    record["residency_after"] = residency(http)


def run_one(
    http,
    fixture: Fixture,
    profile: ModelProfile,
    *,
    think_policy: str = "profile",
    max_seconds: float = 600.0,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], str] = _utc_now,
) -> dict:
    prompt = build_prompt(fixture)
    record = {
        "record_type": "generation",
        "evaluation_id": f"diag-{uuid.uuid4().hex}",
        "harness_revision": HARNESS_REVISION,
        "started_at": now(),
        "fixture": {
            "id": fixture.fixture_id,
            "revision": fixture.revision,
            "sha256": fixture.fixture_sha256,
            "evidence_sha256": fixture.evidence_sha256,
            "evidence_supply": "manual_evidence",
            "evidence_source_kind": fixture.evidence["source_kind"],
        },
        "prompt": {
            "sha256": sha256_hex(prompt),
            "chars": len(prompt),
            "builder": "citation_generation.build_generation_prompt",
            "evidence_mode": EvidenceMode.HYBRID_RETRIEVAL.value,
        },
        "profile": {
            "profile_id": profile.profile_id,
            "model_tag": profile.model_tag,
            "production_selectable": profile.selectable,
        },
    }
    _attempt(http, record, profile, prompt, profile.answer_options,
             think_policy=think_policy, max_seconds=max_seconds, clock=clock)
    record["finished_at"] = now()
    return record


def run_smoke(
    http,
    profile: ModelProfile,
    *,
    think_policy: str = "profile",
    max_seconds: float = 300.0,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], str] = _utc_now,
) -> dict:
    """Load-and-emit readiness probe at the separate 4096/32 smoke settings,
    behind the same identity and thinking gates as a question run."""
    record = {
        "record_type": "smoke",
        "evaluation_id": f"diag-{uuid.uuid4().hex}",
        "harness_revision": HARNESS_REVISION,
        "started_at": now(),
        "profile": {"profile_id": profile.profile_id, "model_tag": profile.model_tag},
    }
    _attempt(http, record, profile, SMOKE_PROMPT, SMOKE_OPTIONS,
             think_policy=think_policy, max_seconds=max_seconds, clock=clock)
    record["finished_at"] = now()
    return record


def status_line(record: dict) -> str:
    """A console line with outcomes and timings only -- never answer text."""
    label = record["fixture"]["id"] if "fixture" in record else record["record_type"]
    return (
        f"{label} {record['profile']['profile_id']}: generated={record['generated']} "
        f"skip={record['skip_reason']!r} failure={(record['stream'] or {}).get('failure')} "
        f"error={record['error']} first_token_s={record['client_first_token_seconds']} "
        f"elapsed_s={record['client_elapsed_seconds']}\n"
    )


# --------------------------------------------------------------------------
# Grading (post hoc; never during generation)
# --------------------------------------------------------------------------


def grade_record(record: dict, anchors: dict) -> Optional[dict]:
    if record.get("record_type") != "generation" or not record.get("generated"):
        return None
    fixture_id = record["fixture"]["id"]
    spec = anchors["fixtures"].get(fixture_id)
    if spec is None:
        return None
    text = record["response"]["text"]
    groups = []
    for group in spec["anchor_groups"]:
        if "all" in group:
            hit = all(re.search(p, text, re.IGNORECASE) for p in group["all"])
        else:
            hit = any(re.search(p, text, re.IGNORECASE) for p in group["any"])
        groups.append({"fact": group["fact"], "hit": bool(hit)})
    return {
        "evaluation_id": record["evaluation_id"],
        "fixture": fixture_id,
        "profile_id": record["profile"]["profile_id"],
        "anchor_groups": groups,
        "all_anchor_groups_hit": all(g["hit"] for g in groups),
        # The fixture supplies exactly one evidence block, [C1] = page 4.
        "cites_supplied_page": "[C1]" in text,
        "stream_failure": record["stream"]["failure"],
        "error": record["error"],
        "note": "Heuristic anchors only; the operator grades each answer manually.",
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse(argv):
    parser = argparse.ArgumentParser(description="C04/C05 native-API generation diagnostic")
    parser.add_argument("--fixture", action="append", dest="fixtures", help="C04 or C05 (repeatable)")
    parser.add_argument("--profile", action="append", dest="profiles", help="registry profile id (repeatable)")
    parser.add_argument("--ollama-url", help="Ollama server root; required for live runs")
    parser.add_argument("--out", help="new JSONL file for live results (never overwritten)")
    parser.add_argument("--dry-run", action="store_true", help="print the exact requests; no network")
    parser.add_argument("--smoke", action="store_true", help="only run the 4096/32 readiness probe per profile")
    parser.add_argument("--think", choices=THINK_POLICIES, default="profile",
                        help="'profile' follows the registry (Gemma think:false, Llama none); "
                             "'false' sends think:false only where the server reports thinking; "
                             "'omit' never sends it")
    # Per-socket-read timeout, not the total deadline (--max-seconds is). The
    # first read waits through the model load: the observed cold Gemma load
    # was about 91.783 s, and 180 s leaves margin so shared-GPU contention
    # does not turn a slow load into a false failure.
    parser.add_argument("--read-timeout", type=float, default=180.0,
                        help="seconds to wait for any one socket read, including the first byte "
                             "after a cold model load (observed ~91.8 s for Gemma); not the total "
                             "deadline, which is --max-seconds")
    parser.add_argument("--max-seconds", type=float, default=600.0, help="wall-clock cap per generation")
    parser.add_argument("--grade", metavar="RESULTS_JSONL", help="grade a results file against gold anchors")
    return parser.parse_args(argv)


def main(argv=None, *, http_factory=OllamaHttp, stdout=sys.stdout,
         registry_factory: Callable[[], ModelRegistry] = default_model_registry) -> int:
    args = _parse(argv)

    if args.grade:
        anchors = json.loads(GRADING_PATH.read_text(encoding="utf-8"))
        for line in Path(args.grade).read_text(encoding="utf-8").splitlines():
            if line.strip():
                graded = grade_record(json.loads(line), anchors)
                if graded is not None:
                    stdout.write(json.dumps(graded) + "\n")
        return 0

    fixtures = [load_fixture(f) for f in (args.fixtures or DEFAULT_FIXTURES)]
    profiles = load_profiles(args.profiles or DEFAULT_PROFILES, registry_factory())

    if args.dry_run:
        for fixture in fixtures:
            prompt = build_prompt(fixture)
            for profile in profiles:
                stdout.write(json.dumps({
                    "fixture": fixture.fixture_id,
                    "fixture_sha256": fixture.fixture_sha256,
                    "evidence_sha256": fixture.evidence_sha256,
                    "prompt_sha256": sha256_hex(prompt),
                    "profile_id": profile.profile_id,
                    "expected_digest": profile.expected_digest,
                    "think_policy": args.think,
                    "profile_think": profile.think,
                    "note": "a live run sends think:false only after the server reports thinking "
                            "capability, and generates only after the expected digest is confirmed",
                    "request": request_body(profile, prompt),
                }, ensure_ascii=False) + "\n")
        return 0

    if not args.ollama_url or not args.out:
        raise HarnessUsageError("a live run needs --ollama-url and --out (or use --dry-run)")
    out_path = Path(args.out)
    if out_path.exists():
        raise HarnessUsageError(f"{out_path} already exists; results are never overwritten")
    http = http_factory(args.ollama_url, read_timeout=args.read_timeout)

    with out_path.open("x", encoding="utf-8") as out:
        out.write(json.dumps({
            "record_type": "run",
            "run_id": f"run-{uuid.uuid4().hex}",
            "harness_revision": HARNESS_REVISION,
            "started_at": _utc_now(),
            "ollama_url": http.base_url,
            "fixtures": [f.fixture_id for f in fixtures],
            "profiles": [p.profile_id for p in profiles],
            "think_policy": args.think,
            "smoke_only": args.smoke,
            "persists_answers": False,
        }) + "\n")
        out.flush()
        if args.smoke:
            for profile in profiles:
                record = run_smoke(http, profile, think_policy=args.think, max_seconds=args.max_seconds)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                stdout.write(status_line(record))
            return 0
        # Sequential on purpose: C04/llama, C04/gemma, C05/llama, C05/gemma.
        for fixture in fixtures:
            for profile in profiles:
                record = run_one(http, fixture, profile, think_policy=args.think,
                                 max_seconds=args.max_seconds)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                stdout.write(status_line(record))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HarnessUsageError, FixtureError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(2)
