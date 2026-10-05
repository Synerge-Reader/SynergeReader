"""The operator C04/C05 diagnostic harness, exercised only against fakes.

No Ollama, database, or network: every live-path test injects an in-memory
transport and, where identity gates are exercised, a synthetic registry with
synthetic digests. Output files go to pytest's temporary directory.
"""

import ast
import dataclasses
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from citation_generation import ABSENCE_RULE_PARTIAL_EVIDENCE, CITATION_RULES
from diagnostics import generation_diagnostic as harness
from generation_profiles import GEMMA_CANDIDATE, LLAMA_CONTROL, ModelRegistry, default_model_registry

_HARNESS_PATH = Path(harness.__file__)
_EVIDENCE_SHA = "0bc45a825ae0a2bce85877d30f6a6fcced4802713bf3fb66db9b13af50f9c8c7"
# Revision 2: only the excerpt-scope note changed; evidence text and prompts did not.
_FIXTURE_SHAS = {
    "C04": "6fd9f3172bb5323766aa101b619704f5c8aa0a6bff46bcf65ac31f227e05880e",
    "C05": "6b32291d9e313750e597a529f91f56a4cb2f4f2521772873c3db4130ca4f12b0",
}
_PROMPT_SHAS = {
    "C04": "492a1828a1297822c7054fb8614b6b8755aa9a95becc81407a0133a94d6cfec7",
    "C05": "fc3e74cdc22ee70227f75e446dfdca260fba1fdfc05d2bda71a5122cf2383ed6",
}
_C04 = (
    "What two operations does IRCoT alternate, and what text is used as the query "
    "during its later retrieval steps?"
)
_C05 = (
    "What conditions terminate IRCoT, and what limits were used for reasoning steps "
    "and collected paragraphs in the experiments?"
)
SYNTHETIC_LLAMA = "1a" * 32
SYNTHETIC_GEMMA = "9a" * 32
_SYNTHETIC_DIGESTS = {"llama3.1:8b": SYNTHETIC_LLAMA, "gemma4:31b": SYNTHETIC_GEMMA}


def _synthetic_registry():
    """The production profiles with synthetic expected digests."""
    base = default_model_registry()
    return ModelRegistry(
        profiles=tuple(
            dataclasses.replace(p, expected_digest=_SYNTHETIC_DIGESTS[p.model_tag]) for p in base.profiles
        ),
        default_profile_id=LLAMA_CONTROL,
        verifier_profile_id=LLAMA_CONTROL,
    )


def _profile(profile_id):
    [profile] = harness.load_profiles([profile_id], _synthetic_registry())
    return profile


def _ndjson(*records):
    return "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")


_DONE = {
    "response": "", "done": True, "done_reason": "stop", "total_duration": 5_000_000_000,
    "load_duration": 2_000_000_000, "prompt_eval_count": 1210, "prompt_eval_duration": 900_000_000,
    "eval_count": 64, "eval_duration": 2_000_000_000,
}
_ANSWER = "IRCoT alternates CoT generation and retrieval [C1]."


class FakeHttp:
    base_url = "http://fake-ollama:11434"

    def __init__(self, stream=None, *, capabilities=None, stream_error=None, chunk_size=5,
                 digests=None, tags_error=None, show_error=None):
        self.stream = stream if stream is not None else _ndjson({"response": _ANSWER}, _DONE)
        self.capabilities = capabilities
        self.stream_error = stream_error
        self.chunk_size = chunk_size
        self.digests = dict(_SYNTHETIC_DIGESTS if digests is None else digests)
        self.tags_error = tags_error
        self.show_error = show_error
        self.calls = []

    def get_json(self, path):
        self.calls.append(("GET", path, None))
        if path == "/api/version":
            return {"version": "0.0.0-test"}
        if path == "/api/tags":
            if self.tags_error:
                raise self.tags_error
            return {"models": [{"name": tag, "digest": digest, "size": 1} for tag, digest in self.digests.items()]}
        if path == "/api/ps":
            return {"models": [{"name": "llama3.1:8b", "size_vram": 5, "context_length": 8192, "template": "x"}]}
        raise AssertionError(f"unexpected GET {path}")

    def post_json(self, path, body):
        self.calls.append(("POST", path, body))
        assert path == "/api/show"
        if self.show_error:
            raise self.show_error
        if self.capabilities is not None:
            capabilities = list(self.capabilities)
        else:
            capabilities = ["completion", "thinking"] if body["model"] == "gemma4:31b" else ["completion"]
        return {
            "capabilities": capabilities,
            "parameters": "top_k 40\ntop_p 0.9",
            "model_info": {"llama.context_length": 131072, "general.architecture": "llama"},
            "license": "long text that must not be recorded",
        }

    def post_stream(self, path, body):
        self.calls.append(("STREAM", path, body))
        assert path == "/api/generate"
        data = self.stream
        for index in range(0, len(data), self.chunk_size):
            yield data[index:index + self.chunk_size]
            if self.stream_error is not None and index >= self.chunk_size:
                raise self.stream_error

    @property
    def generate_bodies(self):
        return [body for kind, _, body in self.calls if kind == "STREAM"]


def _main(args, http=None, stdout=None):
    return harness.main(
        args,
        http_factory=(lambda url, read_timeout: http) if http is not None else harness.OllamaHttp,
        stdout=stdout or io.StringIO(),
        registry_factory=_synthetic_registry,
    )


def _dry_run(*args):
    out = io.StringIO()
    assert harness.main(["--dry-run", *args], stdout=out) == 0
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _gold_strings():
    anchors = json.loads(harness.GRADING_PATH.read_text(encoding="utf-8"))
    for spec in anchors["fixtures"].values():
        yield from spec["required_facts"]


# --- frozen fixtures ----------------------------------------------------------


def test_fixtures_are_frozen_with_the_exact_questions_and_shared_page4_evidence():
    c04, c05 = harness.load_fixture("C04"), harness.load_fixture("C05")
    assert c04.question == _C04 and c05.question == _C05
    assert c04.evidence_sha256 == c05.evidence_sha256 == _EVIDENCE_SHA
    assert (c04.revision, c05.revision) == (2, 2)
    assert {"C04": c04.fixture_sha256, "C05": c05.fixture_sha256} == _FIXTURE_SHAS
    for fixture in (c04, c05):
        assert harness.sha256_hex(harness.build_prompt(fixture)) == _PROMPT_SHAS[fixture.fixture_id]
        assert fixture.evidence["filename"] == "IR-CoT.pdf"
        assert (fixture.evidence["page_start"], fixture.evidence["page_end"]) == (4, 4)
        assert fixture.evidence["source_kind"] == "normalized_excerpt"
        assert "set to 8 in our experiments" in fixture.evidence["text"]
        assert "set to 15 in our experiments" in fixture.evidence["text"]


def test_the_excerpt_note_is_durable_not_environment_specific():
    evidence = harness.load_fixture("C04").evidence
    note = evidence["excerpt_scope_note"]
    assert "normalized relevant excerpt" in note and "reading order" in note
    assert "not a byte-exact whole-page extraction" in note
    rendered = json.dumps(evidence).lower()
    assert "locally" not in rendered and "not available" not in rendered


def test_fixture_hashes_do_not_depend_on_line_endings(tmp_path):
    copy = tmp_path / "fixtures"
    shutil.copytree(harness.FIXTURE_DIR, copy)
    for path in copy.glob("*.json"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert harness.load_fixture("C04", copy).fixture_sha256 == _FIXTURE_SHAS["C04"]


def test_tampered_evidence_is_refused(tmp_path):
    copy = tmp_path / "fixtures"
    shutil.copytree(harness.FIXTURE_DIR, copy)
    evidence_path = copy / "ircot_page4_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["text"] = evidence["text"].replace("set to 8", "set to 9")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(harness.FixtureError):
        harness.load_fixture("C04", copy)


def test_unknown_fixture_ids_are_refused():
    for bad in ("C06", "../c04", "c04"):
        with pytest.raises(harness.FixtureError):
            harness.load_fixture(bad)


# --- request serialization ----------------------------------------------------


def test_the_prompt_is_the_application_serialization_of_question_and_evidence():
    fixture = harness.load_fixture("C05")
    prompt = harness.build_prompt(fixture)
    assert prompt.startswith(
        "<evidence>\n[C1]\nDocument: IR-CoT.pdf\nLocation: page 4\nEvidence:\n" + fixture.evidence["text"]
    )
    assert f"<question>\n{_C05}\n</question>" in prompt
    assert CITATION_RULES in prompt and ABSENCE_RULE_PARTIAL_EVIDENCE in prompt


def test_every_model_gets_the_same_prompt_and_bounded_options_in_order():
    lines = _dry_run()
    assert [(l["fixture"], l["profile_id"]) for l in lines] == [
        ("C04", "llama_control"), ("C04", "gemma_candidate"),
        ("C05", "llama_control"), ("C05", "gemma_candidate"),
    ]
    for fixture in ("C04", "C05"):
        pair = [l for l in lines if l["fixture"] == fixture]
        assert pair[0]["request"]["prompt"] == pair[1]["request"]["prompt"]
        assert pair[0]["prompt_sha256"] == pair[1]["prompt_sha256"] == _PROMPT_SHAS[fixture]
    for line in lines:
        request = line["request"]
        assert request["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
        assert request["keep_alive"] == "5m" and request["stream"] is True
        assert line["evidence_sha256"] == _EVIDENCE_SHA
    assert {l["request"]["model"] for l in lines} == {"llama3.1:8b", "gemma4:31b"}
    assert {l["profile_id"]: l["expected_digest"] for l in lines} == {
        "llama_control": default_model_registry().get(LLAMA_CONTROL).expected_digest,
        "gemma_candidate": default_model_registry().get(GEMMA_CANDIDATE).expected_digest,
    }


def test_gold_anchors_never_reach_a_request(monkeypatch, tmp_path):
    for line in _dry_run():
        rendered = json.dumps(line["request"])
        for gold in _gold_strings():
            assert gold not in rendered
        assert "expected_citation_pages" not in rendered and "anchor_groups" not in rendered

    # A live run never even opens the grading file.
    monkeypatch.setattr(harness, "GRADING_PATH", tmp_path / "missing.json")
    http = FakeHttp()
    _main(["--ollama-url", "http://x:1", "--out", str(tmp_path / "r.jsonl")], http)
    assert len(http.generate_bodies) == 4


# --- identity gate ---------------------------------------------------------------


def test_a_matching_identity_generates_and_records_expected_and_observed_separately():
    http = FakeHttp(_ndjson({"thinking": "plan "}, {"response": "IRCoT alternates "},
                            {"response": "CoT and retrieval [C1]."}, _DONE))
    record = harness.run_one(http, harness.load_fixture("C04"), _profile(GEMMA_CANDIDATE))

    assert record["generated"] is True and record["skip_reason"] is None
    assert record["identity"]["check"] == "match"
    assert record["identity"]["expected_digest"] == SYNTHETIC_GEMMA
    assert record["identity"]["observed_digest"] == SYNTHETIC_GEMMA
    assert "digest" not in record["identity"], "no ambiguous unlabeled digest field"
    assert record["evaluation_id"].startswith("diag-")
    assert "entry_id" not in json.dumps(record)
    assert record["fixture"]["sha256"] == _FIXTURE_SHAS["C04"]
    assert record["profile"] == {"profile_id": "gemma_candidate", "model_tag": "gemma4:31b", "production_selectable": False}
    assert record["requested"]["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1200}
    assert record["identity"]["modelfile_parameters"] == "top_k 40\ntop_p 0.9"
    assert record["identity"]["trained_context_length"] == {"llama.context_length": 131072}
    assert "license" not in json.dumps(record["identity"])
    assert record["residency_before"]["models"] == [{"name": "llama3.1:8b", "size_vram": 5, "context_length": 8192}]
    assert record["response"] == {"text": "IRCoT alternates CoT and retrieval [C1].", "present": True, "chars": 40}
    assert record["thinking"] == {"present": True, "chars": 5}
    assert record["stream"] == {"failure": None, "done_reason": "stop", "malformed_lines": 0}
    assert record["server_metrics"]["prompt_eval_count"] == 1210
    assert record["error"] is None and record["cleanup_note"] is None


@pytest.mark.parametrize(
    "http_kwargs, check",
    [
        (dict(digests={"llama3.1:8b": SYNTHETIC_LLAMA}), "absent"),
        (dict(digests={**_SYNTHETIC_DIGESTS, "gemma4:31b": "77" * 32}), "mismatch"),
        (dict(digests={**_SYNTHETIC_DIGESTS, "gemma4:31b": "garbled"}), "unreadable"),
        (dict(tags_error=urllib.error.URLError("refused /srv/secret")), "unreadable"),
    ],
    ids=["absent", "mismatch", "malformed-digest", "tags-unreadable"],
)
@pytest.mark.parametrize("smoke", [False, True], ids=["question", "smoke"])
def test_a_wrong_or_unknown_identity_never_reaches_generate(http_kwargs, check, smoke):
    http = FakeHttp(**http_kwargs)
    profile = _profile(GEMMA_CANDIDATE)
    if smoke:
        record = harness.run_smoke(http, profile)
    else:
        record = harness.run_one(http, harness.load_fixture("C05"), profile)

    assert http.generate_bodies == [], "no /api/generate call without a confirmed identity"
    assert not [c for c in http.calls if c[1] in ("/api/pull", "/api/delete")]
    assert record["generated"] is False and record["requested"]["sent"] is False
    assert record["identity"]["check"] == check
    assert record["identity"]["expected_digest"] == SYNTHETIC_GEMMA
    assert record["skip_reason"].startswith(f"identity {check}: generation refused")
    assert record["response"] is None and record["thinking"] is None
    assert "/srv/secret" not in json.dumps(record)
    if check == "mismatch":
        assert record["identity"]["observed_digest"] == "77" * 32


def test_only_the_wrong_profile_is_skipped_in_a_sequential_run(tmp_path):
    out = tmp_path / "run.jsonl"
    stdout = io.StringIO()
    http = FakeHttp(digests={**_SYNTHETIC_DIGESTS, "gemma4:31b": "77" * 32})
    assert _main(["--ollama-url", "http://x:1", "--out", str(out)], http, stdout) == 0

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(r.get("fixture", {}).get("id"), r.get("profile", {}).get("profile_id"), r.get("generated"))
            for r in records[1:]] == [
        ("C04", "llama_control", True), ("C04", "gemma_candidate", False),
        ("C05", "llama_control", True), ("C05", "gemma_candidate", False),
    ]
    assert [b["model"] for b in http.generate_bodies] == ["llama3.1:8b", "llama3.1:8b"]
    console = stdout.getvalue()
    assert "generated=False" in console and "identity mismatch" in console
    assert _ANSWER not in console and "IRCoT" not in console, "answer text never reaches the console"


# --- thinking policy ---------------------------------------------------------------


def test_default_policy_gives_gemma_think_false_and_llama_nothing():
    http = FakeHttp()
    gemma = harness.run_one(http, harness.load_fixture("C04"), _profile(GEMMA_CANDIDATE))
    llama = harness.run_one(http, harness.load_fixture("C04"), _profile(LLAMA_CONTROL))

    gemma_body, llama_body = http.generate_bodies
    assert gemma_body["think"] is False
    assert "think" not in llama_body
    assert gemma["thinking_policy"]["outcome"].startswith("think:false selected")
    assert gemma["requested"]["think"] is False and gemma["requested"]["sent"] is True
    assert llama["thinking_policy"]["outcome"] == "omitted: the profile sends no think field"
    assert llama["requested"]["think"] == "omitted"


@pytest.mark.parametrize("smoke", [False, True], ids=["question", "smoke"])
@pytest.mark.parametrize(
    "capabilities, show_error",
    [(["completion"], None), (None, urllib.error.URLError("x"))],
    ids=["no-thinking-capability", "show-unreadable"],
)
def test_a_required_think_false_without_server_evidence_fails_closed(smoke, capabilities, show_error):
    http = FakeHttp(capabilities=capabilities, show_error=show_error)
    profile = _profile(GEMMA_CANDIDATE)
    record = (harness.run_smoke(http, profile) if smoke
              else harness.run_one(http, harness.load_fixture("C04"), profile))

    assert http.generate_bodies == []
    assert record["generated"] is False
    assert record["thinking_policy"]["refused"] is True
    assert record["skip_reason"].startswith("thinking policy refused")


@pytest.mark.parametrize("smoke", [False, True], ids=["question", "smoke"])
def test_think_false_flag_is_honoured_on_smoke_and_question_runs(smoke):
    with_thinking = FakeHttp(capabilities=["completion", "thinking"])
    without = FakeHttp(capabilities=["completion"])
    llama = _profile(LLAMA_CONTROL)

    def run(http):
        if smoke:
            return harness.run_smoke(http, llama, think_policy="false")
        return harness.run_one(http, harness.load_fixture("C04"), llama, think_policy="false")

    sent = run(with_thinking)
    assert with_thinking.generate_bodies[0]["think"] is False
    assert sent["thinking_policy"]["outcome"].startswith("think:false selected")

    omitted = run(without)
    assert "think" not in without.generate_bodies[0]
    assert omitted["generated"] is True
    assert omitted["thinking_policy"]["outcome"].startswith("requested false but omitted")


def test_omit_policy_never_sends_think():
    http = FakeHttp()
    record = harness.run_smoke(http, _profile(GEMMA_CANDIDATE), think_policy="omit")
    assert "think" not in http.generate_bodies[0]
    assert record["thinking_policy"]["outcome"] == "omitted: --think omit"


def test_smoke_uses_the_separate_4096_32_settings_and_the_profile_policy():
    http = FakeHttp(_ndjson({"response": "ready"}, _DONE))
    record = harness.run_smoke(http, _profile(GEMMA_CANDIDATE))
    assert record["record_type"] == "smoke"
    [body] = http.generate_bodies
    assert body["options"] == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 32}
    assert body["think"] is False


# --- timing, failures and cleanup wording -------------------------------------


def test_first_token_and_total_time_are_recorded():
    ticks = iter([0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] + [10.0] * 200)
    http = FakeHttp(_ndjson({"thinking": "x" * 20}, {"response": "Answer [C1]."}, _DONE), chunk_size=10)
    record = harness.run_one(http, harness.load_fixture("C04"), _profile(GEMMA_CANDIDATE),
                             clock=lambda: next(ticks))
    assert record["client_first_token_seconds"] is not None
    assert 0 < record["client_first_token_seconds"] <= record["client_elapsed_seconds"]


def test_no_answer_token_means_no_first_token_time():
    http = FakeHttp(_ndjson({"thinking": "only thinking"}, _DONE))
    record = harness.run_one(http, harness.load_fixture("C04"), _profile(GEMMA_CANDIDATE))
    assert record["client_first_token_seconds"] is None
    assert record["response"]["present"] is False and record["thinking"]["present"] is True
    assert record["client_elapsed_seconds"] is not None


@pytest.mark.parametrize(
    "error, kind",
    [
        (socket.timeout("timed out"), "timeout"),
        (urllib.error.URLError(socket.timeout("timed out")), "timeout"),
        (ConnectionResetError("reset"), "connection"),
        (urllib.error.HTTPError("http://x/api/generate", 500, "secret /srv/path", {}, None), "http"),
    ],
    ids=["socket-timeout", "urlerror-timeout", "reset", "http-500"],
)
def test_transport_failures_are_recorded_without_server_text(error, kind):
    record = harness.run_one(FakeHttp(stream_error=error), harness.load_fixture("C04"), _profile(LLAMA_CONTROL))
    assert record["error"]["kind"] == kind
    assert "secret" not in json.dumps(record) and "/srv/path" not in json.dumps(record)
    assert record["stream"]["failure"] is not None
    assert record["cleanup_note"] == (harness.CLEANUP_NOTE if kind == "timeout" else None)


def test_the_cleanup_note_prefers_natural_expiry_and_never_casual_stops():
    note = harness.CLEANUP_NOTE
    assert "does not prove the server stopped" in note
    assert "Inspect residency" in note and "expire naturally" in note
    assert "only the identified test runner" in note
    assert "operator-authorized resource window" in note
    assert "never stops a model or a service" in note
    assert "or stop it manually" not in note


def test_the_wall_clock_cap_stops_waiting_and_asks_for_safe_cleanup():
    ticks = iter(range(0, 10_000, 100))
    record = harness.run_one(FakeHttp(chunk_size=3), harness.load_fixture("C04"), _profile(LLAMA_CONTROL),
                             max_seconds=150, clock=lambda: next(ticks))
    assert record["error"] == {"kind": "timeout", "type": "wall_clock", "max_seconds": 150}
    assert record["cleanup_note"] == harness.CLEANUP_NOTE


@pytest.mark.parametrize(
    "stream, failure",
    [
        (_ndjson({"response": "partial"}, {"error": "runner died /srv/x"}), "model_error"),
        (_ndjson({"response": "cut"}, dict(_DONE, done_reason="length")), "output_limit"),
        (b'{"response": "a"}\n{bad\n', "malformed"),
    ],
    ids=["model-error", "output-limit", "malformed"],
)
def test_stream_outcomes_are_reported(stream, failure):
    record = harness.run_one(FakeHttp(stream), harness.load_fixture("C04"), _profile(LLAMA_CONTROL))
    assert record["stream"]["failure"] == failure
    assert "/srv/x" not in json.dumps(record)


def test_evaluation_ids_are_unique():
    profile = _profile(LLAMA_CONTROL)
    fixture = harness.load_fixture("C04")
    assert harness.run_one(FakeHttp(), fixture, profile)["evaluation_id"] != \
        harness.run_one(FakeHttp(), fixture, profile)["evaluation_id"]


# --- the CLI ---------------------------------------------------------------------


def test_a_live_run_writes_a_new_file_and_never_overwrites(tmp_path):
    out = tmp_path / "run.jsonl"
    http = FakeHttp()
    assert _main(["--ollama-url", "http://x:1", "--out", str(out)], http) == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["record_type"] for r in records] == ["run"] + ["generation"] * 4
    assert records[0]["persists_answers"] is False and records[0]["think_policy"] == "profile"
    assert [(r["fixture"]["id"], r["profile"]["profile_id"]) for r in records[1:]] == [
        ("C04", "llama_control"), ("C04", "gemma_candidate"),
        ("C05", "llama_control"), ("C05", "gemma_candidate"),
    ]
    with pytest.raises(harness.HarnessUsageError):
        _main(["--ollama-url", "http://x:1", "--out", str(out)], http)


def test_smoke_cli_honours_think_false(tmp_path):
    out = tmp_path / "smoke.jsonl"
    http = FakeHttp(_ndjson({"response": "ready"}, _DONE))
    _main(["--ollama-url", "http://x:1", "--out", str(out), "--smoke", "--think", "false"], http)
    bodies = {b["model"]: b for b in http.generate_bodies}
    assert bodies["gemma4:31b"]["think"] is False, "the server reports thinking for Gemma"
    assert "think" not in bodies["llama3.1:8b"], "it does not for Llama, so it is omitted"


@pytest.mark.parametrize(
    "url",
    ["http://user:pass@host:11434", "http://host:11434/api", "ftp://host", "host:11434", ""],
)
def test_only_a_plain_server_root_url_is_accepted(url):
    with pytest.raises(harness.HarnessUsageError):
        harness.validate_base_url(url)


def test_a_live_run_needs_an_explicit_url_and_output():
    with pytest.raises(harness.HarnessUsageError):
        harness.main([], stdout=io.StringIO())


def test_the_default_read_timeout_is_180_seconds_and_reaches_the_transport(tmp_path):
    defaults = harness._parse([])
    assert defaults.read_timeout == 180.0, "a per-read wait above the ~91.8 s cold Gemma load"
    assert defaults.max_seconds == 600.0, "the total deadline is unchanged"

    received = []

    def factory(url, read_timeout):
        received.append(read_timeout)
        return FakeHttp()

    harness.main(["--ollama-url", "http://x:1", "--out", str(tmp_path / "r.jsonl")],
                 http_factory=factory, stdout=io.StringIO(), registry_factory=_synthetic_registry)
    assert received == [180.0]


# --- grading --------------------------------------------------------------------


def test_grading_reports_anchor_hits_after_the_fact_and_ignores_skipped_runs(tmp_path):
    profile = _profile(LLAMA_CONTROL)
    fixture = harness.load_fixture("C05")
    good = harness.run_one(FakeHttp(_ndjson({"response": (
        "IRCoT stops when a CoT sentence says the answer is: or the maximum number of steps "
        "is reached; experiments used 8 steps and 15 paragraphs [C1]."
    )}, _DONE)), fixture, profile)
    weak = harness.run_one(FakeHttp(_ndjson({"response": "It stops eventually."}, _DONE)), fixture, profile)
    skipped = harness.run_one(FakeHttp(digests={}), fixture, profile)
    results = tmp_path / "r.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in (good, weak, skipped)), encoding="utf-8")

    out = io.StringIO()
    harness.main(["--grade", str(results)], stdout=out)
    graded = [json.loads(line) for line in out.getvalue().splitlines()]

    assert len(graded) == 2, "a skipped run has nothing to grade"
    assert graded[0]["all_anchor_groups_hit"] is True and graded[0]["cites_supplied_page"] is True
    assert graded[1]["all_anchor_groups_hit"] is False and graded[1]["cites_supplied_page"] is False


# --- isolation -------------------------------------------------------------------


def test_the_harness_imports_no_database_credential_or_app_module():
    tree = ast.parse(_HARNESS_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    forbidden = {"main", "dbSetup", "psycopg2", "pgvector", "dotenv", "requests", "subprocess", "os"}
    assert not roots & forbidden
    assert roots <= {
        "__future__", "argparse", "datetime", "hashlib", "json", "re", "socket", "sys", "time",
        "urllib", "uuid", "dataclasses", "pathlib", "typing",
        "answer_evidence", "citation_generation", "generation_profiles", "ollama_generation",
    }


def test_the_harness_never_calls_pull_delete_or_unload_endpoints():
    source = _HARNESS_PATH.read_text(encoding="utf-8")
    for endpoint in ("/api/pull", "/api/delete", "/api/copy", "/api/create", "/api/push"):
        assert endpoint not in source
    assert '"keep_alive": 0' not in source and "keep_alive=0" not in source
