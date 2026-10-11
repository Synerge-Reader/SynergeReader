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
from generation_profiles import GEMMA_CANDIDATE, LLAMA_CONTROL, ModelRegistry, Qualification, default_model_registry

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
    base = default_model_registry()
    return ModelRegistry(
        profiles=tuple(_synthetic_profile(p) for p in base.profiles),
        default_profile_id=LLAMA_CONTROL,
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
    # C06 is a frozen fixture since the two-source format; C07 has no file.
    for bad in ("C07", "../c04", "c04"):
        with pytest.raises(harness.FixtureError):
            harness.load_fixture(bad)


# --- newly prepared manual-evidence fixtures (C01, C02, C03, C06) ------------------

# (fixture_sha256, evidence_sha256, prompt_sha256), pinned after the excerpts were
# extracted from the public papers and checked against the rendered pages.
_NEW_PINS = {
    "C01": ("854a86bf9469b7b6cdfdae3088191355259a44f2b7779289c6edcbddb2c8cbe1",
            "7803ed6a5cacc7692b48ce3ad6b71b4a5c98ce0dd2e20391233568950c5220b1",
            "37a7901e17b4e806dfabea739fb54fd0b80480c32b28d1b756ba068f377aff49"),
    "C02": ("e940fedf92b02c02089281bd13af1e92a9f9c5082529bfe73366c82076681a87",
            "720f3f61ad85d089e98c2974c61150f32b3114e5aed10e6a23c8ca104f23d3c2",
            "52be30a58be816974daebeab76bb01d8b306506c1acb8a8902faa35972881daf"),
    "C03": ("f7fffb6c7165aa205bf3724d5314f71ab557c38f40e92e6622145c5f9a40d3bf",
            "0be557ab22ada5b610402f8259bf80c5dcf08e8effcb45894b34c5274f849766",
            "a858371fec22672380796728481d137ec70d994000dc2dab5750772fd49570e5"),
    "C06": ("9782405630a428bad072d444262351fb19dbd2481d65d97b0ef30c4a3ce05263",
            "a4b18b68a4a5c2706f44bd8836e2752119e00bd2f55678a8cb70620adf4aeb99",
            "d231108846f4bc10b7d2bb2410dbf0139cbe348824118a889546aafe90518a55"),
}
_NEW_QUESTIONS = {
    "C01": "What improvements did Step-Back Prompting report for PaLM-2L on MMLU Physics, MMLU Chemistry, TimeQA, "
           "and MuSiQue?",
    "C02": "What BLEU score did the big Transformer achieve on WMT 2014 English-to-German, and how long did its "
           "training take?",
    "C03": "What are the two stages of Step-Back Prompting, and what does each stage do?",
    "C06": "How do IRCoT and Step-Back Prompting differ in the way they improve multi-step reasoning?",
}
_STEP_BACK_PDF = "5133e42bd76de3b31be5fd21a3d36ebb45bfdf58db00047839175829f214eb72"
_ATTENTION_PDF = "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"


def test_new_fixtures_are_frozen_with_the_exact_questions_and_verified_public_sources():
    expected_sources = {
        "C01": [("Step back.pdf", 1, _STEP_BACK_PDF)],
        "C02": [("Attention is all you need.pdf", 8, _ATTENTION_PDF)],
        "C03": [("Step back.pdf", 3, _STEP_BACK_PDF)],
        "C06": [("IR-CoT.pdf", 4, None), ("Step back.pdf", 3, _STEP_BACK_PDF)],
    }
    for fixture_id, (fixture_sha, evidence_sha, prompt_sha) in _NEW_PINS.items():
        fixture = harness.load_fixture(fixture_id)
        assert fixture.question == _NEW_QUESTIONS[fixture_id] and fixture.revision == 1
        assert (fixture.fixture_sha256, fixture.evidence_sha256) == (fixture_sha, evidence_sha)
        assert harness.sha256_hex(harness.build_prompt(fixture)) == prompt_sha
        sources = [(s["filename"], s["page_start"], pdf) for s, (_, _, pdf) in
                   zip(fixture.sources, expected_sources[fixture_id])]
        assert sources == expected_sources[fixture_id]
        for source, (_, _, pdf) in zip(fixture.sources, expected_sources[fixture_id]):
            assert source["page_start"] == source["page_end"], "each excerpt comes from one physical page"
            if pdf is not None:
                assert pdf in source["provenance_note"] and "physical page" in source["provenance_note"]
                assert "whitespace only" in source["excerpt_scope_note"]
    assert "7% and 11% respectively, TimeQA by 27%, and MuSiQue by 7%" in harness.load_fixture("C01").evidence["text"]
    c02 = harness.load_fixture("C02").evidence["text"]
    assert "BLEU score of 28.4" in c02 and "Training took 3.5 days on 8 P100 GPUs." in c02
    c03 = harness.load_fixture("C03").evidence["text"]
    assert "• Abstraction:" in c03 and "• Reasoning:" in c03 and c03.count("\n") == 2


def test_a_two_source_fixture_keeps_citation_order_locators_and_its_hash_formula():
    fixture = harness.load_fixture("C06")
    assert fixture.evidence is None and len(fixture.sources) == 2
    ircot = harness.load_fixture("C04").evidence
    assert fixture.sources[0] == ircot, "C1 is the existing IRCoT page-4 excerpt, unchanged"
    assert fixture.sources[1] == harness.load_fixture("C03").evidence, "C2 is the Step-Back stages excerpt"
    assert fixture.source_text_sha256s == (_EVIDENCE_SHA, _NEW_PINS["C03"][1])
    assert fixture.evidence_sha256 == harness.sha256_hex(harness._canonical(list(fixture.source_text_sha256s)))
    spec = json.loads((harness.FIXTURE_DIR / "c06.json").read_text(encoding="utf-8"))
    assert fixture.fixture_sha256 == harness.sha256_hex(
        harness._canonical({"fixture": spec, "evidence": list(fixture.sources)}))

    prompt = harness.build_prompt(fixture)
    first = prompt.index("[C1]\nDocument: IR-CoT.pdf\nLocation: page 4\nEvidence:\n" + ircot["text"])
    second = prompt.index("[C2]\nDocument: Step back.pdf\nLocation: page 3\nEvidence:\n")
    assert first < second and "[C3]" not in prompt
    items = harness.evidence_items(fixture)
    assert [item.chunk_id for item in items] == ["diagnostic:ircot-page4-normalized-v1",
                                                 "diagnostic:stepback-c03-page3-v1"]


def test_the_single_source_shape_and_hash_formula_are_unchanged():
    fixture = harness.load_fixture("C04")
    spec = json.loads((harness.FIXTURE_DIR / "c04.json").read_text(encoding="utf-8"))
    evidence = json.loads((harness.FIXTURE_DIR / "ircot_page4_evidence.json").read_text(encoding="utf-8"))
    assert fixture.fixture_sha256 == harness.sha256_hex(harness._canonical({"fixture": spec, "evidence": evidence}))
    assert fixture.evidence == evidence and fixture.sources == (evidence,)
    assert fixture.evidence_sha256 == harness.sha256_hex(evidence["text"]) == _EVIDENCE_SHA


@pytest.mark.parametrize("change", ["tampered-second-source", "pins-reordered", "one-file-only",
                                    "duplicate-file", "outside-directory", "extra-key"])
def test_a_two_source_fixture_is_refused_when_not_frozen(tmp_path, change):
    copy = tmp_path / "fixtures"
    shutil.copytree(harness.FIXTURE_DIR, copy)
    spec_path = copy / "c06.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if change == "tampered-second-source":
        path = copy / "stepback_c03_evidence.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["text"] = evidence["text"].replace("original question", "first question")
        path.write_text(json.dumps(evidence), encoding="utf-8")
    elif change == "pins-reordered":
        spec["evidence_text_sha256s"].reverse()
    elif change == "one-file-only":
        spec["evidence_files"], spec["evidence_text_sha256s"] = spec["evidence_files"][:1], spec["evidence_text_sha256s"][:1]
    elif change == "duplicate-file":
        spec["evidence_files"][1] = spec["evidence_files"][0]
        spec["evidence_text_sha256s"][1] = spec["evidence_text_sha256s"][0]
    elif change == "outside-directory":
        spec["evidence_files"][1] = "../stepback_c03_evidence.json"
    else:
        spec["evidence_file"] = "ircot_page4_evidence.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(harness.FixtureError):
        harness.load_fixture("C06", copy)


def test_two_source_hashes_do_not_depend_on_line_endings(tmp_path):
    copy = tmp_path / "fixtures"
    shutil.copytree(harness.FIXTURE_DIR, copy)
    for path in copy.glob("*.json"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    for fixture_id, (fixture_sha, evidence_sha, _) in _NEW_PINS.items():
        loaded = harness.load_fixture(fixture_id, copy)
        assert (loaded.fixture_sha256, loaded.evidence_sha256) == (fixture_sha, evidence_sha)


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


def _generated(fixture_id, text):
    return {"record_type": "generation", "generated": True, "evaluation_id": f"{fixture_id}-1",
            "fixture": {"id": fixture_id}, "profile": {"profile_id": GEMMA_CANDIDATE},
            "response": {"text": text}, "stream": {"failure": None}, "error": None}


def test_a_two_source_answer_is_graded_for_each_citation_id_separately():
    anchors = harness.load_anchors()
    both = harness.grade_record(_generated("C06", (
        "IRCoT interleaves retrieval with chain-of-thought steps [C1]. Step-Back abstracts to a high-level "
        "principle first, then reasons from it [C2].")), anchors)
    only_first = harness.grade_record(_generated("C06", (
        "IRCoT interleaves retrieval with chain-of-thought steps, while Step-Back abstracts and then reasons [C1].")),
        anchors)
    assert both["citation_ids_present"] == {"C1": True, "C2": True} and both["cites_supplied_page"] is True
    assert both["all_anchor_groups_hit"] is True
    assert only_first["citation_ids_present"] == {"C1": True, "C2": False}
    assert only_first["cites_supplied_page"] is False, "every declared source must be cited"


def test_single_source_grading_output_is_unchanged():
    graded = harness.grade_record(_generated("C04", "IRCoT alternates CoT generation and retrieval [C1]."),
                                  harness.load_anchors())
    assert set(graded) == {"evaluation_id", "fixture", "profile_id", "anchor_groups", "all_anchor_groups_hit",
                           "cites_supplied_page", "stream_failure", "error", "note"}
    assert graded["cites_supplied_page"] is True


@pytest.mark.parametrize("fixture_id, text, hits", [
    ("C01", "PaLM-2L gains 7% on MMLU Physics, 11% on MMLU Chemistry, 27% on TimeQA and 7% on MuSiQue [C1].",
     [True, True, True, True]),
    ("C01", "PaLM-2L gains 7% and 11% on Physics. Chemistry is not stated [C1].", [True, False, False, False]),
    ("C02", "It reached 28.4 BLEU and training took 3.5 days on 8 P100 GPUs [C1].", [True, True]),
    ("C03", "Abstraction asks a step-back question about a higher-level principle; Reasoning then answers the "
            "original question grounded on it [C1].", [True, True]),
])
def test_the_new_anchors_are_heuristic_hits_on_answer_text_only(fixture_id, text, hits):
    graded = harness.grade_record(_generated(fixture_id, text), harness.load_anchors())
    assert [group["hit"] for group in graded["anchor_groups"]] == hits


def test_gold_anchor_files_are_merged_and_a_duplicate_fixture_is_refused(tmp_path):
    merged = harness.load_anchors()
    assert sorted(merged["fixtures"]) == ["C01", "C02", "C03", "C04", "C05", "C06"]
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(harness.GRADING_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(harness.FixtureError):
        harness.load_anchors((harness.GRADING_PATH, duplicate))


def test_no_gold_anchor_data_reaches_any_fixture_request():
    # The facts themselves naturally appear in the supplied evidence; what must
    # never appear is anything taken from the grading files.
    anchors = harness.load_anchors()
    gold = [fact for spec in anchors["fixtures"].values() for fact in spec["required_facts"]]
    gold += [fact for spec in anchors["fixtures"].values() for fact in spec.get("supplementary_facts", [])]
    gold += [group["fact"] for spec in anchors["fixtures"].values() for group in spec["anchor_groups"]]
    for fixture_id in ("C01", "C02", "C03", "C04", "C05", "C06"):
        prompt = harness.build_prompt(harness.load_fixture(fixture_id))
        for item in gold + ["Expected:", "anchor_groups", "required_facts", "expected_citation"]:
            assert item not in prompt


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
