"""Contracts for the authoritative release identity (release_metadata.py).

Pure: nothing here imports ``main``, opens a connection, runs a subprocess, or
reads the repository's Git state. Every build SHA below is a synthetic example
injected by the test, never the deployed or current commit.

What these prove: the application version is 0.1.0; a full commit SHA is
accepted only in the 40-hex form this repository uses and is normalized to
lowercase; a missing or malformed SHA becomes "unknown" rather than anything
guessed; only SYNERGE_BUILD_SHA is read; the public version body is exactly
five fields; the module neither runs Git nor hardcodes a commit; and each of
the two independently maintained prompt templates is pinned to its version by
a fingerprint, so editing a template without bumping its version fails here.
"""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from answer_evidence import EvidenceBundle, EvidenceItem, EvidenceMode
from citation_generation import AnswerMode, CitationRegistry, build_generation_prompt, build_mode_prompt
import release_metadata
from release_metadata import (
    ANSWER_PROMPT_VERSION,
    API_CONTRACT_VERSION,
    APP_VERSION,
    BUILD_SHA_ENV,
    CLAIM_SUPPORT_PROMPT_VERSION,
    ReleaseMetadata,
    ReleaseMetadataError,
    normalize_build_sha,
    release_metadata_from_env,
)

_BACKEND = Path(__file__).resolve().parents[1]
_RELEASE_METADATA_PATH = _BACKEND / "release_metadata.py"
_MAIN_PATH = _BACKEND / "main.py"

SYNTHETIC_SHA = "0123456789abcdef0123456789abcdef01234567"
SYNTHETIC_SHA_UPPER = "0123456789ABCDEF0123456789ABCDEF01234567"


# --- versions -----------------------------------------------------------------


def test_the_initial_application_and_prompt_versions():
    assert APP_VERSION == "0.1.0"
    assert ANSWER_PROMPT_VERSION == "rag-answer-v1"
    assert CLAIM_SUPPORT_PROMPT_VERSION == "claim-support-v1"
    assert BUILD_SHA_ENV == "SYNERGE_BUILD_SHA"


def test_the_api_contract_version_is_separate_from_the_application_version():
    assert API_CONTRACT_VERSION == "2.0.0"
    assert APP_VERSION == "0.1.0"
    assert API_CONTRACT_VERSION != APP_VERSION


# --- build SHA normalization --------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [SYNTHETIC_SHA, SYNTHETIC_SHA_UPPER, f"  {SYNTHETIC_SHA}\n", f"\t{SYNTHETIC_SHA_UPPER}\r\n"],
    ids=["lowercase", "uppercase", "trailing-newline", "tabs-and-crlf"],
)
def test_a_full_sha_is_accepted_and_normalized_to_lowercase(raw):
    assert normalize_build_sha(raw) == SYNTHETIC_SHA


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "unknown",
        SYNTHETIC_SHA[:7],                 # abbreviated
        SYNTHETIC_SHA[:39],                # one short
        SYNTHETIC_SHA + "8",               # one long
        "g" + SYNTHETIC_SHA[1:],           # not hex
        f"sha1:{SYNTHETIC_SHA}",           # prefixed
        "ab" * 32,                         # a SHA-256 object name; this repo is SHA-1
        f"{SYNTHETIC_SHA[:20]} {SYNTHETIC_SHA[20:]}",
        12345,
        SYNTHETIC_SHA.encode(),
    ],
    ids=[
        "none", "empty", "blank", "word", "short", "39-chars", "41-chars", "non-hex",
        "prefixed", "sha256-length", "inner-space", "int", "bytes",
    ],
)
def test_a_missing_or_malformed_sha_is_unknown(raw):
    assert normalize_build_sha(raw) is None


def test_a_valid_sha_from_the_environment_is_known():
    release = release_metadata_from_env({BUILD_SHA_ENV: SYNTHETIC_SHA_UPPER + "\n"})

    assert release.app_version == "0.1.0"
    assert release.build_sha == SYNTHETIC_SHA
    assert release.build_sha_short == "0123456"
    assert release.build_known is True
    assert release.display_version == "v0.1.0 · 0123456"


@pytest.mark.parametrize("environ", [{}, {BUILD_SHA_ENV: ""}, {BUILD_SHA_ENV: "c236c6e"}])
def test_a_missing_or_malformed_environment_sha_is_unknown_not_guessed(environ):
    release = release_metadata_from_env(environ)

    assert release.build_sha is None
    assert release.build_sha_short is None
    assert release.build_known is False
    assert release.display_version == "v0.1.0 · build unknown"


def test_only_synerge_build_sha_is_read():
    """Generic CI variables never stand in for the operator-injected value."""
    release = release_metadata_from_env(
        {
            "GITHUB_SHA": SYNTHETIC_SHA,
            "GIT_COMMIT": SYNTHETIC_SHA,
            "SOURCE_VERSION": SYNTHETIC_SHA,
            "BUILD_SHA": SYNTHETIC_SHA,
        }
    )
    assert release.build_sha is None


def test_the_default_environment_is_read_at_call_time(monkeypatch):
    monkeypatch.delenv(BUILD_SHA_ENV, raising=False)
    assert release_metadata_from_env().build_known is False
    monkeypatch.setenv(BUILD_SHA_ENV, SYNTHETIC_SHA)
    assert release_metadata_from_env().build_sha == SYNTHETIC_SHA


# --- the value object ---------------------------------------------------------


def test_the_public_version_body_is_exactly_five_fields():
    known = release_metadata_from_env({BUILD_SHA_ENV: SYNTHETIC_SHA}).public_version()
    unknown = release_metadata_from_env({}).public_version()

    assert known == {
        "app_version": "0.1.0",
        "build_sha": SYNTHETIC_SHA,
        "build_sha_short": "0123456",
        "build_known": True,
        "display_version": "v0.1.0 · 0123456",
    }
    assert unknown == {
        "app_version": "0.1.0",
        "build_sha": None,
        "build_sha_short": None,
        "build_known": False,
        "display_version": "v0.1.0 · build unknown",
    }
    # Prompt versions are provenance, not part of the public version body.
    assert "rag-answer" not in json.dumps(known) and "claim-support" not in json.dumps(known)


@pytest.mark.parametrize(
    "fields",
    [
        dict(app_version="0.1", build_sha=None),
        dict(app_version="v0.1.0", build_sha=None),
        dict(app_version="0.1.0", build_sha=SYNTHETIC_SHA_UPPER),
        dict(app_version="0.1.0", build_sha=SYNTHETIC_SHA[:7]),
        dict(app_version="0.1.0", build_sha=None, answer_prompt_version="rag answer"),
        dict(app_version="0.1.0", build_sha=None, claim_support_prompt_version=""),
    ],
)
def test_release_metadata_refuses_unnormalized_values(fields):
    with pytest.raises(ReleaseMetadataError):
        ReleaseMetadata(**fields)


def test_release_metadata_is_immutable():
    release = release_metadata_from_env({BUILD_SHA_ENV: SYNTHETIC_SHA})
    with pytest.raises(Exception):
        release.build_sha = "f" * 40


# --- never Git, never a hardcoded commit --------------------------------------


def _imported_roots(path):
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_release_metadata_never_runs_git_or_reads_the_source_tree():
    roots = _imported_roots(_RELEASE_METADATA_PATH)
    assert roots.isdisjoint({"subprocess", "pathlib", "glob", "shutil", "git", "pygit2", "dulwich"})
    forbidden = {"open", "popen", "system", "run", "call", "check_output", "Popen",
                 "read_text", "read_bytes", "listdir", "scandir", "walk", "exec", "eval"}
    called = set()
    for node in ast.walk(ast.parse(_RELEASE_METADATA_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            called.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", None))
    assert called.isdisjoint(forbidden), called & forbidden


def test_main_never_runs_git_to_learn_its_build():
    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"))
    assert "subprocess" not in {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "rev-parse" not in node.value and "git log" not in node.value


@pytest.mark.parametrize("path", [_RELEASE_METADATA_PATH, _MAIN_PATH], ids=["release_metadata", "main"])
def test_no_commit_sha_is_hardcoded_as_the_release(path):
    full_sha = re.compile(r"^[0-9a-fA-F]{40}$")
    literals = [
        node.value
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not [value for value in literals if full_sha.match(value.strip())]


# --- prompt-template fingerprints ---------------------------------------------
#
# A prompt version is only honest if it changes when the prompt does. These
# fingerprints are SHA-256 digests of the server-side templates. If one of the
# assertions below fails, a template changed: bump the matching version in
# release_metadata.py (e.g. rag-answer-v2) and update the fingerprint with it.
# Text the client puts in the question (such as the frontend's task-mode
# prefixes) is request input, not part of these server templates.

_ANSWER_PROMPT_FINGERPRINTS = {
    "rag-answer-v1": "886c03093bef6c618d200f06890e61eb31c90da09b43c17416f7d61dda25d72c",
}
_CLAIM_SUPPORT_PROMPT_FINGERPRINTS = {
    "claim-support-v1": "d0c5903c85485a4c6f800cd08f118682f24fe32c242b2e89fcb056ccce1910da",
}


def _answer_prompt_renderings():
    """Every answer-prompt shape /ask can build: each answer mode over each
    evidence mode, including no evidence at all."""
    question = "What is the notice period?"
    items = {
        EvidenceMode.SELECTED_TEXT: EvidenceItem(
            text="Selected: the notice period is thirty days.", source_type="selection",
            document_id=1, filename="lease.pdf", page_start=2, page_end=2, selection_id="s1",
        ),
        EvidenceMode.COMPLETE_DOCUMENT: EvidenceItem(
            text="Complete: the notice period is thirty days.", source_type="complete_document",
            document_id=1, filename="lease.pdf",
        ),
        EvidenceMode.HYBRID_RETRIEVAL: EvidenceItem(
            text="Chunk: the notice period is thirty days.", source_type="document_chunk",
            document_id=1, filename="lease.pdf", chunk_id="1-0", chunk_index=0, page_start=3, page_end=3,
        ),
    }
    registries = [
        CitationRegistry.from_bundle(EvidenceBundle(mode=mode, items=(item,)))
        for mode, item in items.items()
    ]
    registries.append(CitationRegistry.from_bundle(EvidenceBundle(mode=EvidenceMode.NONE)))
    renderings = []
    for registry in registries:
        renderings.append(["document_qa", build_generation_prompt(question, registry)])
        for mode in (AnswerMode.STRUCTURED_JSON, AnswerMode.MODEL_REASONING):
            renderings.append([mode.value, build_mode_prompt(mode, question, registry)])
    return renderings


def _claim_verifier_template():
    tree = ast.parse(_MAIN_PATH.read_text(encoding="utf-8"))
    [node] = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_CLAIM_VERIFIER_PROMPT"
    ]
    return ast.literal_eval(node.value)


def test_the_answer_prompt_version_matches_its_template():
    renderings = _answer_prompt_renderings()
    canonical = json.dumps(renderings, ensure_ascii=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert ANSWER_PROMPT_VERSION in _ANSWER_PROMPT_FINGERPRINTS, "record the new version's fingerprint"
    assert fingerprint == _ANSWER_PROMPT_FINGERPRINTS[ANSWER_PROMPT_VERSION], (
        "the answer prompt template changed: bump ANSWER_PROMPT_VERSION and record its fingerprint"
    )


def test_the_claim_support_prompt_version_matches_its_template():
    fingerprint = hashlib.sha256(_claim_verifier_template().encode("utf-8")).hexdigest()
    assert CLAIM_SUPPORT_PROMPT_VERSION in _CLAIM_SUPPORT_PROMPT_FINGERPRINTS
    assert fingerprint == _CLAIM_SUPPORT_PROMPT_FINGERPRINTS[CLAIM_SUPPORT_PROMPT_VERSION], (
        "the claim-verifier prompt changed: bump CLAIM_SUPPORT_PROMPT_VERSION and record its fingerprint"
    )


def test_the_two_prompts_are_versioned_independently():
    assert ANSWER_PROMPT_VERSION != CLAIM_SUPPORT_PROMPT_VERSION
    verifier = _claim_verifier_template()
    assert all(verifier not in prompt for _, prompt in _answer_prompt_renderings())
