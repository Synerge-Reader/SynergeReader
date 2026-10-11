"""Authoritative application release identity for SynergeReader.

This module is a pure configuration/value-object layer, like
``rag_model_profiles.py``: nothing here touches the network, the filesystem,
a database, or a subprocess, and the environment is read only when the
composition boundary (``main.py``) asks for it, at call time.

What it decides:

* the application's semantic version (``APP_VERSION``);
* the HTTP/OpenAPI contract version (``API_CONTRACT_VERSION``), a separate
  identity that changes independently of the application release;
* the versions of the two independently maintained prompts an answer depends
  on -- the answer prompt (``citation_generation.py``) and the claim-support
  verifier prompt (``main.py``). Each is pinned to a fingerprint of its
  template text in ``tests/test_release_metadata.py``, so a template edit that
  does not bump its version fails a test instead of mislabelling answers;
* the deployed build's Git commit, which only the release operator knows.

How the build commit is known: the operator injects the exact reviewed commit
as ``SYNERGE_BUILD_SHA`` when building/deploying. The running application
never runs Git and never looks at its own source directory to guess -- a
working tree can be dirty, a different checkout, or not a repository at all.
A missing or malformed value is reported as unknown (``None``), never replaced
by anything else.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

# The running application release: what /version and answer provenance report.
APP_VERSION = "0.1.0"

# The HTTP/OpenAPI contract version (FastAPI's ``version=``). It describes the
# API's shape, not the release, and changes independently of APP_VERSION.
API_CONTRACT_VERSION = "2.0.0"

# Versions of the prompt templates an answer is generated and graded with.
ANSWER_PROMPT_VERSION = "rag-answer-v1"
CLAIM_SUPPORT_PROMPT_VERSION = "claim-support-v1"

BUILD_SHA_ENV = "SYNERGE_BUILD_SHA"

# This repository uses SHA-1 object names (``git rev-parse --show-object-format``
# reports sha1): exactly 40 hex digits. Abbreviated, prefixed, or SHA-256
# names are not accepted -- the operator must inject the full commit.
_FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SEMVER_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_PROMPT_VERSION_PATTERN = re.compile(r"^[a-z][a-z0-9-]*-v[1-9][0-9]*$")

# Matches ``git log --oneline`` / GitHub's default abbreviation, so the visible
# short form is what an operator sees next to the commit.
SHORT_SHA_LENGTH = 7

_DISPLAY_SEPARATOR = " · "
_UNKNOWN_BUILD_LABEL = "build unknown"


class ReleaseMetadataError(ValueError):
    """A release-metadata value object was constructed with invalid data."""


def normalize_build_sha(value: object) -> Optional[str]:
    """A full 40-hex commit SHA in lowercase, or None.

    Surrounding whitespace is ignored (``git rev-parse HEAD > file`` leaves a
    newline). Anything else that is not exactly 40 hex digits -- empty, short,
    prefixed, non-hex, or not a string -- is unknown, not an error: an
    unidentified build must still start and serve, and must say so.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if _FULL_SHA_PATTERN.match(candidate) else None


@dataclass(frozen=True)
class ReleaseMetadata:
    app_version: str
    build_sha: Optional[str]
    answer_prompt_version: str = ANSWER_PROMPT_VERSION
    claim_support_prompt_version: str = CLAIM_SUPPORT_PROMPT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.app_version, str) or not _SEMVER_PATTERN.match(self.app_version):
            raise ReleaseMetadataError("app_version must be a MAJOR.MINOR.PATCH version")
        if self.build_sha is not None and normalize_build_sha(self.build_sha) != self.build_sha:
            raise ReleaseMetadataError("build_sha must be None or a normalized 40-hex commit")
        for name in ("answer_prompt_version", "claim_support_prompt_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _PROMPT_VERSION_PATTERN.match(value):
                raise ReleaseMetadataError(f"{name} must look like 'name-vN'")

    @property
    def build_known(self) -> bool:
        return self.build_sha is not None

    @property
    def build_sha_short(self) -> Optional[str]:
        return self.build_sha[:SHORT_SHA_LENGTH] if self.build_sha else None

    @property
    def display_version(self) -> str:
        build = self.build_sha_short or _UNKNOWN_BUILD_LABEL
        return f"v{self.app_version}{_DISPLAY_SEPARATOR}{build}"

    def public_version(self) -> dict:
        """The whole public /version body. Nothing else is ever exposed."""
        return {
            "app_version": self.app_version,
            "build_sha": self.build_sha,
            "build_sha_short": self.build_sha_short,
            "build_known": self.build_known,
            "display_version": self.display_version,
        }


def release_metadata_from_env(environ: Optional[Mapping[str, str]] = None) -> ReleaseMetadata:
    """This process's release identity, read from ``SYNERGE_BUILD_SHA`` only.

    ``environ`` defaults to ``os.environ`` at call time. Only the one key is
    read; no other environment value can reach the result.
    """
    if environ is None:
        environ = os.environ
    return ReleaseMetadata(
        app_version=APP_VERSION,
        build_sha=normalize_build_sha(environ.get(BUILD_SHA_ENV)),
    )
