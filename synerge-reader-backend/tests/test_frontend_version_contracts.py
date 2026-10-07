"""Source contracts for the frontend version display (CP2A).

Reads GridApp.jsx and VersionBadge.jsx as text; imports and executes nothing.
The component's rendered behaviour (known, unknown and unavailable builds,
accessible text, no internal fields) is proven by
src/components/VersionBadge.test.jsx under Jest.

What these prove: the app chrome mounts the badge exactly once, in the
sidebar, with the app's existing backend URL; the badge asks the server's
/version endpoint through the same REACT_APP_BACKEND_URL convention as every
other call, reads no other build-time variable, and names no model, profile,
or Ollama detail.
"""

from pathlib import Path
import re

import pytest

_FRONTEND_SRC = Path(__file__).resolve().parents[2] / "synerge-reader-frontend" / "src"
_GRIDAPP = _FRONTEND_SRC / "GridApp.jsx"
_BADGE = _FRONTEND_SRC / "components" / "VersionBadge.jsx"


@pytest.fixture(scope="module")
def gridapp():
    return _GRIDAPP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def badge():
    return _BADGE.read_text(encoding="utf-8")


def _without_comments(source):
    source = re.sub(r"/\*[\s\S]*?\*/", "", source)
    return re.sub(r"(^|\s)//.*$", "", source, flags=re.MULTILINE)


def test_the_sidebar_mounts_the_version_badge_once_with_the_app_backend(gridapp):
    assert 'import VersionBadge from "./components/VersionBadge";' in gridapp
    assert gridapp.count("<VersionBadge") == 1
    assert "<VersionBadge backendUrl={BACKEND}" in gridapp
    sidebar = gridapp[gridapp.index("── SIDEBAR"):gridapp.index("── MAIN COLUMN")]
    assert "<VersionBadge" in sidebar, "the version lives in the sidebar chrome"


def test_the_badge_uses_the_existing_backend_url_convention_and_only_that(badge):
    assert 'process.env.REACT_APP_BACKEND_URL || "http://localhost:5000"' in badge
    assert set(re.findall(r"process\.env\.([A-Z0-9_]+)", badge)) == {"REACT_APP_BACKEND_URL"}, (
        "no frontend build-time version, commit, or Git variable may stand in for the server's answer"
    )
    assert "fetch(`${backendUrl}/version`" in badge


def test_the_badge_names_no_model_or_server_internals(badge):
    code = _without_comments(badge).lower()
    for internal in ("model", "profile", "ollama", "llama", "gemma", "qwen", "digest", "/admin", "token"):
        assert internal not in code, internal
