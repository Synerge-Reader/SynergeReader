import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import embedding_config


@pytest.fixture(autouse=True)
def _reload_clean_config():
    """Reset embedding_config to real-environment values before and after
    each test, so a test's monkeypatched env vars never leak into the next
    test (monkeypatch reverts os.environ in its own teardown, which runs
    before this fixture's post-yield reload)."""
    importlib.reload(embedding_config)
    yield
    importlib.reload(embedding_config)


def test_reads_real_env_values_after_reload(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "some-other-model:test")
    monkeypatch.setenv("EMBEDDING_VECTOR_DIMENSION", "999")

    reloaded = importlib.reload(embedding_config)

    assert reloaded.EMBED_MODEL == "some-other-model:test"
    assert reloaded.EMBEDDING_VECTOR_DIMENSION == 999


def test_empty_embed_model_raises_at_reload_time(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "   ")

    with pytest.raises(RuntimeError):
        importlib.reload(embedding_config)


@pytest.mark.parametrize("bad_value", ["0", "-5", "abc", "3.5"])
def test_invalid_embedding_dimension_raises_at_reload_time(monkeypatch, bad_value):
    monkeypatch.setenv("EMBEDDING_VECTOR_DIMENSION", bad_value)

    with pytest.raises(RuntimeError):
        importlib.reload(embedding_config)
