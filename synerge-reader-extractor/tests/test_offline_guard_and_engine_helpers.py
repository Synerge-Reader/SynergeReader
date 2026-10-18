import socket
import sys
import types

import pytest

from extractor_service import offline_guard
from extractor_service.engine import _clean_text


def test_clean_text_collapses_whitespace_and_drops_blank_lines():
    assert _clean_text(None) == ""
    assert _clean_text("  Invoice   No.\n\n\t Total  $1.00 \n") == "Invoice No.\nTotal $1.00"


def test_download_helpers_are_replaced_with_fail_closed_stubs(monkeypatch):
    fake = types.ModuleType("fake_hub_for_guard_test")
    fake.snapshot_download = lambda *a, **k: "downloaded"
    fake.unrelated = lambda: "kept"
    monkeypatch.setitem(sys.modules, "fake_hub_for_guard_test", fake)
    replaced = offline_guard.deny_download_helpers()
    assert "fake_hub_for_guard_test.snapshot_download" in replaced
    with pytest.raises(offline_guard.DownloadDenied):
        fake.snapshot_download("repo")
    assert fake.unrelated() == "kept"


def test_network_deny_blocks_ip_sockets_and_dns(monkeypatch):
    monkeypatch.setattr(offline_guard, "_installed", False)
    for name in ("getaddrinfo", "gethostbyname", "create_connection"):
        monkeypatch.setattr(socket, name, getattr(socket, name))
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    offline_guard.install_network_deny()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(offline_guard.NetworkDenied):
            s.connect(("127.0.0.1", 9))
        with pytest.raises(offline_guard.NetworkDenied):
            s.connect_ex(("127.0.0.1", 9))
    finally:
        s.close()
    with pytest.raises(offline_guard.NetworkDenied):
        socket.getaddrinfo("example.org", 443)
