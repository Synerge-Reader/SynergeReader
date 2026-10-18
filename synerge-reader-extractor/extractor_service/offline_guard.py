"""Defense-in-depth for the engine process: no outbound IP connections, no model downloads.

The container already runs on an internal-only network with HF/Transformers offline
flags. This guard additionally refuses AF_INET/AF_INET6 connections and name resolution
inside the engine process (AF_UNIX stays available for multiprocessing), and replaces
known model-download helpers with fail-closed stubs after the libraries are imported.
"""

from __future__ import annotations

import socket
import sys

_installed = False


class NetworkDenied(OSError):
    pass


class DownloadDenied(RuntimeError):
    pass


def install_network_deny() -> None:
    global _installed
    if _installed:
        return
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkDenied("outbound network access is disabled in the extraction engine")
        return orig_connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkDenied("outbound network access is disabled in the extraction engine")
        return orig_connect_ex(self, address)

    def no_dns(*args, **kwargs):
        raise NetworkDenied("name resolution is disabled in the extraction engine")

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = no_dns
    socket.gethostbyname = no_dns
    socket.create_connection = no_dns
    _installed = True


_HELPERS = ("snapshot_download", "hf_hub_download", "download_models", "download_hf_model")


def deny_download_helpers() -> list[str]:
    replaced = []

    def stub(*args, **kwargs):
        raise DownloadDenied("model downloads are disabled; use the staged local models")

    for modname, mod in list(sys.modules.items()):
        if mod is None:
            continue
        for name in _HELPERS:
            fn = getattr(mod, name, None)
            if callable(fn) and getattr(fn, "__name__", "") != "stub":
                try:
                    setattr(mod, name, stub)
                    replaced.append(f"{modname}.{name}")
                except Exception:
                    pass
    return replaced
