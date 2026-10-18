"""Importable child-process targets for the supervisor tests (spawn needs module-level functions)."""

import time


def echo_worker(conn, cfg):
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job.get("mode") == "sleep":
            time.sleep(job["seconds"])
        if job.get("mode") == "die":
            raise SystemExit(3)
        if job.get("mode") == "fail":
            conn.send({"ok": False, "kind": "input", "message": "the PDF could not be opened"})
            continue
        conn.send({"ok": True, "result": {"echo": job.get("value"), "cfg": cfg}})
