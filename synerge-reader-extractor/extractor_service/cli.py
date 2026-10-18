"""Run one extraction job from the command line and print the raw v1 JSON (acceptance evidence).

    docker run --rm -i --network none <image> \
        python -m extractor_service.cli --ocr-pages 1 --table-pages 4 --timeout-s 900 < input.pdf > raw.json

Reads the PDF from stdin, so no host directory is mounted. Uses the same supervised engine,
limits and page-list parser as the HTTP service; the token is not needed.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import SERVICE_SCHEMA
from .app import BadRequest, parse_page_spec
from .config import ServiceSettings
from .engine import worker_main
from .supervisor import JobFailed, Supervisor, WorkerCrashed, WorkerMemoryExceeded, WorkerTimeout


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="extractor_service.cli")
    parser.add_argument("--ocr-pages", default="")
    parser.add_argument("--table-pages", default="")
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args(argv)
    settings = ServiceSettings.from_env()
    try:
        ocr = parse_page_spec(args.ocr_pages, settings.max_ocr_pages)
        table = parse_page_spec(args.table_pages, settings.max_table_pages)
    except BadRequest as exc:
        print(json.dumps({"error": exc.message}), file=sys.stderr)
        return 2
    if not ocr and not table or set(ocr) & set(table):
        print(json.dumps({"error": "request OCR and/or table pages, without overlap"}), file=sys.stderr)
        return 2
    pdf = sys.stdin.buffer.read(settings.max_body_bytes + 1)
    if len(pdf) > settings.max_body_bytes or not pdf.startswith(b"%PDF-"):
        print(json.dumps({"error": "stdin must be a PDF within the size limit"}), file=sys.stderr)
        return 2
    sup = Supervisor(target=worker_main, engine_config=settings.engine_config(),
                     rss_limit_bytes=settings.worker_rss_limit_bytes)
    try:
        result, stats = sup.run({"pdf_bytes": pdf, "ocr_pages": ocr, "table_pages": table},
                                float(min(args.timeout_s, settings.max_job_timeout_s)))
    except (WorkerTimeout, WorkerMemoryExceeded, WorkerCrashed, JobFailed) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)[:300]}), file=sys.stderr)
        return 1
    finally:
        sup.shutdown()
    json.dump({"schema": SERVICE_SCHEMA, "pages": result["pages"], "engine": result.get("engine", {}),
               "stats": {"elapsed_s": round(stats.elapsed_s, 3), "peak_rss_bytes": stats.peak_rss_bytes,
                         "timings": result.get("timings", {})}}, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
