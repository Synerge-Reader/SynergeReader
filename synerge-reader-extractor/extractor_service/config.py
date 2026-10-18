"""Worker service settings from the container environment. The token is never logged."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return min(max(int(raw), low), high)
    except ValueError:
        return default


@dataclass(frozen=True)
class ServiceSettings:
    token: str = field(default="", repr=False)
    max_body_bytes: int = 50 * 1024 * 1024
    max_pdf_pages: int = 150
    max_ocr_pages: int = 40
    max_table_pages: int = 10
    max_job_timeout_s: int = 1200
    max_waiting: int = 2
    worker_rss_limit_bytes: int = 6 * 1024 ** 3
    ocr_dpi: int = 300
    model_dir: str = "/models"
    tessdata_dir: str = "/models/tessdata"
    torch_threads: int = 2
    work_dir: str = "/tmp/extractor-jobs"

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            token=os.environ.get("EXTRACTOR_TOKEN", "").strip(),
            max_body_bytes=_int("EXTRACTOR_MAX_BODY_BYTES", 50 * 1024 * 1024, 1024, 200 * 1024 * 1024),
            max_pdf_pages=_int("EXTRACTOR_MAX_PDF_PAGES", 150, 1, 1000),
            max_ocr_pages=_int("EXTRACTOR_MAX_OCR_PAGES", 40, 1, 150),
            max_table_pages=_int("EXTRACTOR_MAX_TABLE_PAGES", 10, 1, 50),
            max_job_timeout_s=_int("EXTRACTOR_MAX_JOB_TIMEOUT_S", 1200, 30, 3600),
            max_waiting=_int("EXTRACTOR_MAX_WAITING", 2, 0, 16),
            worker_rss_limit_bytes=_int("EXTRACTOR_WORKER_RSS_LIMIT_MB", 6144, 512, 65536) * 1024 * 1024,
            ocr_dpi=_int("EXTRACTOR_OCR_DPI", 300, 150, 400),
            model_dir=os.environ.get("EXTRACTOR_MODEL_DIR", "/models"),
            tessdata_dir=os.environ.get("EXTRACTOR_TESSDATA_DIR", "/models/tessdata"),
            torch_threads=_int("EXTRACTOR_TORCH_THREADS", 2, 1, 16),
            work_dir=os.environ.get("EXTRACTOR_WORK_DIR", "/tmp/extractor-jobs"),
        )

    def engine_config(self) -> dict:
        """The subset passed to the worker process (no token)."""
        return {"model_dir": self.model_dir, "tessdata_dir": self.tessdata_dir, "ocr_dpi": self.ocr_dpi,
                "torch_threads": self.torch_threads, "work_dir": self.work_dir, "max_pdf_pages": self.max_pdf_pages}
