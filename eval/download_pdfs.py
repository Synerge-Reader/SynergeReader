#!/usr/bin/env python3
"""
Download source PDFs from pdfs_needed.json and upload them to SynergeReader.

Run after setup_pdfqa.py:
    python eval/download_pdfs.py

Environment variables:
    SYNERGEREADER_URL   Base URL of the backend (default: http://localhost:5000)
"""

import json
import os
import sys
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = os.environ.get("SYNERGEREADER_URL", "http://localhost:5000").rstrip("/")
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"

EVAL_DIR = Path(__file__).parent
PDFS_NEEDED = EVAL_DIR / "pdfs_needed.json"
SOURCE_PDFS_DIR = EVAL_DIR / "source_pdfs"

DOWNLOAD_TIMEOUT = 30
UPLOAD_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> tuple[bool, str]:
    """Download url → dest. Returns (success, reason_if_failed)."""
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                return False, f"Server returned HTML instead of PDF (Content-Type: {content_type})"
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        if dest.stat().st_size == 0:
            dest.unlink()
            return False, "Downloaded file is empty"
        return True, ""
    except requests.exceptions.Timeout:
        return False, f"Timeout after {DOWNLOAD_TIMEOUT}s"
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {e}"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_pdf(path: Path) -> tuple[bool, str]:
    """
    Upload a PDF to POST /upload as multipart/form-data.
    Matches the endpoint signature:
        file: UploadFile
        author, title, publication_date, source, doi_url: optional Form fields
    """
    try:
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "application/pdf")}
            resp = requests.post(
                UPLOAD_ENDPOINT,
                files=files,
                timeout=UPLOAD_TIMEOUT,
            )
        resp.raise_for_status()
        body = resp.json()

        # Response is a list of result dicts
        if isinstance(body, list):
            result = body[0] if body else {}
        else:
            result = body

        if "error" in result:
            return False, result["error"]
        return True, result.get("message", "ok")

    except requests.exceptions.Timeout:
        return False, f"Upload timeout after {UPLOAD_TIMEOUT}s"
    except requests.exceptions.ConnectionError as e:
        return False, f"Connection error: {e}"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:200]
        return False, f"HTTP {e.response.status_code}: {detail}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not PDFS_NEEDED.exists():
        print(f"ERROR: {PDFS_NEEDED} not found.")
        print("Run setup_pdfqa.py first.")
        sys.exit(1)

    with open(PDFS_NEEDED, "r", encoding="utf-8") as f:
        pdfs = json.load(f)

    if not pdfs:
        print("pdfs_needed.json is empty — nothing to do.")
        sys.exit(0)

    SOURCE_PDFS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(pdfs)
    dl_ok, dl_fail = [], []
    up_ok, up_fail = [], []

    print(f"{'='*60}")
    print(f"  PDF Download + Upload")
    print(f"  Backend: {BASE_URL}")
    print(f"  PDFs: {total}")
    print(f"{'='*60}\n")

    for i, entry in enumerate(pdfs, start=1):
        doc_name = entry["document_name"]
        pdf_url = entry.get("pdf_url")
        dest = SOURCE_PDFS_DIR / doc_name

        # ---- Download ----
        print(f"Downloading {i}/{total}: {doc_name}")

        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ↳ Already exists, skipping download.")
            dl_ok.append(doc_name)
        elif not pdf_url:
            print(f"  ↳ ⚠  No URL in dataset — place file manually at:")
            print(f"       {dest}")
            dl_fail.append((doc_name, "no URL available"))
        else:
            ok, reason = download_pdf(pdf_url, dest)
            if ok:
                size_kb = dest.stat().st_size // 1024
                print(f"  ↳ ✅ Downloaded ({size_kb} KB)")
                dl_ok.append(doc_name)
            else:
                print(f"  ↳ ❌ Download failed: {reason}")
                dl_fail.append((doc_name, reason))

        # ---- Upload (only if file is present) ----
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  ↳ Uploading to SynergeReader...")
            ok, reason = upload_pdf(dest)
            if ok:
                print(f"  ↳ ✅ Uploaded: {doc_name}")
                up_ok.append(doc_name)
            else:
                print(f"  ↳ ❌ Upload failed: {reason}")
                up_fail.append((doc_name, reason))
        else:
            up_fail.append((doc_name, "file not available for upload"))

        print()

    # ---- Summary ----
    print(f"{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded : {len(dl_ok)}/{total}")
    print(f"  Uploaded   : {len(up_ok)}/{total}")

    if dl_fail:
        print(f"\n  Download failures ({len(dl_fail)}):")
        for name, reason in dl_fail:
            print(f"    ❌ {name}: {reason}")

    if up_fail:
        print(f"\n  Upload failures ({len(up_fail)}):")
        for name, reason in up_fail:
            print(f"    ❌ {name}: {reason}")

    if not dl_fail and not up_fail:
        print("\n  All PDFs downloaded and uploaded successfully.")

    print()
    print("Next step:")
    print("  python eval/run_eval.py")


if __name__ == "__main__":
    main()
