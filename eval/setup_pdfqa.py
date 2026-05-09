#!/usr/bin/env python3
"""
Fetch the real-pdfQA benchmark parquet directly from GitHub and build eval/dataset.json.

Run first:
    pip install requests pandas pyarrow
    python eval/setup_pdfqa.py
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_DIR = Path(__file__).parent
DATASET_OUT = EVAL_DIR / "dataset.json"
PDFS_OUT = EVAL_DIR / "pdfs_needed.json"

TARGET_COUNT = 15
TARGET_TYPES = {"yes-no-question", "one-sentence-answer", "value-question"}
AVOID_TYPE = "open-ended-question-long"
DEFAULT_MODEL = "qwen3"


# ---------------------------------------------------------------------------
# Column-name candidates (ordered by preference) for each logical field.
# We try each in turn and take the first one that exists.
# ---------------------------------------------------------------------------
CANDIDATES = {
    "question":      ["question", "Question", "query"],
    "answer":        ["answer", "Answer", "expected_answer", "ground_truth"],
    "document_name": ["document_name", "file_name", "doc_name", "filename"],
    "pdf_url":       ["pdf_url", "url", "pdf_link", "download_url"],
    "pdfqa_id":      ["id", "ID", "pdfqa_id", "index"],
    "answer_type":   ["answer_type", "type", "question_type", "complexity"],
}


def resolve(row, key: str):
    """Return the value for the first matching candidate column, or None."""
    for col in CANDIDATES[key]:
        if col in row and row[col] is not None:
            val = row[col]
            return str(val).strip() if val != "" else None
    return None


def fetch_qa_rows() -> list[dict]:
    url = "https://github.com/tobischimanski/pdfQA/raw/main/real-pdfQA.parquet"
    print(f"Fetching: {url}")
    try:
        r = requests.get(url, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        return []
    if r.status_code != 200:
        print(f"ERROR: HTTP {r.status_code}")
        return []
    df = pd.read_parquet(io.BytesIO(r.content))
    print(f"Columns  : {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    print(f"First row: {df.iloc[0].to_dict()}")
    return df.to_dict(orient="records")


def inspect(first_row: dict, column_names: list):
    print(f"\n{'='*60}")
    print("DATASET SCHEMA")
    print(f"{'='*60}")
    print(f"Columns    : {column_names}")
    print(f"\nFirst row:")
    for col, val in first_row.items():
        preview = str(val)[:120] + ("..." if len(str(val)) > 120 else "")
        print(f"  {col!r:25s} = {preview!r}")
    print()


def pick_column(first_row: dict, key: str) -> str | None:
    """Return which actual column name maps to our logical field, or None."""
    for col in CANDIDATES[key]:
        if col in first_row:
            return col
    return None


def select_rows(ds, first_row: dict) -> list[dict]:
    """
    Select TARGET_COUNT rows from the full list with these priorities:
      1. Prefer answer_type in TARGET_TYPES (avoid AVOID_TYPE)
      2. Maximise document diversity (at least 3 distinct source PDFs)
    """
    col_answer_type = pick_column(first_row, "answer_type")
    col_doc = pick_column(first_row, "document_name")

    preferred, fallback = [], []
    seen_docs: set = set()

    for i, row in enumerate(ds):
        atype = str(row[col_answer_type]).strip() if col_answer_type else ""
        if atype == AVOID_TYPE:
            continue
        bucket = preferred if atype in TARGET_TYPES else fallback
        bucket.append((i, row, atype))

    chosen = []
    doc_counts: Counter = Counter()

    def add(idx, row, atype):
        doc = str(row[col_doc]).strip() if col_doc else f"unknown_{idx}"
        chosen.append((idx, row, atype, doc))
        doc_counts[doc] += 1

    # First pass: fill from preferred, biased toward diversity
    for idx, row, atype in preferred:
        if len(chosen) >= TARGET_COUNT:
            break
        doc = str(row[col_doc]).strip() if col_doc else f"unknown_{idx}"
        if doc not in seen_docs or sum(1 for d in doc_counts if d == doc) == 0:
            add(idx, row, atype)
            seen_docs.add(doc)

    # Second pass: backfill preferred without diversity constraint
    for idx, row, atype in preferred:
        if len(chosen) >= TARGET_COUNT:
            break
        if not any(c[0] == idx for c in chosen):
            doc = str(row[col_doc]).strip() if col_doc else f"unknown_{idx}"
            add(idx, row, atype)

    # Third pass: fill from fallback if still short
    for idx, row, atype in fallback:
        if len(chosen) >= TARGET_COUNT:
            break
        if not any(c[0] == idx for c in chosen):
            doc = str(row[col_doc]).strip() if col_doc else f"unknown_{idx}"
            add(idx, row, atype)

    return chosen


_HF_PDF_BASE = (
    "https://huggingface.co/datasets/pdfqa/pdfQA-Benchmark"
    "/resolve/main/real-pdfQA/01.2_Input_Files_PDF"
)


def build_entry(idx: int, row, atype: str, doc: str) -> dict:
    file_name = row.get("file_name") or None
    dataset = row.get("dataset") or ""
    if file_name:
        pdf_url = f"{_HF_PDF_BASE}/{dataset}/{file_name}.pdf"
    else:
        pdf_url = None

    return {
        "question":        resolve(row, "question") or f"(no question at row {idx})",
        "expected_answer": resolve(row, "answer") or "",
        "document_name":   doc,
        "pdf_url":         pdf_url,
        "pdfqa_id":        str(idx),      # row index from select_rows enumeration
        # answer_type is not a column in real-pdfQA; left as None intentionally
        "answer_type":     None,
        "model":           DEFAULT_MODEL,
    }


def main():
    rows = fetch_qa_rows()
    if not rows:
        sys.exit(1)

    first_row = rows[0]
    inspect(first_row, list(first_row.keys()))

    # Warn about any fields we could not map
    for key in CANDIDATES:
        col = pick_column(first_row, key)
        status = f"→ {col!r}" if col else "NOT FOUND (will be null)"
        print(f"  Field {key!r:15s} {status}")
    print()

    chosen = select_rows(rows, first_row)

    if not chosen:
        print("ERROR: Could not select any rows from dataset.")
        print("Check that column names above matched expectations.")
        sys.exit(1)

    entries = [build_entry(idx, row, atype, doc) for idx, row, atype, doc in chosen]

    # Save dataset.json
    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    # Build pdfs_needed.json — unique (document_name, pdf_url) pairs
    seen: set = set()
    pdfs_needed = []
    for e in entries:
        key = e["document_name"]
        if key not in seen:
            seen.add(key)
            pdfs_needed.append({
                "document_name": e["document_name"],
                "pdf_url": e["pdf_url"],
            })

    with open(PDFS_OUT, "w", encoding="utf-8") as f:
        json.dump(pdfs_needed, f, indent=2, ensure_ascii=False)

    # Summary
    type_counts = Counter(e["answer_type"] for e in entries)
    print(f"{'='*60}")
    print(f"SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions selected : {len(entries)}")
    print(f"Unique source PDFs       : {len(pdfs_needed)}")
    print()
    print("Breakdown by answer_type:")
    for atype, count in type_counts.most_common():
        print(f"  {str(atype or 'none'):35s} {count}")
    print()
    print("PDFs needed:")
    for p in pdfs_needed:
        url_display = p["pdf_url"] or "(no URL in dataset — manual download required)"
        print(f"  {p['document_name']}")
        print(f"    {url_display}")
    print()

    if any(p["pdf_url"] is None for p in pdfs_needed):
        print("NOTE: Some PDFs have no URL in the dataset.")
        print("      You will need to source those PDFs manually and place them in")
        print("      eval/source_pdfs/<document_name> before running download_pdfs.py.")
        print()

    print(f"Saved: {DATASET_OUT}")
    print(f"Saved: {PDFS_OUT}")
    print()
    print("Next step:")
    print("  python eval/download_pdfs.py")


if __name__ == "__main__":
    main()
