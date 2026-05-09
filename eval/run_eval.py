#!/usr/bin/env python3
"""
SynergeReader evaluation script.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --dry-run
    SYNERGEREADER_URL=http://localhost:5000 python eval/run_eval.py
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# Ensure Unicode output works on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from metrics import exact_match, f1_score

BASE_URL = os.environ.get("SYNERGEREADER_URL", "http://localhost:5000").rstrip("/")
ASK_ENDPOINT = f"{BASE_URL}/ask"
TIMEOUT = 60

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"
SOURCE_PDFS_DIR = Path(__file__).parent / "source_pdfs"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_stream(response: requests.Response) -> tuple[str, str | None]:
    """
    Parse a plain-text streaming response from /ask.

    Returns (answer_text, error_message_or_None).
    Collects tokens that appear after __READY__ and before __ENTRY_ID__.
    """
    buffer = ""
    collecting = False
    answer_parts = []
    error = None

    for chunk in response.iter_content(decode_unicode=True, chunk_size=64):
        if not chunk:
            continue
        buffer += chunk

        # Error marker — may span a chunk boundary, so check the whole buffer
        err_match = re.search(r"__ERROR__(.+?)__", buffer)
        if err_match:
            error = err_match.group(1)
            break

        # Entry ID signals end of stream
        if "__ENTRY_ID__" in buffer:
            # Grab any trailing answer text before the marker
            before_id = buffer.split("__ENTRY_ID__")[0]
            if collecting:
                # Strip leading \n\n that precede the marker
                answer_parts.append(before_id.rstrip("\n"))
            break

        # Transition to collecting state at __READY__
        if "__READY__" in buffer:
            collecting = True
            _, buffer = buffer.split("__READY__", 1)
            # Drop the trailing \n after __READY__
            buffer = buffer.lstrip("\n")
            continue

        # While collecting, flush complete lines to answer_parts
        if collecting and "\n" in buffer:
            lines = buffer.split("\n")
            # Keep last (possibly incomplete) line in buffer
            buffer = lines[-1]
            for line in lines[:-1]:
                answer_parts.append(line)

    # Anything left in the buffer that we were collecting
    if collecting and buffer and "__ENTRY_ID__" not in buffer:
        answer_parts.append(buffer)

    answer = " ".join(p for p in answer_parts if p.strip()).strip()
    return answer, error


def call_api(question: str, model: str, selected_text: str = "") -> tuple[str, str | None]:
    payload = {
        "selected_text": selected_text,
        "question": question,
        "model": model,
        "auth_token": None,
    }
    try:
        with requests.post(
            ASK_ENDPOINT,
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            return parse_stream(resp)
    except requests.exceptions.Timeout:
        return "", "TIMEOUT"
    except requests.exceptions.ConnectionError as e:
        return "", f"CONNECTION ERROR: {e}"
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP ERROR: {e}"


def truncate(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def run(dry_run: bool = False):
    dataset = load_dataset()
    total = len(dataset)

    # Group by model for the header
    models_used = {item.get("model", "unknown") for item in dataset}
    model_label = ", ".join(sorted(models_used))

    width = 61
    print("=" * width)
    print(f"  SynergeReader — Evaluation Run")
    print(f"  Benchmark: pdfQA (real-pdfQA) | Source: ETH Zurich / UZH 2026")
    print(f"  Model: {model_label} | Questions: {total}")
    if dry_run:
        print("  Mode: DRY RUN (no API calls)")
    print("=" * width)

    results = []

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        expected = item["expected_answer"]
        model = item.get("model", "qwen3")

        print(f"\n[{i}] Q: {truncate(question)}")
        print(f"    Expected : \"{truncate(expected)}\"")

        if dry_run:
            print("    Got      : (skipped — dry run)")
            print("    EM: -   F1: -")
            results.append({
                "index": i,
                "question": question,
                "expected": expected,
                "got": None,
                "em": None,
                "f1": None,
                "error": "dry-run",
            })
            continue

        got, error = call_api(question, model)

        if error == "TIMEOUT":
            print(f"    Got      : ⚠ TIMEOUT after {TIMEOUT}s")
            print("    EM: -   F1: -")
            results.append({
                "index": i,
                "question": question,
                "expected": expected,
                "got": None,
                "em": None,
                "f1": None,
                "error": "timeout",
            })
            continue

        if error:
            print(f"    Got      : ⚠ ERROR: {truncate(error)}")
            print("    EM: -   F1: -")
            results.append({
                "index": i,
                "question": question,
                "expected": expected,
                "got": None,
                "em": None,
                "f1": None,
                "error": error,
            })
            continue

        em = exact_match(got, expected)
        f1 = f1_score(got, expected)

        em_icon = "✅" if em else "❌"
        print(f"    Got      : \"{truncate(got)}\"")
        print(f"    EM: {em_icon}  F1: {f1:.2f}")

        results.append({
            "index": i,
            "question": question,
            "expected": expected,
            "got": got,
            "em": em,
            "f1": round(f1, 4),
            "error": None,
        })

    # Compute overall scores (skip errored/dry-run questions)
    scored = [r for r in results if r["em"] is not None]
    em_pct = round(100 * sum(r["em"] for r in scored) / len(scored)) if scored else 0
    avg_f1 = round(sum(r["f1"] for r in scored) / len(scored), 2) if scored else 0.0

    print()
    print("=" * width)
    if scored:
        print(f"  OVERALL  →  EM: {em_pct}%  |  Avg F1: {avg_f1}  ({len(scored)}/{total} scored)")
    else:
        print("  OVERALL  →  No questions scored (all errored or dry-run)")
    print("=" * width)

    if not dry_run:
        save_results(results, em_pct, avg_f1, total, model_label)


def save_results(results, em_pct, avg_f1, total, model_label):
    RESULTS_DIR.mkdir(exist_ok=True)
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "benchmark": "pdfQA-real",
        "benchmark_paper": "arxiv.org/abs/2601.02285",
        "model": model_label,
        "total_questions": total,
        "scored": len([r for r in results if r["em"] is not None]),
        "overall_em_pct": em_pct,
        "overall_avg_f1": avg_f1,
        "questions": results,
    }
    path = RESULTS_DIR / "latest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {path}")


def setup_check():
    """Check that all PDFs referenced in dataset.json are present in source_pdfs/."""
    dataset = load_dataset()
    needed = {
        item["document_name"]
        for item in dataset
        if item.get("document_name")
    }

    width = 61
    print("=" * width)
    print("  Setup Check — PDF availability")
    print("=" * width)

    if not needed:
        print("  No document_name entries in dataset.json.")
        return

    present, missing = [], []
    for name in sorted(needed):
        path = SOURCE_PDFS_DIR / name
        if path.exists() and path.stat().st_size > 0:
            present.append(name)
        else:
            missing.append(name)

    for name in present:
        print(f"  ✅ {name}")
    for name in missing:
        print(f"  ❌ {name}  ← not found in eval/source_pdfs/")

    print()
    print(f"  {len(present)}/{len(needed)} PDFs present")
    if missing:
        print()
        print("  Run download_pdfs.py to fetch missing PDFs.")
    print("=" * width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynergeReader evaluation script")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset and print structure without calling the API",
    )
    parser.add_argument(
        "--setup-check",
        action="store_true",
        help="Check which source PDFs are present in eval/source_pdfs/ without running eval",
    )
    args = parser.parse_args()

    # Resolve relative imports when running as a script
    sys.path.insert(0, str(Path(__file__).parent))

    if args.setup_check:
        setup_check()
    else:
        run(dry_run=args.dry_run)
