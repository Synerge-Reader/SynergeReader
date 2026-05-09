# SynergeReader — Evaluation System

Measures answer quality against a fixed Q&A dataset using Exact Match (EM) and token-level F1.

## Files

| File | Purpose |
|------|---------|
| `run_eval.py` | Main script — sends questions, scores answers, prints report |
| `dataset.json` | 10 Q&A pairs (replace with real pdfQA questions) |
| `metrics.py` | EM and F1 functions (no external dependencies) |
| `results/latest.json` | Written after each run |

## Requirements

Only the standard library plus `requests`:

```bash
pip install requests
```

## Usage

```bash
# From the repo root
python eval/run_eval.py

# Custom backend URL
SYNERGEREADER_URL=http://localhost:5000 python eval/run_eval.py

# Dry run — verify dataset loads, no API calls
python eval/run_eval.py --dry-run
```

## API endpoint

The script posts to `POST /ask` directly on port 5000. If you're routing through
nginx (e.g. `localhost:80/api/ask`), set `SYNERGEREADER_URL=http://localhost:80/api`
and the script will hit `http://localhost:80/api/ask`.

Default: `http://localhost:5000`

## Request payload

```json
{
  "selected_text": "",
  "question": "What is the penalty under Section 4?",
  "model": "qwen3",
  "auth_token": null
}
```

`selected_text` is left empty for all eval questions — the backend will use
pgvector retrieval to find relevant context automatically.

## dataset.json schema

```json
[
  {
    "question": "...",
    "expected_answer": "...",
    "document_name": null,
    "model": "qwen3"
  }
]
```

`document_name` is metadata only — not sent in the request (current backend
schema does not support scoped document filtering via the request body).

## Scoring

- **Exact Match (EM):** case-insensitive, punctuation-stripped string equality.
- **F1:** token overlap between predicted and expected answer (same normalisation).
  A score of 1.0 means perfect overlap; 0.0 means no shared tokens.

## Upgrading to real pdfQA data

Replace `dataset.json` entries with questions from the
[pdfQA benchmark](https://arxiv.org/abs/2601.02285) (`pdfqa/pdfQA-Benchmark` on HuggingFace).
Keep the same JSON shape; set `document_name` for reference (documents must already be
uploaded to the server for retrieval to work).
