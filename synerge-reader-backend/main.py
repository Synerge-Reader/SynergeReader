from contextlib import asynccontextmanager
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
from document_parser import sanitize_filename
from document_ingestion import (
    CommittedDocument,
    DocumentIngestionService,
    DocumentMetadata,
    UploadDocument,
)
from document_retrieval import (
    build_authorized_lexical_query,
    build_authorized_semantic_query,
    build_relevant_chunks_query,
    lexical_chunk_from_row,
    reciprocal_rank_fusion,
    retrieved_chunk_from_row,
)
from answer_evidence import (
    AnswerEvidencePlanner,
    AuthorizedDocument,
    AuthorizedScope,
    EvidenceItem,
    EvidenceLimits,
    EvidenceRequest,
    SelectedTextInput,
)
from citation_generation import (
    AnswerMode,
    CitationGenerationResult,
    CitationLimits,
    CitationRegistry,
    build_generation_prompt,
    build_mode_prompt,
    citations_without_claims,
    generate_citations,
    safe_error,
)
from ask_stream import (
    STREAM_MEDIA_TYPE,
    delta_event,
    done_event,
    entry_id_event,
    error_event,
    evidence_event,
    verification_event,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from schemas import AskRequest, AskResponse, CorrectionRequest, RatingRequest,GoogleLoginRequest,LoginRequest,RegisterRequest,ResetPasswordRequest,ResendVerificationRequest
from schemas import HistoryItem,HistoryRequest, KnowledgeItem,KnowledgeInsertRequest,ForgotPasswordRequest,KnowledgeUrlImportRequest
import os
import string
import datetime
import re
from typing import List, Optional
from dbSetup import VectorSchemaError,init_db,connect_to_postgres,test_postgres_connection
from rag_model_profiles import resolve_embedding_profile
from ollama_embedding_provider import EmbeddingProviderError, OllamaEmbeddingProvider
import requests
import json
import time
from pydantic import BaseModel
import bcrypt
import secrets
from dotenv import load_dotenv
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
import resend
import subprocess
import tempfile
import shutil
import socket
import ipaddress
from urllib.parse import urlparse, urljoin
from lxml import html as lxml_html
from email_validator import validate_email, EmailNotValidError
from fastapi.responses import Response

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize required database state before accepting requests."""
    try:
        _initialize_application()
    except VectorSchemaError as exc:
        message = " ".join(str(exc).splitlines())
        print(
            f"Application startup blocked by vector schema validation: {message}",
            file=sys.stderr,
            flush=True,
        )
        raise
    yield


app = FastAPI(title="SynergeReader API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "172.18.0.1")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip()
OLLAMA_FALLBACK_HOSTS = [
    host.strip()
    for host in os.getenv(
        "OLLAMA_FALLBACK_HOSTS",
        "host.docker.internal,172.18.0.1,127.0.0.1,localhost",
    ).split(",")
    if host.strip()
]
OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "1.5"))
OLLAMA_READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "60"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
_ACTIVE_OLLAMA_BASE_URL = None
_OLLAMA_HEALTH_CHECKED_AT = 0
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
resend.api_key = os.getenv("EMAIL_KEY")
# Where the verify-email and reset-password links in emails point back to.
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
# Emails that get is_admin=1 automatically at the moment they register.
# Does NOT retroactively affect accounts that already exist — those are
# still promoted by hand via the admin dashboard's User Management tab.
ADMIN_EMAILS = {
    e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()
}
# ------------------- Utilities -------------------


def ollama_base_urls() -> list[str]:
    """Return Ollama base URLs in priority order."""
    if OLLAMA_BASE_URL:
        return [OLLAMA_BASE_URL.rstrip("/")]

    urls = [f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"]
    for host in OLLAMA_FALLBACK_HOSTS:
        url = f"http://{host}:{OLLAMA_PORT}"
        if url not in urls:
            urls.append(url)
    return urls


def get_active_ollama_base_url() -> str:
    """Resolve and cache the first reachable Ollama endpoint."""
    global _ACTIVE_OLLAMA_BASE_URL, _OLLAMA_HEALTH_CHECKED_AT

    now = time.time()
    if _ACTIVE_OLLAMA_BASE_URL and now - _OLLAMA_HEALTH_CHECKED_AT < 30:
        return _ACTIVE_OLLAMA_BASE_URL

    errors = []
    for base_url in ollama_base_urls():
        try:
            resp = requests.get(
                f"{base_url}/api/tags",
                timeout=(OLLAMA_CONNECT_TIMEOUT, 5),
            )
            resp.raise_for_status()
            _ACTIVE_OLLAMA_BASE_URL = base_url
            _OLLAMA_HEALTH_CHECKED_AT = now
            return base_url
        except Exception as e:
            errors.append(f"{base_url}: {e}")

    _ACTIVE_OLLAMA_BASE_URL = None
    _OLLAMA_HEALTH_CHECKED_AT = now
    raise RuntimeError("Ollama is not reachable. Checked " + " | ".join(errors))


def post_ollama(endpoint: str, payload: dict, *, stream: bool = False, timeout: int = 60):
    base_url = get_active_ollama_base_url()
    return requests.post(
        f"{base_url}{endpoint}",
        json=payload,
        stream=stream,
        timeout=(OLLAMA_CONNECT_TIMEOUT, timeout or OLLAMA_READ_TIMEOUT),
    )


# ------------------- Embedding composition boundary -------------------
#
# resolve_embedding_profile(os.environ) is the only permitted entry point
# here (never known_embedding_profile(...) directly): it enforces the
# all-or-nothing five-field EMBEDDING_* override, the unverified-profile
# acknowledgement, registry dimension compatibility, and the documented
# mxbai/1024 default when the override keys are absent. This runs once at
# import time, so a partial or unacknowledged override fails startup rather
# than surfacing later as a confusing runtime error.
_EMBEDDING_PROFILE = resolve_embedding_profile(os.environ)


def _initialize_application() -> None:
    """Initialize database state using the resolved embedding dimension."""
    init_db(expected_dimension=_EMBEDDING_PROFILE.dimension)


def _post_embedding_request(endpoint: str, payload: dict):
    """Adapter binding OllamaEmbeddingProvider's transport to post_ollama,
    with the embedding request timeout pinned to 30s regardless of the
    generation timeout used elsewhere."""
    return post_ollama(endpoint, payload, timeout=30)


_EMBEDDING_PROVIDER = OllamaEmbeddingProvider(
    profile=_EMBEDDING_PROFILE,
    http_post=_post_embedding_request,
    keep_alive=OLLAMA_KEEP_ALIVE,
)


def get_relevant_chunks(
    question: str, top_k: int = 3, document_names: Optional[List[str]] = None
) -> List[dict]:
    """Get relevant chunks ranked by embedding similarity."""
    conn = None
    try:
        # Embed before touching the database: a query-embedding failure must
        # propagate as EmbeddingProviderError (see the except clause below),
        # never silently degrade into "no relevant chunks found".
        question_embedding = _EMBEDDING_PROVIDER.embed_query(question)

        conn = connect_to_postgres()
        if conn is None:
            return []
        c = conn.cursor()
        query, params = build_relevant_chunks_query(question_embedding, top_k, document_names)
        c.execute(query, params)
        rows = c.fetchall()
        return [retrieved_chunk_from_row(row) for row in rows]
    except EmbeddingProviderError:
        raise
    except Exception as e:
        print(f"Error in get_relevant_chunks: {e}")
        return []
    finally:
        if conn is not None:
            conn.close()


def get_relevant_history(
    question: str, selected_text: str, token: Optional[str] = None, limit: int = 3
) -> List[dict]:
    try:
        conn = connect_to_postgres()
        c = conn.cursor()

        user_id = None
        if token:
            c.execute("SELECT id FROM users WHERE token = %s", (token,))
            row = c.fetchone()
            if row:
                user_id = row[0]

        if user_id:
            c.execute(
                "SELECT id, ts, selected_text, question, answer FROM chat_history WHERE user_id = %s ORDER BY id DESC LIMIT 20",
                (user_id,),
            )
        else:
            c.execute(
                "SELECT id, ts, selected_text, question, answer FROM chat_history WHERE user_id IS NULL ORDER BY id DESC LIMIT 20"
            )
        rows = c.fetchall()
        conn.close()

        scored = []
        for id, ts, sel, q, a in rows:
            score = sum(
                [1 if word in q.lower() else 0 for word in question.lower().split()]
            ) + sum(
                [
                    2 if word in sel.lower() else 0
                    for word in selected_text.lower().split()
                ]
            )
            if score > 0:
                scored.append(
                    {
                        "id": id,
                        "timestamp": ts,
                        "selected_text": sel,
                        "question": q,
                        "answer": a,
                        "relevance_score": score,
                    }
                )

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored[:limit]
    except Exception:
        return []


def get_relevant_knowledge_base(question: str, limit: int = 3) -> List[dict]:
    """Retrieve relevant knowledge base entries using semantic similarity (pgvector)."""
    try:
        # Embed the incoming question. A query-embedding failure must
        # propagate as EmbeddingProviderError (see the except clauses below)
        # rather than silently degrading into the keyword fallback below, as
        # though semantic search had merely found nothing.
        q_vec = _EMBEDDING_PROVIDER.embed_query(question)

        conn = connect_to_postgres()
        c = conn.cursor()

        # Try semantic search first
        try:
            c.execute(
                """
                SELECT id, question, corrected_answer, context_text, corrected_by, usage_count,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM knowledge_base
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q_vec, q_vec, limit),
            )
            rows = c.fetchall()
            conn.close()

            results = []
            for row in rows:
                id_, kb_q, kb_ans, ctx, corrected_by, usage_count, sim = row
                if sim >= 0.5:   # Only return if reasonably similar
                    results.append({
                        "id": id_,
                        "question": kb_q,
                        "answer": kb_ans,
                        "context": ctx or "",
                        "corrected_by": corrected_by or "Unknown",
                        "usage_count": usage_count or 0,
                        "relevance_score": float(sim),
                    })
            return results

        except EmbeddingProviderError:
            raise
        except Exception as e:
            print(f"Semantic KB search failed, falling back to keyword: {e}")
            conn.rollback()
            # Fallback: keyword overlap
            c.execute("SELECT id, question, corrected_answer, context_text, corrected_by, usage_count FROM knowledge_base")
            rows = c.fetchall()
            conn.close()
            question_words = set(question.lower().split())
            scored = []
            for id_, kb_q, kb_ans, ctx, corrected_by, usage_count in rows:
                overlap = len(question_words & set(kb_q.lower().split()))
                if overlap > 0:
                    scored.append({
                        "id": id_, "question": kb_q, "answer": kb_ans,
                        "context": ctx or "", "corrected_by": corrected_by or "Unknown",
                        "usage_count": usage_count or 0, "relevance_score": float(overlap),
                    })
            scored.sort(key=lambda x: x["relevance_score"], reverse=True)
            return scored[:limit]

    except EmbeddingProviderError:
        raise
    except Exception as e:
        print(f"Error retrieving knowledge base: {e}")
        return []


def increment_kb_usage(entry_ids: List[int]):
    """Increment usage_count for KB entries that fired."""
    if not entry_ids:
        return
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute(
            "UPDATE knowledge_base SET usage_count = COALESCE(usage_count,0) + 1 WHERE id = ANY(%s)",
            (entry_ids,)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error incrementing KB usage: {e}")


def auto_save_to_kb(question: str, answer: str, source: str = "auto"):
    """
    Silently save a Q&A pair to the knowledge base after every query.
    Skips if: answer is too short, looks like an error, or a duplicate already exists.
    """
    try:
        # Skip low quality answers
        if len(answer.strip()) < 80:
            return
        error_phrases = ["i don't know", "i cannot", "not enough context",
                         "insufficient", "unable to answer", "__error__"]
        if any(p in answer.lower() for p in error_phrases):
            return

        # Embed the question. This runs before any database connection is
        # opened, so an embedding failure never leaves a half-written row —
        # it just logs and skips this save (non-critical background path).
        try:
            q_vec = _EMBEDDING_PROVIDER.embed_documents([question])[0]
        except EmbeddingProviderError as exc:
            print(f"[KB] Auto-save skipped — embedding unavailable: {type(exc).__name__}")
            return

        conn = connect_to_postgres()
        c = conn.cursor()

        # Check for near-duplicate (similarity > 0.92 means essentially the same question)
        try:
            c.execute(
                """
                SELECT id FROM knowledge_base
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> %s::vector) > 0.92
                LIMIT 1
                """,
                (q_vec,)
            )
            if c.fetchone():
                conn.close()
                return  # Already have a very similar entry
        except Exception:
            pass  # pgvector issue — still proceed with insert

        c.execute(
            """
            INSERT INTO knowledge_base
              (question, original_answer, corrected_answer, created_at,
               context_text, corrected_by, usage_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, 1, %s)
            """,
            (question, answer, answer,
             datetime.datetime.now().isoformat(),
             "", source, q_vec)
        )
        conn.commit()
        conn.close()
        print(f"[KB] Auto-saved Q&A: {question[:60]}...")
    except Exception as e:
        print(f"[KB] Auto-save failed (non-critical): {e}")


def _pick_kb_generation_model(base_url: str) -> Optional[str]:
    generation_models = [
        "saul-instruct:latest", "llama3.1:latest", "llama3.1:8b",
        "llama3:latest", "mistral:latest", "qwen2.5:latest",
        "qwen2.5:7b", "phi3:latest", "gemma2:latest", "gemma:latest",
    ]
    for model in generation_models:
        try:
            test = requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": "Say OK", "stream": False},
                timeout=(3, 20),
            )
            if test.status_code == 200 and test.json().get("response"):
                return model
        except Exception:
            continue
    return None


def _generate_qa_pairs_from_text(text: str, label: str) -> List[dict]:
    """Ask the LLM to write Q:/A: pairs from arbitrary text (a document excerpt
    or scraped external page). Shared by document-upload seeding and the
    external URL importer — the only difference between those two source
    kinds is what gets stored alongside the pairs, not how they're generated."""
    if len(text.strip()) < 200:
        return []
    try:
        base_url = get_active_ollama_base_url()
    except Exception:
        print(f"[KB] Ollama not reachable — skipping KB generation for {label}")
        return []

    active_model = _pick_kb_generation_model(base_url)
    if not active_model:
        print(f"[KB] No text-generation model available for {label}")
        return []

    mid = min(len(text) // 2, 4000)
    excerpts = [text[:4000].strip(), text[mid:mid + 4000].strip()]
    all_pairs = []

    for i, excerpt in enumerate(excerpts):
        if len(excerpt) < 150:
            continue

        prompt = f"""Read the document excerpt below and write 5 question-answer pairs.

Format — use EXACTLY this pattern for each pair, nothing else:
Q: <your question here>
A: <your answer here (2-3 sentences)>

Q: <next question>
A: <next answer>

Rules:
- Questions must be specific and answerable from the text
- Answers must come directly from the text, 2-3 sentences each
- Do not add any intro text, numbering, bullets, or JSON

Document (part {i+1}):
{excerpt}

Start with Q:"""

        try:
            resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": active_model, "prompt": prompt, "stream": False, "temperature": 0.2},
                timeout=(5, 120),
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            print(f"[KB] Raw output part {i+1} (first 300 chars): {raw[:300]}")

            current_q = None
            current_a_lines = []
            for line in raw.splitlines():
                line = line.strip()
                if line.lower().startswith("q:"):
                    if current_q and current_a_lines:
                        all_pairs.append({"question": current_q, "answer": " ".join(current_a_lines).strip()})
                    current_q = line[2:].strip()
                    current_a_lines = []
                elif line.lower().startswith("a:") and current_q:
                    current_a_lines = [line[2:].strip()]
                elif current_a_lines and line:
                    current_a_lines.append(line)
            if current_q and current_a_lines:
                all_pairs.append({"question": current_q, "answer": " ".join(current_a_lines).strip()})

            print(f"[KB] Parsed {len(all_pairs)} pairs so far after part {i+1}")
        except Exception as e:
            print(f"[KB] LLM call failed for excerpt {i+1}: {e}")
            continue

    return all_pairs


def _save_kb_pairs(pairs: List[dict], source_label: str, source_type: str, corrected_by: str) -> int:
    """Dedupe within the batch, embed, and insert. Returns the number saved.

    Fails closed: if embedding a pair raises EmbeddingProviderError, the
    entire uncommitted batch is rolled back and this returns 0 immediately —
    never a NULL-embedding row, and never a partially committed batch. Both
    callers (generate_kb_from_document, import_knowledge_from_url) already
    treat a return value of 0 as "nothing was saved".
    """
    if not pairs:
        return 0
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        saved = 0
        seen = set()
        for pair in pairs:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()
            if not q or not a or len(a) < 20:
                continue
            if q.lower() in seen:
                continue
            seen.add(q.lower())

            try:
                q_vec = _EMBEDDING_PROVIDER.embed_documents([q])[0]
            except EmbeddingProviderError as exc:
                print(
                    f"[KB] Embedding failed while saving a {source_type} batch; "
                    f"rolling back {saved} pending row(s): {type(exc).__name__}"
                )
                conn.rollback()
                return 0

            c.execute(
                """
                INSERT INTO knowledge_base
                  (question, original_answer, corrected_answer, created_at,
                   context_text, corrected_by, usage_count, embedding, source_type)
                VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s)
                """,
                (q, a, a, datetime.datetime.now().isoformat(), source_label, corrected_by, q_vec, source_type),
            )
            saved += 1
        conn.commit()
        return saved
    finally:
        conn.close()


def generate_kb_from_document(doc_id: int, filename: str, text: str):
    """After a document is uploaded, seed the Knowledge Base with Q&A pairs
    generated from it. Called in a background daemon thread — never blocks
    the upload response."""
    try:
        pairs = _generate_qa_pairs_from_text(text, filename)
        if not pairs:
            print(f"[KB] No Q&A pairs extracted from {filename}")
            return
        saved = _save_kb_pairs(pairs, filename, "document", f"Auto-generated from: {filename}")
        print(f"[KB] ✓ Final: saved {saved}/{len(pairs)} KB entries from: {filename}")
    except Exception as e:
        print(f"[KB] Document KB generation failed (non-critical): {e}")


def _extract_json_block(raw: str) -> Optional[dict]:
    """Local models rarely emit clean JSON — they wrap it in markdown fences or
    add a sentence of preamble/postamble. Pull out the first {...} block and
    parse that, tolerating the noise around it."""
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except Exception:
        return None


def _extract_document_insights(doc_id: int, filename: str, text: str):
    """Ask the LLM to read a document and pull out a type classification, key
    facts, keywords, and named entities/parties — then store one row per
    document for the admin dashboard's Insights tab. Runs in a background
    daemon thread; never blocks the upload response, and any failure here is
    non-critical (the document itself is already saved and usable)."""
    if len(text.strip()) < 200:
        return
    try:
        base_url = get_active_ollama_base_url()
    except Exception:
        print(f"[Insights] Ollama not reachable — skipping insight extraction for {filename}")
        return

    active_model = _pick_kb_generation_model(base_url)
    if not active_model:
        print(f"[Insights] No text-generation model available for {filename}")
        return

    excerpt = text[:6000].strip()
    prompt = f"""Read the legal document excerpt below and respond with ONLY a JSON object — no markdown, no commentary, no code fences. Use exactly this shape:

{{
  "doc_type": "<one short category, e.g. Contract, NDA, Lease, Statute, Case Law, Policy, Correspondence, Pleading, Other>",
  "keywords": ["<5 to 10 short keywords or key phrases from the document>"],
  "facts": ["<3 to 6 standalone key facts stated in the document, one sentence each>"],
  "entities": ["<up to 8 named parties, organizations, or people mentioned>"]
}}

Document excerpt ({filename}):
{excerpt}

JSON:"""

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": active_model, "prompt": prompt, "stream": False, "temperature": 0.1},
            timeout=(5, 120),
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        parsed = _extract_json_block(raw)
        if not parsed:
            print(f"[Insights] Could not parse JSON for {filename}: {raw[:200]}")
            return

        doc_type = str(parsed.get("doc_type") or "Other").strip()[:60]
        keywords = [str(k).strip() for k in (parsed.get("keywords") or []) if str(k).strip()][:10]
        facts = [str(f).strip() for f in (parsed.get("facts") or []) if str(f).strip()][:8]
        entities = [str(e).strip() for e in (parsed.get("entities") or []) if str(e).strip()][:10]

        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO document_insights (document_id, doc_type, keywords, facts, entities, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                doc_type = EXCLUDED.doc_type,
                keywords = EXCLUDED.keywords,
                facts = EXCLUDED.facts,
                entities = EXCLUDED.entities,
                created_at = EXCLUDED.created_at
            """,
            (doc_id, doc_type, json.dumps(keywords), json.dumps(facts), json.dumps(entities), datetime.datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        print(f"[Insights] ✓ {filename}: type={doc_type}, {len(keywords)} keywords, {len(facts)} facts, {len(entities)} entities")
    except Exception as e:
        print(f"[Insights] Extraction failed for {filename} (non-critical): {e}")


async def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode()



def _resolve_uploader_id(auth_token: Optional[str]):
    """Resolve an upload's owner from the existing users.token column.

    Unchanged auth semantics for this slice: a token that matches a row gives
    that user's id, anything else gives None (an anonymous upload), exactly as
    before. The only behavioural change is hygiene -- the lookup cursor is now
    closed as well as the connection, and a lookup failure is logged by
    exception class name only, never with the token, the SQL, or the raw
    exception text.
    """
    if not auth_token:
        return None

    lookup_conn = connect_to_postgres()
    if lookup_conn is None:
        return None

    lookup_cursor = None
    try:
        lookup_cursor = lookup_conn.cursor()
        lookup_cursor.execute("SELECT id FROM users WHERE token = %s", (auth_token,))
        lookup_row = lookup_cursor.fetchone()
        return lookup_row[0] if lookup_row else None
    except Exception as exc:
        print(f"[Upload] uploader lookup failed: {type(exc).__name__}")
        return None
    finally:
        if lookup_cursor is not None:
            try:
                lookup_cursor.close()
            except Exception:
                pass
        try:
            lookup_conn.close()
        except Exception:
            pass


class _PostCommitDispatchError(RuntimeError):
    """At least one post-commit follow-up could not be started.

    Raised with a fixed message and nothing else: no exception text, no
    document or chunk content, no filename, no auth token, no SQL, and no
    connection detail. DocumentIngestionService catches it and converts it into
    its existing fixed warning on an already-indexed result.
    """


def _start_background_task(target, args) -> None:
    """Construct and start one daemon follow-up thread.

    Deliberately does NOT suppress a construction or start failure. A follow-up
    that never began is something the caller has to know about, so that the
    ingestion service can record its fixed post-commit warning; swallowing it
    here would report a clean ingestion for work that never ran.

    Isolated as its own module-level function so the post-commit dispatcher can
    be exercised without spawning real threads.
    """
    from threading import Thread
    Thread(target=target, args=args, daemon=True).start()


def _dispatch_upload_followups(committed: CommittedDocument) -> None:
    """Post-commit work for one ingested document.

    DocumentIngestionService calls this only after the document's transaction
    has committed and its connection has been closed, and treats any failure
    here as a warning on an already-indexed result -- so neither follow-up can
    turn a committed document into a reported ingestion failure. Both existing
    follow-ups are preserved and receive the committed document id, the
    service-sanitized filename, and the server-extracted text.

    Both follow-ups are attempted even if the first one cannot be started, and
    only the FACT of a failed start is recorded -- never the exception, its
    text, or anything about the document. If either start failed, one generic
    _PostCommitDispatchError is raised after both attempts.

    Scope of what this can detect: only a failure to START a thread. Once a
    follow-up thread is running, it owns its own errors (both targets already
    swallow and log their own failures), and nothing that happens inside it
    afterwards can be observed synchronously here or turned into a warning on
    the ingestion result.
    """
    payload = (committed.document_id, committed.filename, committed.text)
    failed_starts = 0
    for follow_up in (generate_kb_from_document, _extract_document_insights):
        try:
            _start_background_task(follow_up, payload)
        except Exception:
            # Only the count is kept. The exception object is never bound,
            # logged, re-raised, or attached to anything that leaves here.
            failed_starts += 1

    if failed_starts:
        # Raised outside the except block on purpose: with no active exception
        # context there is no implicit chaining, so the original error cannot
        # ride along on __context__ into any caller's log.
        print(f"[Upload] post-commit follow-ups not started: {failed_starts}")
        raise _PostCommitDispatchError("post-commit follow-up could not be started")


def _build_ingestion_service() -> DocumentIngestionService:
    """Compose the verified ingestion service with this application's parts.

    connect_to_postgres is passed as the connection factory specifically
    because it already applies pgvector's register_vector() to every
    connection it returns; the service inserts each embedding as a plain
    list[float], which psycopg2 can only adapt to the `vector` type on a
    registered connection. No second, unregistered psycopg2 connection path
    is introduced anywhere in the upload flow.
    """
    return DocumentIngestionService(
        connection_factory=connect_to_postgres,
        embedding_provider=_EMBEDDING_PROVIDER,
        embedding_profile=_EMBEDDING_PROFILE,
        dispatch_after_commit=_dispatch_upload_followups,
    )


@app.post("/upload")
async def upload_documents(
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    author: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    publication_date: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    doi_url: Optional[str] = Form(None),
    auth_token: Optional[str] = Form(None),
):
    """Thin transport adapter over DocumentIngestionService.

    This route no longer parses, chunks, embeds, or writes anything itself. It
    reads each uploaded file's ORIGINAL bytes exactly once, maps the form
    fields onto the ingestion contract, hands the batch to the service in the
    order the client sent it, and returns the service's batch envelope under
    the service's own aggregated HTTP status (200 / 207 / 422 / 503). Each
    document owns a separate transaction inside the service, so one file's
    failure can never erase a sibling that already committed.
    """
    # The empty-request check comes first on purpose: a request with no files
    # has nothing to ingest, so it must not open a database connection or run
    # an uploader lookup even when an auth_token field was supplied.
    if file and files:
        upload_list = [file] + files
    elif files:
        upload_list = files
    elif file:
        upload_list = [file]
    else:
        raise HTTPException(400, "No files provided")

    uploader_id = _resolve_uploader_id(auth_token)

    metadata = DocumentMetadata(
        author=author,
        title=title,
        publication_date=publication_date,
        source=source,
        doi_url=doi_url,
    )

    uploads = []
    for upload_file in upload_list:
        try:
            # Read once: an UploadFile's stream cannot be replayed, and these
            # are the original PDF/DOCX/TXT bytes the service must parse.
            content = await upload_file.read()
        except Exception as exc:
            print(f"[Upload] could not read an uploaded file: {type(exc).__name__}")
            # A non-bytes payload is rejected per-file by the service as an
            # invalid upload, which keeps the rest of the batch intact.
            content = None
        uploads.append(
            UploadDocument(
                filename=upload_file.filename,
                content=content,
                metadata=metadata,
                uploader_id=uploader_id,
            )
        )

    batch = _build_ingestion_service().ingest_batch(uploads)
    return JSONResponse(status_code=batch.http_status, content=batch.to_dict())



# --- E2: answer evidence, citations, and the /ask event stream --------------
#
# Composition only. The decisions live in the route-independent modules:
# answer_evidence.py owns the selected-text / complete-document / hybrid-
# retrieval priority, citation_generation.py owns registration, prompting,
# marker validation and claim status, and ask_stream.py owns the wire format.

_EVIDENCE_LIMITS = EvidenceLimits()
# Verification is one local model call per claim, serialised, so an answer with
# many cited claims would add that latency to every question. This route caps it
# at two claims; the remainder are reported as `unverified`, which is the honest
# label and never an upgrade to `supported`. The module default is left alone so
# other callers and the unit tests keep their own budget.
_CITATION_LIMITS = CitationLimits(max_claims_verified=2)

# Candidates pulled from each ranker before fusion. Larger than the final
# evidence budget on purpose: fusion needs room to promote a chunk that both
# rankers found.
_HYBRID_CANDIDATES_PER_RANKER = 12


def _resolve_authorized_scope(auth_token: Optional[str]) -> AuthorizedScope:
    """Every document this caller may be shown -- or an unresolved scope.

    Fails closed. If the connection cannot be opened, the lookup raises, or a
    token is supplied that matches no user, the caller gets an unresolved scope
    and therefore no server-loaded evidence at all.

    A caller with no token keeps the established anonymous behaviour: the
    documents that were uploaded without an owner. That is deliberately
    narrower than the previous /ask behaviour, which searched every document in
    the database regardless of who uploaded it.
    """
    connection = connect_to_postgres()
    if connection is None:
        return AuthorizedScope.unresolved()

    cursor = None
    try:
        cursor = connection.cursor()
        user_id = None
        if auth_token:
            cursor.execute("SELECT id FROM users WHERE token = %s", (auth_token,))
            row = cursor.fetchone()
            if not row:
                # A token that resolves to nobody must not quietly degrade into
                # the anonymous corpus.
                return AuthorizedScope.unresolved()
            user_id = row[0]

        if user_id is None:
            cursor.execute(
                """
                SELECT id, filename, title, length(content)
                FROM documents
                WHERE user_id IS NULL
                ORDER BY id
                """
            )
        else:
            cursor.execute(
                """
                SELECT id, filename, title, length(content)
                FROM documents
                WHERE user_id = %s
                ORDER BY id
                """,
                (user_id,),
            )

        documents = []
        for document_id, filename, title, char_length in cursor.fetchall():
            if not filename:
                # A citation must name a file. A row with no filename cannot be
                # cited truthfully, so it is not offered as evidence.
                continue
            documents.append(
                AuthorizedDocument(
                    document_id=document_id,
                    filename=filename,
                    title=title,
                    char_length=char_length,
                )
            )
        return AuthorizedScope(
            documents=tuple(documents),
            established=True,
            user_id=user_id,
            anonymous=user_id is None,
        )
    except Exception as exc:
        print(f"[Ask] authorization scope lookup failed: {type(exc).__name__}")
        return AuthorizedScope.unresolved()
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass


def _load_authorized_document_text(scope: AuthorizedScope, document_id) -> Optional[str]:
    """One authorized document's stored text, or None.

    Scoped twice on purpose: the id must already be in the in-memory authorized
    scope, and the SELECT is scoped to the same owner again, so a bug in the
    caller cannot turn into a cross-user read.
    """
    if not scope.authorizes(document_id):
        return None

    connection = connect_to_postgres()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor()
        if scope.user_id is None:
            cursor.execute(
                "SELECT content FROM documents WHERE id = %s AND user_id IS NULL",
                (document_id,),
            )
        else:
            cursor.execute(
                "SELECT content FROM documents WHERE id = %s AND user_id = %s",
                (document_id, scope.user_id),
            )
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception as exc:
        print(f"[Ask] document load failed: {type(exc).__name__}")
        return None
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass


def _evidence_item_from_chunk(chunk: dict) -> EvidenceItem:
    """One fused retrieval row as evidence, with its locator metadata intact."""
    return EvidenceItem(
        text=chunk.get("text") or "",
        source_type="document_chunk",
        document_id=chunk.get("document_id"),
        filename=chunk.get("document_name"),
        chunk_id=chunk.get("chunk_id"),
        chunk_index=chunk.get("chunk_index"),
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        locator_json=chunk.get("locator"),
        semantic_score=chunk.get("similarity"),
        lexical_score=chunk.get("lexical_rank"),
        combined_score=chunk.get("fusion_score"),
    )


def _hybrid_retrieve_evidence(question: str, document_ids, top_k: int) -> List[EvidenceItem]:
    """Authorized hybrid retrieval: semantic + lexical, fused deterministically.

    The query embedding is produced before the database is touched, so an
    embedding outage propagates as EmbeddingProviderError rather than silently
    degrading into "no relevant evidence". Both queries are scoped to the
    caller's authorized document ids; there is no global search path.
    """
    if not document_ids:
        return []

    question_embedding = _EMBEDDING_PROVIDER.embed_query(question)

    connection = connect_to_postgres()
    if connection is None:
        raise RuntimeError("document search is unavailable")

    cursor = None
    try:
        cursor = connection.cursor()
        semantic_query, semantic_params = build_authorized_semantic_query(
            question_embedding, _HYBRID_CANDIDATES_PER_RANKER, document_ids
        )
        cursor.execute(semantic_query, semantic_params)
        semantic_chunks = [retrieved_chunk_from_row(row) for row in cursor.fetchall()]

        lexical_chunks = []
        try:
            lexical_query, lexical_params = build_authorized_lexical_query(
                question, _HYBRID_CANDIDATES_PER_RANKER, document_ids
            )
            cursor.execute(lexical_query, lexical_params)
            lexical_chunks = [lexical_chunk_from_row(row) for row in cursor.fetchall()]
        except Exception as exc:
            # Lexical ranking is an enhancement, not a precondition: if
            # full-text search is unavailable the answer is still built from
            # the semantic evidence rather than failing outright.
            print(f"[Ask] lexical retrieval unavailable: {type(exc).__name__}")
            try:
                connection.rollback()
            except Exception:
                pass

        fused = reciprocal_rank_fusion(semantic_chunks, lexical_chunks, top_k=top_k)
        return [_evidence_item_from_chunk(chunk) for chunk in fused]
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass


def _build_answer_evidence_planner(scope: AuthorizedScope) -> AnswerEvidencePlanner:
    """The planner, bound to one caller's authorized scope."""
    return AnswerEvidencePlanner(
        load_document_text=lambda document_id: _load_authorized_document_text(
            scope, document_id
        ),
        retrieve=_hybrid_retrieve_evidence,
        limits=_EVIDENCE_LIMITS,
        propagate_exceptions=(EmbeddingProviderError,),
    )


def _evidence_request_from_ask(request: AskRequest, scope: AuthorizedScope) -> EvidenceRequest:
    """Map the wire request onto the planner's input.

    Document scope is taken from backend ids. ``active_document_name`` is still
    honoured for older clients, but only by looking the name up INSIDE the
    already-authorized scope -- a filename never grants access to anything.
    """
    selections = []
    for index, selection in enumerate(request.selections or []):
        text = (selection.text or "").strip()
        if not text:
            continue
        selections.append(
            SelectedTextInput(
                text=text,
                selection_id=selection.id or f"selection-{index + 1}",
                document_id=selection.document_id,
                filename=selection.document_name or None,
                locator_json=selection.locator,
                page_start=selection.page_start,
                page_end=selection.page_end,
            )
        )

    requested: List[int] = []
    if request.active_document_id is not None:
        requested.append(request.active_document_id)
    for document_id in list(request.document_ids or []):
        if document_id not in requested:
            requested.append(document_id)

    if not requested and request.active_document_name:
        for document in scope.documents:
            if document.filename == request.active_document_name:
                requested.append(document.document_id)
                break

    return EvidenceRequest(
        question=request.question,
        selected_text=request.selected_text or "",
        selections=tuple(selections),
        requested_document_ids=tuple(requested),
        top_k=_EVIDENCE_LIMITS.max_evidence_items,
    )


_CLAIM_VERIFIER_PROMPT = (
    "You check whether a claim is supported by evidence excerpts.\n"
    "Answer with exactly one word: supported, partially_supported, or unsupported.\n"
    "\n"
    "Rules:\n"
    "- Answer supported ONLY if the evidence states every part of the claim.\n"
    "- Being about the same topic is NOT support. If the evidence discusses "
    "the subject but does not state what the claim says, answer unsupported.\n"
    "- Every number, model name, dataset name, and section identifier in the "
    "claim must appear in the evidence with the same value. If any differs or "
    "is absent, the claim is not supported.\n"
    "- If the claim makes several assertions and the evidence states only "
    "some of them, answer partially_supported.\n"
    "- If you are unsure, answer unsupported. Do not guess in favour of the "
    "claim.\n"
    "\n"
    "Evidence:\n{evidence}\n\n"
    "Claim:\n{claim}\n\n"
    "One word:"
)

_CLAIM_VERIFIER_WORDS = ("supported", "partially_supported", "unsupported")


def _claim_verifier_status(reply: object) -> Optional[str]:
    """The one status a verifier reply consists of, or None.

    The whole reply must BE a status, not merely contain one: "not supported"
    contains "supported", and prose naming several statuses has chosen none.
    Only surrounding whitespace, quotes, markdown emphasis and trailing
    punctuation are ignored, and a space or hyphen may stand in for the
    underscore in partially_supported. Anything else is ambiguous or malformed
    and returns None, which citation_generation records as ``unverified``.
    """
    if not isinstance(reply, str):
        return None
    word = reply.strip().strip("\"'`*_.!:;,").strip().lower()
    word = re.sub(r"[\s-]+", "_", word)
    return word if word in _CLAIM_VERIFIER_WORDS else None


def _ollama_claim_verifier(claim: str, records, model: str) -> Optional[str]:
    """Local-only claim verification against the cited evidence.

    Reads each record's INTERNAL ``evidence_text`` -- the same text the answer
    model saw -- not the bounded public excerpt. Judging against the excerpt
    would mark a claim unsupported whenever its support happens to sit past the
    excerpt bound, which is exactly what a long selection or a complete short
    document looks like. The verifier's own result stays public-safe: it
    returns one status word, and the claim payload carries only citation ids.

    Runs through the same local Ollama transport as generation and nothing
    else. A transport or model failure raises, and a reply that is not exactly
    one status returns None; citation_generation turns either into
    ``unverified`` -- it can never become ``supported``.
    """
    evidence = "\n\n".join(
        f"[{record.citation_id}] {record.evidence_text}" for record in records
    )
    prompt = _CLAIM_VERIFIER_PROMPT.format(evidence=evidence, claim=claim.strip())
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    response = post_ollama("/api/generate", payload, timeout=30)
    response.raise_for_status()
    return _claim_verifier_status(response.json().get("response"))


def _resolve_citations(
    answer_text: str, registry, verifier=None, *, mode: AnswerMode = AnswerMode.DOCUMENT_QA
) -> CitationGenerationResult:
    """Parse one answer for its mode, never raising into the response stream.

    document_qa: claims are extracted and verified, and the used and invalid
    citation ids are reported -- the only graded, source-linked answers.
    structured_json: only the used ids are parsed; no claim, no verifier call.
    model_reasoning: nothing. Ungraded AI analysis links to no source, so the
    client can never render a navigable source card for an unchecked claim.
    """
    try:
        if mode is AnswerMode.MODEL_REASONING:
            return CitationGenerationResult(registry=registry)
        if mode is AnswerMode.STRUCTURED_JSON:
            return citations_without_claims(answer_text, registry)
        return generate_citations(answer_text, registry, verifier, _CITATION_LIMITS)
    except Exception as exc:
        print(f"[Ask] citation resolution failed: {type(exc).__name__}")
        return CitationGenerationResult(registry=registry)


@app.post("/ask")
async def ask_question(request: AskRequest):
    answer_parts = []
    entry_id = None
    # Decides the prompt rules and whether claims are graded; evidence
    # planning and authorization below are the same in every mode.
    answer_mode = AnswerMode(request.mode)

    def verify_claim(claim, records):
        return _ollama_claim_verifier(claim, records, request.model)

    def build_evidence():
        """Plan the evidence and register its citations. Authorization-scoped."""
        scope = _resolve_authorized_scope(request.auth_token)
        planner = _build_answer_evidence_planner(scope)
        planning = planner.plan(_evidence_request_from_ask(request, scope), scope)
        registry = CitationRegistry.from_bundle(planning.bundle, _CITATION_LIMITS)
        return planning, registry

    def stream_generate():
        nonlocal answer_parts, entry_id
        yield '{"type": "delta", "text": ""}\n'

        stream_error = None
        kb_ids_fired = []
        try:
            planning, registry = build_evidence()

            # ── Knowledge Base injection ──────────────────────────────────
            # Only document_qa reads the knowledge base. Tool JSON and AI
            # analysis are neither steered by saved answers nor count as a
            # use of them. This narrows which answers read the KB; it does
            # not change that the KB itself is shared across users.
            kb_block = ""
            if answer_mode is AnswerMode.DOCUMENT_QA:
                kb_entries = get_relevant_knowledge_base(request.question, limit=3)
                kb_ids_fired = [e["id"] for e in kb_entries]
                if kb_entries:
                    kb_block = "<knowledge_base_corrections>\n"
                    for e in kb_entries:
                        kb_block += f"Q: {e['question']}\nA: {e['answer']}\n---\n"
                    kb_block += "</knowledge_base_corrections>"
            # ─────────────────────────────────────────────────────────────

            if answer_mode is AnswerMode.DOCUMENT_QA:
                prompt = build_generation_prompt(
                    request.question, registry, extra_context=kb_block
                )
            else:
                prompt = build_mode_prompt(answer_mode, request.question, registry)

            yield evidence_event(
                registry.to_dict()["citations"],
                mode=planning.mode.value,
                truncated=planning.bundle.truncated,
                warnings=planning.warnings,
            )

            payload = {
                "model": request.model,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": True,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
        except EmbeddingProviderError:
            yield '{"type": "error", "code": "search_unavailable", "message": "Search is temporarily unavailable. Please try again shortly."}\n'
            yield '{"type": "done", "ok": false}\n'
            return
        except Exception as exc:
            print(f"[Ask] evidence planning failed: {type(exc).__name__}")
            yield '{"type": "error", "code": "evidence_unavailable", "message": "Document evidence is temporarily unavailable."}\n'
            yield '{"type": "done", "ok": false}\n'
            return

        try:
            with post_ollama("/api/generate", payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                buffer = ""
                for chunk in r.iter_content(decode_unicode=True, chunk_size=32):
                    if chunk:
                        if isinstance(chunk, bytes):
                            chunk = chunk.decode("utf-8")
                        buffer += chunk
                        # Process complete JSON objects (lines ending with \n)
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line:
                                try:
                                    data = json.loads(line)
                                    token = data.get("response", "")
                                    if token:
                                        answer_parts.append(token)
                                        yield delta_event(token)
                                except Exception:
                                    continue
                # Handle any remaining buffered data
                if buffer:
                    try:
                        data = json.loads(buffer)
                        token = data.get("response", "")
                        if token:
                            answer_parts.append(token)
                            yield delta_event(token)
                    except Exception:
                        pass
        except Exception as e:
            stream_error = True
            response = getattr(e, "response", None)
            if response is not None:
                yield '{"type": "error", "code": "generation_http_error", "message": "The local model request failed. Check that the requested model is installed in Ollama."}\n'
            else:
                yield '{"type": "error", "code": "generation_unreachable", "message": "The local LLM server is not reachable. Start Ollama or update OLLAMA_BASE_URL / OLLAMA_PORT in .env."}\n'

        # Increment KB usage counts for entries that fired this query
        if kb_ids_fired:
            increment_kb_usage(kb_ids_fired)

        if stream_error or not answer_parts:
            # An interrupted or empty generation must never terminate as a
            # success: done carries ok=false and no entry id is written.
            yield done_event(ok=False)
            return

        full_answer = "".join(answer_parts)

        citation_result = _resolve_citations(
            full_answer, registry, verify_claim, mode=answer_mode
        )
        yield verification_event(
            [claim.to_dict() for claim in citation_result.claims],
            invalid_citation_ids=citation_result.invalid_citation_ids,
            used_citation_ids=citation_result.used_citation_ids,
        )

        conn = None
        committed = False
        try:
            conn = connect_to_postgres()
            c = conn.cursor()

            user_id = None
            if request.auth_token:
                c.execute("SELECT id FROM users WHERE token = %s", (request.auth_token,))
                row = c.fetchone()
                if row:
                    user_id = row[0]

            c.execute(
                """
                INSERT INTO chat_history (ts, selected_text, question, answer, user_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    datetime.datetime.now().isoformat(),
                    request.selected_text or "",
                    request.question,
                    full_answer,
                    user_id,
                )
            )

            entry_id = c.fetchone()[0]
            conn.commit()
            committed = True
        except Exception as exc:
            print(f"[Ask] chat history persistence failed: {type(exc).__name__}")
            # Reached only before the commit completed, so nothing is durable:
            # roll it back. A failed rollback must not replace the safe error.
            if conn is not None:
                try:
                    conn.rollback()
                except Exception as cleanup_exc:
                    print(f"[Ask] chat history rollback failed: {type(cleanup_exc).__name__}")
        finally:
            # Closed on every path. After a commit the row is durable, so a
            # failed close is logged and the answer still succeeds.
            if conn is not None:
                try:
                    conn.close()
                except Exception as cleanup_exc:
                    print(f"[Ask] chat history close failed: {type(cleanup_exc).__name__}")

        if not committed:
            yield error_event(**safe_error("internal_error"))
            yield done_event(ok=False)
            return

        yield entry_id_event(entry_id)

        # Auto-save this Q&A to the Knowledge Base in the background. Only a
        # committed document_qa answer qualifies: tool JSON and ungraded AI
        # analysis are not answers that belong in the KB.
        if answer_mode is AnswerMode.DOCUMENT_QA:
            try:
                from threading import Thread
                Thread(
                    target=auto_save_to_kb,
                    args=(request.question, full_answer, "auto-query"),
                    daemon=True
                ).start()
            except Exception:
                pass

        yield done_event(ok=True)

    return StreamingResponse(
        stream_generate(),
        media_type=STREAM_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )



@app.post("/history", response_model=List[HistoryItem])
async def get_history(request: HistoryRequest):
    user_id = None
    if request.token:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute(
        "SELECT id FROM users WHERE token = %s",
        (request.token,)
        )
        row = c.fetchone()
        if row:
            user_id = row[0]
        conn.close()
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        if user_id:
            c.execute(
                "SELECT id, ts, selected_text, question, answer FROM chat_history WHERE user_id = %s ORDER BY id DESC LIMIT 20",
                (user_id,),
            )
        else:
            c.execute(
                "SELECT id, ts, selected_text, question, answer FROM chat_history WHERE user_id IS NULL ORDER BY id DESC LIMIT 20"
            )
        rows = c.fetchall()
        conn.close()
        return [
            HistoryItem(
                id=r[0], timestamp=r[1], selected_text=r[2], question=r[3], answer=r[4]
            )
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/documents")
async def get_documents():
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("""SELECT id, filename, upload_timestamp, author, title, publication_date, source, doi_url,
                     (SELECT COUNT(*) FROM document_chunks WHERE document_id = documents.id)
                     FROM documents ORDER BY upload_timestamp DESC""")
        rows = c.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "filename": r[1],
                "upload_timestamp": r[2],
                "author": r[3],
                "title": r[4],
                "publication_date": r[5],
                "source": r[6],
                "doi_url": r[7],
                "chunks_count": r[8],
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Token from an ``Authorization: Bearer <token>`` header, or None when the
    header is missing or malformed. Used by the document-history endpoints below
    instead of a ``?token=`` query parameter, which ends up in browser history
    and proxy/access logs."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def _open_db():
    """A live connection, or a clean 503. connect_to_postgres() returns None
    when the database is unreachable, and callers that went straight to
    conn.cursor() / conn.close() crashed with an AttributeError instead."""
    conn = connect_to_postgres()
    if conn is None:
        raise HTTPException(503, "Database temporarily unavailable")
    return conn


def _user_id_for_token(cursor, token: Optional[str]):
    if not token:
        raise HTTPException(401, "Unauthorized")
    cursor.execute("SELECT id FROM users WHERE token = %s", (token,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(401, "Invalid session")
    return row[0]


@app.get("/me/documents/search")
async def search_my_documents(
    q: str = "",
    limit: int = 8,
    authorization: Optional[str] = Header(None),
):
    """Search the caller's own previously-uploaded documents by filename, for
    the header search bar. Admin-only for now — everyone's document history
    exists, but this is a new surface, so it's gated to admins the same way new
    features already start in this codebase (see ADMIN_EMAILS / the KB write
    endpoints) until it's proven out."""
    token = _bearer_token(authorization)
    conn = _open_db()
    try:
        c = conn.cursor()
        _require_admin(c, token)
        user_id = _user_id_for_token(c, token)

        limit = max(1, min(limit, 25))
        q_trimmed = q.strip()
        if q_trimmed:
            c.execute(
                """
                SELECT d.id, d.filename, d.upload_timestamp,
                       (SELECT COUNT(*) FROM document_chunks WHERE document_id = d.id) AS chunks,
                       di.doc_type
                FROM documents d
                LEFT JOIN document_insights di ON di.document_id = d.id
                WHERE d.user_id = %s AND d.filename ILIKE %s
                ORDER BY d.upload_timestamp DESC
                LIMIT %s
                """,
                (user_id, f"%{q_trimmed}%", limit),
            )
        else:
            # Empty query — show the user's most recent uploads rather than
            # nothing, so opening the search bar isn't a blank state.
            c.execute(
                """
                SELECT d.id, d.filename, d.upload_timestamp,
                       (SELECT COUNT(*) FROM document_chunks WHERE document_id = d.id) AS chunks,
                       di.doc_type
                FROM documents d
                LEFT JOIN document_insights di ON di.document_id = d.id
                WHERE d.user_id = %s
                ORDER BY d.upload_timestamp DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
        rows = c.fetchall()
        return [
            {"id": r[0], "filename": r[1], "upload_timestamp": r[2], "chunks": r[3], "doc_type": r[4]}
            for r in rows
        ]
    except HTTPException:
        raise
    except Exception as e:
        print(f"[search_my_documents] failed: {type(e).__name__}: {e}")
        raise HTTPException(500, "Could not search documents")
    finally:
        conn.close()


@app.get("/documents/{document_id}/content")
async def get_document_content(document_id: int, authorization: Optional[str] = Header(None)):
    """Fetch one document's stored extracted text by id, so a search result
    can be opened without re-uploading the original file. Admin-only, and
    scoped to the caller's own documents — matches search_my_documents above."""
    token = _bearer_token(authorization)
    conn = _open_db()
    try:
        c = conn.cursor()
        _require_admin(c, token)
        user_id = _user_id_for_token(c, token)

        c.execute(
            "SELECT filename, content, upload_timestamp FROM documents WHERE id = %s AND user_id = %s",
            (document_id, user_id),
        )
        doc = c.fetchone()
        if not doc:
            raise HTTPException(404, "Document not found")
        return {"id": document_id, "filename": doc[0], "content": doc[1] or "", "upload_timestamp": doc[2]}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[get_document_content] failed: {type(e).__name__}: {e}")
        raise HTTPException(500, "Could not load the document")
    finally:
        conn.close()


@app.delete("/me/documents/{document_id}")
async def delete_my_document(document_id: int, authorization: Optional[str] = Header(None)):
    """Delete one of the caller's own documents, by id.

    Replaces the old unauthenticated ``DELETE /documents/delete`` (filename in
    the body), which let anyone delete any user's document by knowing or
    guessing its filename, and picked an arbitrary row when filenames were
    duplicated. Here the caller comes from the Authorization header, and the
    delete is scoped ``WHERE id = ... AND user_id = ...`` in one transaction, so
    a document that doesn't exist and one that belongs to someone else are
    indistinguishable (both 404) and nothing is touched in either case.

    The document's chunks are removed in the same transaction. Its
    document_insights row goes with it via ON DELETE CASCADE. knowledge_base
    entries are not linked to a document and are deliberately left alone.
    """
    token = _bearer_token(authorization)
    conn = _open_db()
    try:
        c = conn.cursor()
        user_id = _user_id_for_token(c, token)

        c.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id IN (SELECT id FROM documents WHERE id = %s AND user_id = %s)
            """,
            (document_id, user_id),
        )
        chunks_deleted = c.rowcount
        c.execute(
            "DELETE FROM documents WHERE id = %s AND user_id = %s RETURNING id, filename",
            (document_id, user_id),
        )
        row = c.fetchone()
        if not row:
            conn.rollback()
            raise HTTPException(404, "Document not found")
        conn.commit()
        return {"deleted_id": row[0], "filename": row[1], "chunks_deleted": chunks_deleted}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"[delete_my_document] failed: {type(e).__name__}: {e}")
        raise HTTPException(500, "Could not delete the document")
    finally:
        conn.close()


@app.put("/put_ratings")
async def put_ratings(request: RatingRequest):
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("UPDATE chat_history SET rating = %s, comment = %s WHERE id = %s", (request.rating, request.comment, request.id))
        conn.commit()
        conn.close()
        return {"message": "Rating updated", "id": request.id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/register")
async def register(request: RegisterRequest):
    # Reject malformed addresses and placeholder/nonexistent domains outright —
    # this checks the address is well-formed AND that its domain actually has
    # mail servers configured (an MX/DNS lookup), without emailing anything.
    # Full ownership verification (click-to-confirm) is a separate, later step.
    try:
        validated = validate_email(request.email, check_deliverability=True)
        normalized_email = validated.normalized.lower()
    except EmailNotValidError as e:
        raise HTTPException(400, f"Please enter a real, working email address ({e})")

    conn = connect_to_postgres()
    c = conn.cursor()
    # Case-insensitive: "User@Gmail.com" and "user@gmail.com" are the same
    # account for registration purposes, so one person can't re-register by
    # varying case.
    c.execute("SELECT id FROM users WHERE username = %s OR LOWER(email) = %s", (request.username, normalized_email))
    if c.fetchone():
        conn.close()
        raise HTTPException(400, "Username or email already exists")
    hashed = await hash_password(request.password)
    token = secrets.token_hex(32)

    # New accounts start unverified — email_verified defaults to 1 at the
    # column level (so pre-existing accounts are unaffected), but a fresh
    # registration explicitly sets 0 and gets a time-limited verification
    # link. /login refuses to issue a session until that link is clicked.
    verification_token = secrets.token_urlsafe(32)
    verification_expires = (datetime.datetime.now() + datetime.timedelta(hours=24)).isoformat()

    # Pre-authorized admins (ADMIN_EMAILS env var) get is_admin=1 right at
    # creation. Only checked here, at registration — an email added to the
    # list later doesn't touch accounts that already exist.
    is_admin_flag = 1 if normalized_email in ADMIN_EMAILS else 0

    c.execute(
        """
        INSERT INTO users (username, password, token, email, email_verified, email_verification_token, email_verification_expires, is_admin)
        VALUES (%s, %s, %s, %s, 0, %s, %s, %s)
        """,
        (request.username, hashed, token, normalized_email, verification_token, verification_expires, is_admin_flag),
    )
    conn.commit()
    conn.close()

    verify_link = f"{FRONTEND_URL}/#verify-email?token={verification_token}"
    params = {
    "from": "Synerge <no-reply@synergereader.ai>",
    "to": [normalized_email],
    "subject": "Verify your SynergeReader account",
    "html": f"""
        <h2>Welcome to SynergeReader!</h2>
        <p>One last step — confirm this is your email address to activate your account.</p>
        <p><a href="{verify_link}">Verify my email</a></p>
        <p>Or paste this link into your browser:<br>{verify_link}</p>
        <p style="color:#888;font-size:13px">This link expires in 24 hours. If you didn't create this account, you can ignore this email.</p>
    """
    }

    if resend.api_key:
        try:
            resend.Emails.send(params)
        except Exception as e:
            print(f"DEBUG: verification email failed to send: {e}")
    else:
        # No email provider configured (local/dev) — the link still works,
        # it just isn't delivered anywhere. Printed so it's reachable for
        # manual testing without a real inbox.
        print(f"DEBUG: no EMAIL_KEY configured — verification link for {normalized_email}: {verify_link}")

    # Deliberately not returning `token` here — a session token issued before
    # the account is verified would work against every other endpoint that
    # only checks token validity (none of them currently also check
    # email_verified). The frontend never used this value anyway; it sends
    # the user to "check your email" instead of logging them in.
    return {"message": "Registered — check your email to verify your account before logging in."}


@app.get("/verify-email")
async def verify_email(token: str):
    conn = connect_to_postgres()
    c = conn.cursor()
    c.execute(
        "SELECT id, email_verification_expires FROM users WHERE email_verification_token = %s",
        (token,),
    )
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(400, "This verification link is invalid or has already been used.")

    user_id, expires = row
    if expires and datetime.datetime.fromisoformat(expires) < datetime.datetime.now():
        conn.close()
        raise HTTPException(400, "This verification link has expired. Request a new one from the sign-in screen.")

    c.execute(
        "UPDATE users SET email_verified = 1, email_verification_token = NULL, email_verification_expires = NULL WHERE id = %s",
        (user_id,),
    )
    conn.commit()
    conn.close()
    return {"message": "Email verified — you can now log in."}


@app.post("/resend-verification")
async def resend_verification(request: ResendVerificationRequest):
    """Always returns the same generic message regardless of whether the
    email exists or is already verified — mirrors /forgot-password's
    anti-enumeration stance, for the same reason."""
    conn = connect_to_postgres()
    c = conn.cursor()
    normalized = request.email.strip().lower()
    c.execute("SELECT id, email_verified FROM users WHERE LOWER(email) = %s", (normalized,))
    row = c.fetchone()

    if row and row[1] is not None and not row[1]:
        user_id = row[0]
        verification_token = secrets.token_urlsafe(32)
        verification_expires = (datetime.datetime.now() + datetime.timedelta(hours=24)).isoformat()
        c.execute(
            "UPDATE users SET email_verification_token = %s, email_verification_expires = %s WHERE id = %s",
            (verification_token, verification_expires, user_id),
        )
        conn.commit()
        verify_link = f"{FRONTEND_URL}/#verify-email?token={verification_token}"
        if resend.api_key:
            try:
                resend.Emails.send({
                    "from": "Synerge <no-reply@synergereader.ai>",
                    "to": [normalized],
                    "subject": "Verify your SynergeReader account",
                    "html": f'<p><a href="{verify_link}">Verify my email</a></p><p>Or paste this link into your browser:<br>{verify_link}</p>',
                })
            except Exception as e:
                print(f"DEBUG: resend-verification email failed to send: {e}")
        else:
            print(f"DEBUG: no EMAIL_KEY configured — verification link for {normalized}: {verify_link}")

    conn.close()
    return {"message": "If that email exists and isn't verified yet, a new verification link has been sent."}


@app.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Always returns the same generic message whether or not the email is
    registered — the old behavior (400 'User not found') let anyone probe
    which emails have accounts. Also now sends a time-limited reset *link*
    instead of emailing a freshly-generated plaintext password."""
    conn = connect_to_postgres()
    c = conn.cursor()
    normalized = request.email.strip().lower()
    c.execute("SELECT id FROM users WHERE LOWER(email) = %s", (normalized,))
    row = c.fetchone()

    if row:
        user_id = row[0]
        reset_token = secrets.token_urlsafe(32)
        reset_expires = (datetime.datetime.now() + datetime.timedelta(hours=1)).isoformat()
        c.execute(
            "UPDATE users SET password_reset_token = %s, password_reset_expires = %s WHERE id = %s",
            (reset_token, reset_expires, user_id),
        )
        conn.commit()
        reset_link = f"{FRONTEND_URL}/#reset-password?token={reset_token}"
        params = {
            "from": "Synerge <no-reply@synergereader.ai>",
            "to": [normalized],
            "subject": "Reset your SynergeReader password",
            "html": f"""
                <p>Click below to choose a new password. This link expires in 1 hour.</p>
                <p><a href="{reset_link}">Reset my password</a></p>
                <p>Or paste this link into your browser:<br>{reset_link}</p>
                <p style="color:#888;font-size:13px">If you didn't request this, you can safely ignore this email — your password won't change.</p>
            """,
        }
        if resend.api_key:
            try:
                resend.Emails.send(params)
            except Exception as e:
                print(f"DEBUG: forgot-password email failed to send: {e}")
        else:
            print(f"DEBUG: no EMAIL_KEY configured — reset link for {normalized}: {reset_link}")

    conn.close()
    return {"message": "If that email is registered, we've sent a password reset link."}


@app.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    if len(request.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    conn = connect_to_postgres()
    c = conn.cursor()
    c.execute(
        "SELECT id, password_reset_expires FROM users WHERE password_reset_token = %s",
        (request.token,),
    )
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(400, "This reset link is invalid or has already been used.")

    user_id, expires = row
    if expires and datetime.datetime.fromisoformat(expires) < datetime.datetime.now():
        conn.close()
        raise HTTPException(400, "This reset link has expired. Request a new one from the sign-in screen.")

    hashed = await hash_password(request.new_password)
    # A successfully-used reset link is itself proof the person controls the
    # inbox, so this also marks the email verified — closes the loop for an
    # account that registered but never clicked the original verify link.
    c.execute(
        "UPDATE users SET password = %s, password_reset_token = NULL, password_reset_expires = NULL, email_verified = 1 WHERE id = %s",
        (hashed, user_id),
    )
    conn.commit()
    conn.close()
    return {"message": "Password updated — you can now log in with your new password."}


@app.post("/login")
async def login(request: LoginRequest):
    conn = connect_to_postgres()
    c = conn.cursor()
    # The single "username" field accepts either the actual username or the
    # account's email (case-insensitive, matching how it's normalized at
    # registration) — most people try to log in with whichever one they
    # remember, and there's no reason to force just one.
    #
    # Username is checked first, exact match, on its own — only if that
    # finds nothing do we fall back to matching on email. A single combined
    # "OR" query can't be trusted here: some old rows have garbage/legacy
    # values in the email column (pre-dating email validation) that can
    # coincidentally match someone else's real username, and an OR query
    # has no defined way to prefer one match over the other.
    c.execute("SELECT password, token, is_active, email_verified FROM users WHERE username = %s", (request.username,))
    row = c.fetchone()
    if not row:
        c.execute("SELECT password, token, is_active, email_verified FROM users WHERE LOWER(email) = LOWER(%s)", (request.username,))
        row = c.fetchone()
    conn.close()
    if not row or not bcrypt.checkpw(request.password.encode(), row[0].encode()):
        raise HTTPException(400, "Invalid username or password")
    if row[2] is not None and not row[2]:
        raise HTTPException(403, "This account has been suspended. Contact an administrator.")
    if row[3] is not None and not row[3]:
        raise HTTPException(403, "Please verify your email before logging in — check your inbox, or resend the verification email.")
    return {"message": "Login successful", "token": row[1]}


@app.post("/google-login")
async def google_login(request: GoogleLoginRequest):
    """
    Google OAuth 2.0 Login Endpoint

    Flow:
    1. Frontend sends Google ID token
    2. Backend verifies token with Google
    3. Backend extracts user email/name from token
    4. Backend creates/updates user in database
    5. Backend returns app token
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "Google Client ID not configured")

    try:
        # Verify the token with Google
        # This checks that the token is valid and hasn't been tampered with
        idinfo = id_token.verify_oauth2_token(
            request.token, google_requests.Request(), GOOGLE_CLIENT_ID
        )

        # Extract user information from the verified token
        email = idinfo.get("email", "")
        name = idinfo.get("name", "")
        google_id = idinfo.get("sub", "")  # Unique Google user ID

        if not email:
            raise HTTPException(400, "Email not provided by Google")

        # Connect to database
        conn = connect_to_postgres()
        c = conn.cursor()

        # Check if user already exists
        c.execute("SELECT id, token FROM users WHERE username = %s", (email,))
        row = c.fetchone()

        if row:
            # User exists, return their token
            conn.close()
            return {
                "message": "Login successful",
                "token": row[1],
                "email": email,
                "name": name,
            }
        else:
            # Create new user with Google email as username
            # Password is not needed for Google users, we can use a placeholder
            app_token = secrets.token_hex(32)
            placeholder_password = secrets.token_hex(32)  # Random, unused password

            c.execute("INSERT INTO users (username, password, token) VALUES (%s, %s, %s)", (email, placeholder_password, app_token))
            conn.commit()
            conn.close()

            return {
                "message": "Registration and login successful",
                "token": app_token,
                "email": email,
                "name": name,
            }

    except ValueError as e:
        # Token verification failed
        raise HTTPException(401, f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Google login error: {str(e)}")

@app.post("/submit_correction")
async def submit_correction(request: CorrectionRequest):
    try:
        conn = connect_to_postgres()
        c = conn.cursor()

        # Get original question and answer
        c.execute("SELECT question, answer FROM chat_history WHERE id = %s", (request.chat_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, "Chat ID not found")

        question, original_answer = row

        # Update chat history
        c.execute("UPDATE chat_history SET answer = %s, comment = %s WHERE id = %s", (request.corrected_answer, request.comment, request.chat_id))

        # Insert into knowledge base
        # Embed the question for semantic matching. A failure here rolls
        # back the chat_history update above too, rather than persisting a
        # NULL-embedding KB row that can never be found by semantic search.
        try:
            q_vec = _EMBEDDING_PROVIDER.embed_documents([question])[0]
        except EmbeddingProviderError as exc:
            conn.rollback()
            conn.close()
            print(f"[Correction] Embedding failed for chat_id={request.chat_id}: {type(exc).__name__}")
            raise HTTPException(503, "Could not save correction: embedding service is temporarily unavailable")

        c.execute(
            "INSERT INTO knowledge_base (question, original_answer, corrected_answer, created_at, chat_history_id, corrected_by, usage_count, embedding) VALUES (%s, %s, %s, %s, %s, %s, 0, %s)",
            (question, original_answer, request.corrected_answer, datetime.datetime.now().isoformat(), request.chat_id, getattr(request, 'corrected_by', 'User'), q_vec)
        )

        conn.commit()
        conn.close()
        return {
            "message": "Correction submitted and saved to KB",
            "chat_id": request.chat_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/knowledge_base")
async def knowledge_base():
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("""SELECT id, question, original_answer, corrected_answer, created_at,
                            chat_history_id, corrected_by, COALESCE(usage_count,0), context_text,
                            COALESCE(source_type, 'document')
                     FROM knowledge_base ORDER BY COALESCE(usage_count,0) DESC, id DESC""")
        rows = c.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "question": r[1],
                "original_answer": r[2] or "",
                "answer": r[3],
                "created_at": r[4],
                "chat_history_id": r[5],
                "corrected_by": r[6] or "User",
                "usage_count": r[7],
                "context_text": r[8] or "",
                "source_type": r[9],
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/knowledge_base")
async def add_knowledge(request: KnowledgeInsertRequest):
    """Add knowledge items directly — used for a single manual entry from the
    KB page, and for batch imports of Q&A pairs from an external file.
    Admin-only: this is the shared, org-wide knowledge base every user's
    answers draw on, and the UI already only shows these actions to admins —
    this just makes the backend actually enforce that instead of trusting
    the frontend to hide the buttons."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, request.token)
        c.execute("SELECT username FROM users WHERE token = %s", (request.token,))
        row = c.fetchone()
        conn.close()
        author = row[0] if row else None

        source_type = request.source_type if request.source_type in ("manual", "external_import") else "manual"
        label = "Manual Entry" if source_type == "manual" else "Imported batch"
        if author:
            label += f" by {author}"

        pairs = [{"question": item.question, "answer": item.answer} for item in request.items]
        # Preserve each item's own "source" (e.g. filename of the imported file)
        # in context_text when present; fall back to a shared label otherwise.
        conn = connect_to_postgres()
        c = conn.cursor()
        saved = 0
        for item in request.items:
            if not item.question.strip() or not item.answer.strip():
                continue
            try:
                q_vec = _EMBEDDING_PROVIDER.embed_documents([item.question])[0]
            except EmbeddingProviderError as exc:
                # Roll back the whole uncommitted batch rather than persist a
                # NULL-embedding row or silently drop just this one item.
                conn.rollback()
                print(f"[KB] add_knowledge embedding failed; rolled back {saved} pending row(s): {type(exc).__name__}")
                raise HTTPException(503, "Could not add knowledge items: embedding service is temporarily unavailable")
            c.execute(
                """
                INSERT INTO knowledge_base
                  (question, original_answer, corrected_answer, created_at,
                   context_text, corrected_by, usage_count, embedding, source_type)
                VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s)
                """,
                (item.question, "", item.answer, datetime.datetime.now().isoformat(),
                 item.source or "", label, q_vec, source_type),
            )
            saved += 1
        conn.commit()
        conn.close()
        return {"message": f"{saved} knowledge item(s) added", "added": saved}
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


def _validate_external_url(url: str) -> None:
    """Basic SSRF guard for the URL importer: only plain http(s) URLs whose
    hostname resolves to a public address are allowed — no fetching internal
    services, loopback, link-local, or cloud metadata endpoints."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Only http:// or https:// URLs are supported")
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(400, "That doesn't look like a valid URL")
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise HTTPException(400, "Could not resolve that host")
    for family, _, _, _, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise HTTPException(400, "This URL points to a private or internal address and can't be imported")


def _safe_fetch_url(url: str, max_redirects: int = 3):
    """Fetch a URL the SSRF-safe way. requests.get(url, allow_redirects=True)
    (the default) only validates the URL you hand it — not where a 3xx
    response actually takes it next. A URL that resolves publicly can still
    redirect to a private/internal address or a cloud metadata endpoint, and
    requests would follow it with no further check. So redirects are
    followed manually here, re-running the same hostname/IP validation
    before every single hop."""
    current_url = url
    for _ in range(max_redirects + 1):
        _validate_external_url(current_url)
        resp = requests.get(
            current_url,
            timeout=(5, 15),
            headers={"User-Agent": "SynergeReader-KB-Importer/1.0"},
            stream=True,
            allow_redirects=False,
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            if not location:
                raise HTTPException(400, "That URL redirected without a destination")
            current_url = urljoin(current_url, location)
            continue
        return resp
    raise HTTPException(400, "Too many redirects")


def _extract_readable_text(html: str) -> str:
    try:
        doc = lxml_html.fromstring(html)
    except Exception:
        return ""
    for tag in doc.xpath("//script | //style | //nav | //footer | //header | //aside | //noscript"):
        tag.drop_tree()
    text = doc.text_content()
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


@app.post("/knowledge_base/import_url")
async def import_knowledge_from_url(request: KnowledgeUrlImportRequest):
    """Pull an external web page into the Knowledge Base: fetch it, extract
    readable text, and generate Q&A pairs from it the same way a freshly
    uploaded document is seeded — so external sources become first-class,
    searchable KB entries instead of just a link. Admin-only — see the note
    on POST /knowledge_base above."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, request.token)
        c.execute("SELECT username FROM users WHERE token = %s", (request.token,))
        row = c.fetchone()
        author = row[0] if row else None
    finally:
        conn.close()

    try:
        resp = _safe_fetch_url(request.url)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raise HTTPException(400, f"That URL returned '{content_type or 'unknown content'}', not a web page")
        raw = resp.raw.read(2_000_000, decode_content=True)  # cap at 2MB
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        raise HTTPException(400, f"Could not fetch that URL: {e}")

    text = _extract_readable_text(raw.decode("utf-8", errors="ignore"))
    if len(text) < 200:
        raise HTTPException(400, "Could not extract enough readable text from that page")

    domain = urlparse(request.url).hostname or request.url
    label = f"Imported from {domain}" + (f" by {author}" if author else "")

    pairs = _generate_qa_pairs_from_text(text, domain)
    if not pairs:
        raise HTTPException(422, "Could not generate any question-answer pairs from that page's content")

    saved = _save_kb_pairs(pairs, request.url, "external_url", label)
    if saved == 0:
        raise HTTPException(
            422,
            "Nothing was saved — the generated pairs may have all been too "
            "short or duplicates, or the embedding service may be "
            "temporarily unavailable",
        )

    return {"message": f"{saved} knowledge entries added from {domain}", "added": saved, "source": request.url}


@app.delete("/knowledge_base/{entry_id}")
async def delete_knowledge(entry_id: int, token: Optional[str] = None):
    """Delete a knowledge base entry by ID. Admin-only — see the note on
    POST /knowledge_base above."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)
        c.execute("DELETE FROM knowledge_base WHERE id = %s RETURNING id", (entry_id,))
        deleted = c.fetchone()
        conn.commit()
        conn.close()
        if not deleted:
            raise HTTPException(404, "Entry not found")
        return {"message": f"Entry {entry_id} deleted"}
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


class KBUpdateRequest(BaseModel):
    question: str
    answer: str
    token: Optional[str] = None


@app.put("/knowledge_base/{entry_id}")
async def update_knowledge(entry_id: int, request: KBUpdateRequest):
    """Edit a knowledge base entry. Admin-only — see the note on
    POST /knowledge_base above."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, request.token)
        # Re-embed if question changed
        try:
            q_vec = _EMBEDDING_PROVIDER.embed_documents([request.question])[0]
        except EmbeddingProviderError as exc:
            conn.rollback()
            print(f"[KB] update_knowledge embedding failed for entry {entry_id}: {type(exc).__name__}")
            raise HTTPException(503, "Could not update knowledge item: embedding service is temporarily unavailable")
        c.execute(
            "UPDATE knowledge_base SET question=%s, corrected_answer=%s, embedding=%s WHERE id=%s RETURNING id",
            (request.question, request.answer, q_vec, entry_id)
        )
        updated = c.fetchone()
        conn.commit()
        conn.close()
        if not updated:
            raise HTTPException(404, "Entry not found")
        return {"message": f"Entry {entry_id} updated"}
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.get("/admin/check")
async def check_admin_status(token: Optional[str] = None):
    """Check if user with given token is admin"""
    if not token:
        return {"is_admin": False}

    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE token = %s", (token,))
        row = c.fetchone()
        conn.close()

        if row:
            return {"is_admin": bool(row[0])}
        return {"is_admin": False}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/admin/ratings")
async def get_all_ratings(token: Optional[str] = None):
    """Get all ratings and feedback from responses"""
    if not token:
        raise HTTPException(401, "Unauthorized")

    try:
        conn = connect_to_postgres()
        c = conn.cursor()

        # Check if user is admin
        c.execute("SELECT is_admin FROM users WHERE token = %s", (token,))
        row = c.fetchone()

        if not row or not row[0]:
            conn.close()
            raise HTTPException(403, "Forbidden: Admin access required")

        # Fetch all ratings from chat history
        c.execute("""
            SELECT 
                ch.id,
                ch.user_id,
                u.username,
                ch.ts,
                ch.question,
                ch.answer,
                ch.rating,
                ch.comment,
                ch.selected_text
            FROM chat_history ch
            LEFT JOIN users u ON ch.user_id = u.id
            WHERE ch.rating IS NOT NULL
            ORDER BY ch.ts DESC
        """)

        rows = c.fetchall()
        conn.close()

        ratings = []
        for row in rows:
            ratings.append(
                {
                    "id": row[0],
                    "user_id": row[1],
                    "username": row[2] or "Anonymous",
                    "timestamp": row[3],
                    "question": row[4],
                    "answer": row[5],
                    "rating": row[6],
                    "comment": row[7],
                    "selected_text": row[8],
                }
            )

        return {"ratings": ratings, "total_count": len(ratings)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/admin/ratings/stats")
async def get_rating_stats(token: Optional[str] = None):
    """Get statistics about ratings"""
    if not token:
        raise HTTPException(401, "Unauthorized")

    try:
        conn = connect_to_postgres()
        c = conn.cursor()

        # Check if user is admin
        c.execute("SELECT is_admin FROM users WHERE token = %s", (token,))
        row = c.fetchone()

        if not row or not row[0]:
            conn.close()
            raise HTTPException(403, "Forbidden: Admin access required")

        # Get rating statistics
        c.execute("""
            SELECT 
                COUNT(*) as total_ratings,
                AVG(rating) as average_rating,
                MIN(rating) as min_rating,
                MAX(rating) as max_rating
            FROM chat_history
            WHERE rating IS NOT NULL
        """)

        stats_row = c.fetchone()

        # Get rating distribution
        c.execute("""
            SELECT rating, COUNT(*) as count
            FROM chat_history
            WHERE rating IS NOT NULL
            GROUP BY rating
            ORDER BY rating
        """)

        distribution_rows = c.fetchall()
        conn.close()

        distribution = {}
        for rating, count in distribution_rows:
            distribution[int(rating)] = count

        return {
            "total_ratings": stats_row[0] or 0,
            "average_rating": round(stats_row[1], 2) if stats_row[1] else 0,
            "min_rating": stats_row[2] or 0,
            "max_rating": stats_row[3] or 0,
            "distribution": distribution,
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/me")
async def get_me(token: Optional[str] = None):
    """Resolve the current session's user from an auth token, for the frontend to
    restore login state and decide whether to show the admin dashboard."""
    if not token:
        raise HTTPException(401, "Unauthorized")
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("SELECT username, email, is_admin, is_active FROM users WHERE token = %s", (token,))
        row = c.fetchone()
        conn.close()
        if not row:
            raise HTTPException(401, "Invalid session")
        if row[3] is not None and not row[3]:
            raise HTTPException(403, "This account has been suspended.")
        return {"username": row[0], "email": row[1], "is_admin": bool(row[2])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/me/stats")
async def get_my_stats(token: Optional[str] = None, days: int = 7):
    """Personal usage overview for the signed-in user's own Dashboard — scoped
    entirely to their own documents/chats, never other users' data. No admin
    access required; a user can only ever see their own numbers here."""
    if not token:
        raise HTTPException(401, "Unauthorized")
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("SELECT id, username FROM users WHERE token = %s", (token,))
        row = c.fetchone()
        if not row:
            raise HTTPException(401, "Invalid session")
        user_id, username = row[0], row[1]

        days = max(7, min(days, 90))
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=days - 1)

        c.execute("SELECT COUNT(*) FROM documents WHERE user_id = %s", (user_id,))
        total_documents = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM chat_history WHERE user_id = %s", (user_id,))
        total_chats = c.fetchone()[0]

        c.execute(
            "SELECT COUNT(*), AVG(rating) FROM chat_history WHERE user_id = %s AND rating IS NOT NULL",
            (user_id,),
        )
        rated_row = c.fetchone()
        rated_count = rated_row[0] or 0
        average_rating = round(float(rated_row[1]), 2) if rated_row[1] is not None else None

        last7_key = (now - datetime.timedelta(days=7)).date().isoformat()
        c.execute("SELECT COUNT(*) FROM chat_history WHERE user_id = %s AND ts >= %s", (user_id, last7_key))
        chats_last_7d = c.fetchone()[0]

        # daily volume, zero-filled, scoped to this user only
        start_key = start.date().isoformat()
        c.execute(
            "SELECT LEFT(ts, 10) AS day, COUNT(*) FROM chat_history WHERE user_id = %s AND ts >= %s GROUP BY day",
            (user_id, start_key),
        )
        counts_by_day = {r[0]: r[1] for r in c.fetchall()}
        chats_per_day = [
            {"date": (start + datetime.timedelta(days=i)).date().isoformat(),
             "count": counts_by_day.get((start + datetime.timedelta(days=i)).date().isoformat(), 0)}
            for i in range(days)
        ]

        # usage by task mode, scoped to this user. NOTE: a params tuple is passed
        # here (unlike admin_analytics' equivalent query), so psycopg2 will do
        # %s-style substitution on the whole string — every literal '%' in the
        # LIKE patterns below must be escaped as '%%' or it's misread as a
        # placeholder ("tuple index out of range").
        case_clauses = " ".join(
            f"WHEN ch.question LIKE '{prefix.replace(chr(39), chr(39)*2)}%%' THEN '{label}'"
            for label, prefix in TASK_MODE_PREFIXES.items()
        )
        suggested_escaped = SUGGESTED_QUESTIONS_PREFIX.replace("'", "''")
        c.execute(
            f"""
            SELECT
                CASE
                    {case_clauses}
                    WHEN ch.question LIKE '{suggested_escaped}%%' THEN 'Suggested Questions (system)'
                    ELSE 'Research & Q&A'
                END AS bucket,
                COUNT(*)
            FROM chat_history ch
            WHERE ch.user_id = %s
            GROUP BY bucket
            ORDER BY COUNT(*) DESC
            """,
            (user_id,),
        )
        task_mode_usage = [{"label": r[0], "count": r[1]} for r in c.fetchall() if r[0] != "Suggested Questions (system)"]

        c.execute(
            "SELECT id, ts, question, rating FROM chat_history WHERE user_id = %s ORDER BY id DESC LIMIT 6",
            (user_id,),
        )
        recent_chats = [{"id": r[0], "ts": r[1], "question": r[2], "rating": r[3]} for r in c.fetchall()]

        c.execute("SELECT COUNT(*) FROM knowledge_base")
        kb_total = c.fetchone()[0]

        # This user's own documents, with their Document Insights result if the
        # background extraction has finished — lets a member see what the AI
        # pulled out of their own upload, a view previously admin-only.
        c.execute(
            """
            SELECT d.id, d.filename, d.upload_timestamp,
                   (SELECT COUNT(*) FROM document_chunks WHERE document_id = d.id) AS chunks,
                   di.doc_type, di.keywords
            FROM documents d
            LEFT JOIN document_insights di ON di.document_id = d.id
            WHERE d.user_id = %s
            ORDER BY d.upload_timestamp DESC
            LIMIT 6
            """,
            (user_id,),
        )
        my_documents = [
            {
                "id": r[0], "filename": r[1], "upload_timestamp": r[2], "chunks": r[3],
                "doc_type": r[4], "keywords": (r[5] or [])[:4],
            }
            for r in c.fetchall()
        ]

        conn.close()
        return {
            "username": username,
            "total_documents": total_documents,
            "total_chats": total_chats,
            "rated_count": rated_count,
            "average_rating": average_rating,
            "chats_last_7d": chats_last_7d,
            "chats_per_day": chats_per_day,
            "task_mode_usage": task_mode_usage,
            "recent_chats": recent_chats,
            "kb_total": kb_total,
            "my_documents": my_documents,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


def _require_admin(cursor, token: Optional[str]) -> None:
    if not token:
        raise HTTPException(401, "Unauthorized")
    cursor.execute("SELECT is_admin FROM users WHERE token = %s", (token,))
    row = cursor.fetchone()
    if not row or not row[0]:
        raise HTTPException(403, "Forbidden: Admin access required")


def _log_admin_action(cursor, token: Optional[str], action: str, target_id=None, target_username=None, detail=None):
    """Record an admin action (promote/demote/suspend/reactivate/delete) for the
    audit trail — this is a compliance feature, so it never blocks the caller if
    logging itself fails."""
    try:
        actor_id, actor_username = None, None
        if token:
            cursor.execute("SELECT id, username FROM users WHERE token = %s", (token,))
            row = cursor.fetchone()
            if row:
                actor_id, actor_username = row[0], row[1]
        cursor.execute(
            """
            INSERT INTO admin_audit_log (ts, actor_id, actor_username, action, target_id, target_username, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (datetime.datetime.now().isoformat(), actor_id, actor_username, action, target_id, target_username, detail),
        )
    except Exception as e:
        print(f"DEBUG: failed to write audit log entry: {e}")


# Task-mode system-prompt prefixes — kept in sync with TASK_PROMPTS in the
# frontend (GridApp.jsx). A stored question either starts with exactly one of
# these (task mode was active) or none (plain Research & Q&A, or an internal
# suggested-questions call, detected separately below).
TASK_MODE_PREFIXES = {
    "Argument Generator": "You are a legal research assistant. Structure every argument in IRAC format",
    "Risk Analysis":      "You are a legal risk analyst. Identify ambiguous language",
    "Clause Extractor":   "You are a contract analysis assistant. Extract the requested clause type precisely",
    "Summarize":          "You are a legal document analyst. Provide a structured summary",
    "Related Precedents": "You are a legal research assistant. Identify and explain relevant legal precedents",
}
SUGGESTED_QUESTIONS_PREFIX = "Generate exactly 4 specific questions a lawyer would ask"


@app.get("/admin/overview")
async def admin_overview(token: Optional[str] = None):
    """Summary stats for the admin dashboard's overview cards."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM documents")
        total_documents = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM chat_history")
        total_chats = c.fetchone()[0]

        c.execute("SELECT COUNT(*) FROM knowledge_base")
        total_kb_entries = c.fetchone()[0]

        c.execute("SELECT COUNT(*), AVG(rating) FROM chat_history WHERE rating IS NOT NULL")
        rated_count, avg_rating = c.fetchone()

        conn.close()
        return {
            "total_users": total_users,
            "total_documents": total_documents,
            "total_chats": total_chats,
            "total_kb_entries": total_kb_entries,
            "rated_count": rated_count or 0,
            "average_rating": round(avg_rating, 2) if avg_rating else 0,
        }
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.get("/admin/chat_history")
async def admin_chat_history(
    token: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    search: Optional[str] = None,
    user_id: Optional[str] = None,
    rating: Optional[str] = None,     # "1".."5", "unrated", or omitted for all
    since_days: Optional[int] = None, # e.g. 1, 7, 30 — chats from the last N days
):
    """Full chat history across every user, for the admin dashboard's history table."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        where_clauses = []
        params: list = []
        if search:
            where_clauses.append("(ch.question ILIKE %s OR ch.answer ILIKE %s)")
            like = f"%{search}%"
            params.extend([like, like])
        if user_id:
            where_clauses.append("ch.user_id = %s")
            params.append(user_id)
        if rating == "unrated":
            where_clauses.append("ch.rating IS NULL")
        elif rating and rating.isdigit():
            where_clauses.append("ch.rating = %s")
            params.append(int(rating))
        if since_days:
            cutoff = (datetime.datetime.now() - datetime.timedelta(days=since_days)).date().isoformat()
            where_clauses.append("ch.ts >= %s")
            params.append(cutoff)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        c.execute(f"SELECT COUNT(*) FROM chat_history ch {where_sql}", params)
        total_count = c.fetchone()[0]

        c.execute(
            f"""
            SELECT ch.id, ch.user_id, u.username, ch.ts, ch.question, ch.answer, ch.rating, ch.comment
            FROM chat_history ch
            LEFT JOIN users u ON ch.user_id = u.id
            {where_sql}
            ORDER BY ch.id DESC
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        rows = c.fetchall()
        conn.close()

        return {
            "items": [
                {
                    "id": r[0],
                    "user_id": r[1],
                    "username": r[2] or "Anonymous",
                    "timestamp": r[3],
                    "question": r[4],
                    "answer": r[5],
                    "rating": r[6],
                    "comment": r[7],
                }
                for r in rows
            ],
            "total_count": total_count,
        }
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


def _zero_filled_daily_series(cursor, table: str, ts_col: str, days: int, start: datetime.datetime) -> list:
    start_key = start.date().isoformat()
    cursor.execute(
        f"SELECT LEFT({ts_col}, 10) AS day, COUNT(*) FROM {table} WHERE {ts_col} >= %s GROUP BY day",
        (start_key,),
    )
    counts_by_day = {row[0]: row[1] for row in cursor.fetchall()}
    series = []
    for i in range(days):
        day = (start + datetime.timedelta(days=i)).date().isoformat()
        series.append({"date": day, "count": counts_by_day.get(day, 0)})
    return series


WEEKDAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
WEEKDAY_FULL_PLURAL = ["Sundays", "Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays"]


@app.get("/admin/analytics")
async def admin_analytics(token: Optional[str] = None, days: int = 14):
    """Chart data for the admin dashboard: daily volume across chats/documents/KB
    growth, rating distribution, task-mode usage, an hour-by-weekday activity
    heatmap, the most active users, and week-over-week volume for the trend delta."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        days = max(7, min(days, 90))
        now = datetime.datetime.now()
        start = now - datetime.timedelta(days=days - 1)

        chats_per_day = _zero_filled_daily_series(c, "chat_history", "ts", days, start)
        documents_per_day = _zero_filled_daily_series(c, "documents", "upload_timestamp", days, start)
        kb_growth_per_day = _zero_filled_daily_series(c, "knowledge_base", "created_at", days, start)

        c.execute(
            "SELECT rating, COUNT(*) FROM chat_history WHERE rating IS NOT NULL GROUP BY rating"
        )
        dist_rows = c.fetchall()
        rating_distribution = {str(i): 0 for i in range(1, 6)}
        for rating, cnt in dist_rows:
            rating_distribution[str(int(rating))] = cnt

        c.execute(
            """
            SELECT u.username, COUNT(ch.id) AS cnt
            FROM chat_history ch
            JOIN users u ON ch.user_id = u.id
            GROUP BY u.username
            ORDER BY cnt DESC
            LIMIT 6
            """
        )
        top_users = [{"username": r[0], "chat_count": r[1]} for r in c.fetchall()]

        last7_key = (now - datetime.timedelta(days=7)).date().isoformat()
        prev7_key = (now - datetime.timedelta(days=14)).date().isoformat()
        c.execute("SELECT COUNT(*) FROM chat_history WHERE ts >= %s", (last7_key,))
        chats_last_7d = c.fetchone()[0]
        c.execute(
            "SELECT COUNT(*) FROM chat_history WHERE ts >= %s AND ts < %s",
            (prev7_key, last7_key),
        )
        chats_prev_7d = c.fetchone()[0]

        # ── Usage by task mode — classified from the known system-prompt
        # prefixes each task mode prepends to the question (see TASK_MODE_PREFIXES).
        case_clauses = " ".join(
            f"WHEN ch.question LIKE '{prefix.replace(chr(39), chr(39)*2)}%' THEN '{label}'"
            for label, prefix in TASK_MODE_PREFIXES.items()
        )
        suggested_escaped = SUGGESTED_QUESTIONS_PREFIX.replace("'", "''")
        c.execute(
            f"""
            SELECT
                CASE
                    {case_clauses}
                    WHEN ch.question LIKE '{suggested_escaped}%' THEN 'Suggested Questions (system)'
                    ELSE 'Research & Q&A'
                END AS bucket,
                COUNT(*)
            FROM chat_history ch
            GROUP BY bucket
            ORDER BY COUNT(*) DESC
            """
        )
        task_mode_usage = [{"label": r[0], "count": r[1]} for r in c.fetchall()]

        # ── Activity heatmap — every chat, bucketed by weekday × hour.
        c.execute(
            """
            SELECT EXTRACT(DOW FROM ts::timestamp)::int AS dow,
                   EXTRACT(HOUR FROM ts::timestamp)::int AS hr,
                   COUNT(*)
            FROM chat_history
            WHERE ts IS NOT NULL AND ts != ''
            GROUP BY dow, hr
            """
        )
        heatmap_counts = {(int(r[0]), int(r[1])): r[2] for r in c.fetchall()}
        activity_heatmap = [
            {"dow": d, "hour": h, "count": heatmap_counts.get((d, h), 0)}
            for d in range(7) for h in range(24)
        ]
        peak = max(activity_heatmap, key=lambda cell: cell["count"]) if activity_heatmap else None
        peak_insight = None
        if peak and peak["count"] > 0:
            hr = peak["hour"]
            hr_label = f"{hr % 12 or 12}{'AM' if hr < 12 else 'PM'}"
            peak_insight = f"{WEEKDAY_FULL_PLURAL[peak['dow']]} around {hr_label}"

        conn.close()
        return {
            "chats_per_day": chats_per_day,
            "documents_per_day": documents_per_day,
            "kb_growth_per_day": kb_growth_per_day,
            "rating_distribution": rating_distribution,
            "top_users": top_users,
            "chats_last_7d": chats_last_7d,
            "chats_prev_7d": chats_prev_7d,
            "task_mode_usage": task_mode_usage,
            "activity_heatmap": activity_heatmap,
            "peak_insight": peak_insight,
        }
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


class UpdateUserRequest(BaseModel):
    token: str
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None


@app.get("/admin/users")
async def admin_list_users(token: Optional[str] = None):
    """Every user account, with activity counts, for the admin user-management tab."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        c.execute(
            """
            SELECT
                u.id, u.username, u.email, u.is_admin, u.is_active,
                (SELECT COUNT(*) FROM chat_history ch WHERE ch.user_id = u.id) AS chat_count,
                (SELECT COUNT(*) FROM documents d WHERE d.user_id = u.id) AS document_count,
                (SELECT MAX(ch.ts) FROM chat_history ch WHERE ch.user_id = u.id) AS last_active
            FROM users u
            ORDER BY u.username
            """
        )
        rows = c.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "username": r[1],
                "email": r[2],
                "is_admin": bool(r[3]),
                "is_active": r[4] is None or bool(r[4]),
                "chat_count": r[5],
                "document_count": r[6],
                "last_active": r[7],
            }
            for r in rows
        ]
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.patch("/admin/users/{user_id}")
async def admin_update_user(user_id: str, request: UpdateUserRequest):
    """Promote/demote admin status or suspend/reactivate a user account."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, request.token)

        c.execute("SELECT id FROM users WHERE token = %s", (request.token,))
        acting_row = c.fetchone()
        if acting_row and str(acting_row[0]) == str(user_id):
            conn.close()
            raise HTTPException(400, "You cannot change your own admin or active status.")

        c.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
        target_row = c.fetchone()
        if not target_row:
            conn.close()
            raise HTTPException(404, "User not found")

        if request.is_admin is not None:
            c.execute("UPDATE users SET is_admin = %s WHERE id = %s", (int(request.is_admin), user_id))
            _log_admin_action(c, request.token, "grant_admin" if request.is_admin else "revoke_admin",
                               target_id=user_id, target_username=target_row[1],
                               detail="is_admin: false → true" if request.is_admin else "is_admin: true → false")
        if request.is_active is not None:
            c.execute("UPDATE users SET is_active = %s WHERE id = %s", (int(request.is_active), user_id))
            _log_admin_action(c, request.token, "reactivate_user" if request.is_active else "suspend_user",
                               target_id=user_id, target_username=target_row[1],
                               detail="is_active: false → true" if request.is_active else "is_active: true → false")

        conn.commit()
        conn.close()
        return {"message": "Updated"}
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: str, token: Optional[str] = None):
    """Delete a user account. Their past chats/documents are kept for the audit
    trail — only the ownership link is cleared, matching how anonymous rows
    already display as 'Anonymous'."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        c.execute("SELECT id FROM users WHERE token = %s", (token,))
        acting_row = c.fetchone()
        if acting_row and str(acting_row[0]) == str(user_id):
            conn.close()
            raise HTTPException(400, "You cannot delete your own account.")

        c.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
        target_row = c.fetchone()
        if not target_row:
            conn.close()
            raise HTTPException(404, "User not found")

        c.execute("SELECT COUNT(*) FROM chat_history WHERE user_id = %s", (user_id,))
        chat_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM documents WHERE user_id = %s", (user_id,))
        doc_count = c.fetchone()[0]
        c.execute("UPDATE chat_history SET user_id = NULL WHERE user_id = %s", (user_id,))
        c.execute("UPDATE documents SET user_id = NULL WHERE user_id = %s", (user_id,))
        c.execute("DELETE FROM users WHERE id = %s", (user_id,))
        _log_admin_action(c, token, "delete_user", target_id=user_id, target_username=target_row[1],
                           detail=f"account removed; {chat_count} chats and {doc_count} documents kept, unlinked")
        conn.commit()
        conn.close()
        return {"message": "Deleted"}
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.get("/admin/audit_log")
async def admin_audit_log(
    token: Optional[str] = None, limit: int = 30, offset: int = 0,
    action: Optional[str] = None, search: Optional[str] = None,
):
    """Admin actions (promote/demote/suspend/delete) — the accountability trail
    a legal-compliance-conscious admin dashboard should carry. Returns a page
    of items plus the total count and a breakdown by action type, so callers
    that just want the last few (the Overview timeline) and callers that want
    a full searchable log (the Audit Log tab) share one endpoint."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        where, params = [], []
        if action:
            where.append("action = %s")
            params.append(action)
        if search:
            where.append("(actor_username ILIKE %s OR target_username ILIKE %s)")
            params.extend([f"%{search}%", f"%{search}%"])
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        c.execute(f"SELECT COUNT(*) FROM admin_audit_log {where_sql}", params)
        total_count = c.fetchone()[0]

        c.execute(
            f"""
            SELECT ts, actor_username, action, target_username, detail
            FROM admin_audit_log
            {where_sql}
            ORDER BY id DESC
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        rows = c.fetchall()

        c.execute("SELECT action, COUNT(*) FROM admin_audit_log GROUP BY action ORDER BY COUNT(*) DESC")
        action_breakdown = [{"label": r[0], "count": r[1]} for r in c.fetchall()]

        # 14-day zero-filled series for a second, differently-shaped chart
        # (the breakdown above is a donut; this is a time series) — unfiltered,
        # so it always reflects overall activity regardless of the search/action filter.
        days = 14
        start = datetime.datetime.now() - datetime.timedelta(days=days - 1)
        actions_per_day = _zero_filled_daily_series(c, "admin_audit_log", "ts", days, start)

        conn.close()
        return {
            "items": [
                {
                    "timestamp": r[0],
                    "actor": r[1] or "Unknown",
                    "action": r[2],
                    "target": r[3] or "—",
                    "detail": r[4],
                }
                for r in rows
            ],
            "total_count": total_count,
            "action_breakdown": action_breakdown,
            "actions_per_day": actions_per_day,
        }
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.get("/admin/documents")
async def admin_documents(token: Optional[str] = None, limit: int = 40, offset: int = 0, search: Optional[str] = None):
    """Every document uploaded by every user — who uploaded it, when, how many
    chunks it produced, and whether it's been through Document Insights yet.
    No admin-wide document view existed before this; each user could only see
    their own uploads in the sidebar."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        where, params = "", []
        if search:
            where = "WHERE d.filename ILIKE %s OR u.username ILIKE %s"
            params = [f"%{search}%", f"%{search}%"]

        c.execute(f"SELECT COUNT(*) FROM documents d LEFT JOIN users u ON u.id = d.user_id {where}", params)
        total_count = c.fetchone()[0]

        c.execute(
            f"""
            SELECT d.id, d.filename, d.upload_timestamp, u.username,
                   (SELECT COUNT(*) FROM document_chunks WHERE document_id = d.id) AS chunks,
                   di.doc_type
            FROM documents d
            LEFT JOIN users u ON u.id = d.user_id
            LEFT JOIN document_insights di ON di.document_id = d.id
            {where}
            ORDER BY d.upload_timestamp DESC
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        )
        rows = c.fetchall()

        c.execute(
            """
            SELECT COALESCE(u.username, 'Anonymous'), COUNT(*)
            FROM documents d LEFT JOIN users u ON u.id = d.user_id
            GROUP BY COALESCE(u.username, 'Anonymous')
            ORDER BY COUNT(*) DESC
            LIMIT 8
            """
        )
        top_uploaders = [{"label": r[0], "count": r[1]} for r in c.fetchall()]

        # doc-type breakdown (from Document Insights classification) — a donut,
        # distinct in form from the bar-list uploaders and the bar+line series below.
        # The LLM's free-text classification produces many one-off labels, so cap
        # to the top 5 and fold the long tail into "Other" — a donut reads badly
        # past a handful of slices.
        c.execute(
            "SELECT COALESCE(doc_type, 'Other'), COUNT(*) FROM document_insights GROUP BY COALESCE(doc_type, 'Other') ORDER BY COUNT(*) DESC"
        )
        doc_type_rows = c.fetchall()
        doc_types = [{"label": r[0], "count": r[1]} for r in doc_type_rows[:5]]
        other_count = sum(r[1] for r in doc_type_rows[5:])
        if other_count:
            doc_types.append({"label": "Other", "count": other_count})

        c.execute("SELECT COUNT(*) FROM documents")
        total_documents = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM document_insights")
        analyzed_count = c.fetchone()[0]
        c.execute("SELECT COALESCE(SUM(cnt), 0) FROM (SELECT COUNT(*) cnt FROM document_chunks GROUP BY document_id) t")
        total_chunks_row = c.fetchone()
        c.execute("SELECT COUNT(DISTINCT user_id) FROM documents WHERE user_id IS NOT NULL")
        unique_uploaders = c.fetchone()[0]

        conn.close()
        return {
            "total_documents": total_documents,
            "analyzed_count": analyzed_count,
            "unique_uploaders": unique_uploaders,
            "total_chunks": total_chunks_row[0] if total_chunks_row else 0,
            "top_uploaders": top_uploaders,
            "doc_types": doc_types,
            "total_count": total_count,
            "documents": [
                {
                    "id": r[0],
                    "filename": r[1],
                    "upload_timestamp": r[2],
                    "uploaded_by": r[3] or "Anonymous",
                    "chunks": r[4],
                    "doc_type": r[5],
                }
                for r in rows
            ],
        }
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.get("/admin/document_insights")
async def admin_document_insights(token: Optional[str] = None, limit: int = 40):
    """Aggregated + per-document facts/keywords/entities for the admin
    dashboard's Insights tab: what's actually in the document set, not just
    how many documents there are."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        c.execute("SELECT COUNT(*) FROM documents")
        total_documents = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM document_insights")
        analyzed_count = c.fetchone()[0]

        c.execute("SELECT keywords, facts, entities, doc_type FROM document_insights")
        rows = c.fetchall()

        keyword_counts, entity_counts, doc_type_counts = {}, {}, {}
        total_facts = 0
        for keywords, facts, entities, doc_type in rows:
            for k in (keywords or []):
                key = k.strip().lower()
                if key:
                    keyword_counts[key] = keyword_counts.get(key, 0) + 1
            for e in (entities or []):
                key = e.strip()
                if key:
                    entity_counts[key] = entity_counts.get(key, 0) + 1
            total_facts += len(facts or [])
            dt = (doc_type or "Other").strip() or "Other"
            doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1

        top_keywords = sorted(keyword_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
        top_entities = sorted(entity_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
        # Cap to the top 5 + "Other" — the LLM's free-text classification produces
        # many one-off labels, and a donut reads badly past a handful of slices.
        doc_type_sorted = sorted(doc_type_counts.items(), key=lambda kv: kv[1], reverse=True)
        doc_types = doc_type_sorted[:5]
        other_doc_types_count = sum(v for _, v in doc_type_sorted[5:])
        if other_doc_types_count:
            doc_types.append(("Other", other_doc_types_count))

        c.execute(
            """
            SELECT d.id, d.filename, d.upload_timestamp, di.doc_type, di.keywords, di.facts, di.entities, di.created_at
            FROM document_insights di
            JOIN documents d ON d.id = di.document_id
            ORDER BY di.id DESC
            LIMIT %s
            """,
            (max(1, min(limit, 200)),),
        )
        doc_rows = c.fetchall()
        documents = [
            {
                "document_id": r[0],
                "filename": r[1],
                "upload_timestamp": r[2],
                "doc_type": r[3],
                "keywords": r[4] or [],
                "facts": r[5] or [],
                "entities": r[6] or [],
                "analyzed_at": r[7],
            }
            for r in doc_rows
        ]

        conn.close()
        return {
            "total_documents": total_documents,
            "analyzed_count": analyzed_count,
            "pending_count": max(0, total_documents - analyzed_count),
            "total_facts": total_facts,
            "unique_keywords": len(keyword_counts),
            "top_keywords": [{"label": k, "count": v} for k, v in top_keywords],
            "top_entities": [{"label": k, "count": v} for k, v in top_entities],
            "doc_types": [{"label": k, "count": v} for k, v in doc_types],
            "documents": documents,
        }
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.post("/admin/document_insights/analyze_pending")
async def admin_analyze_pending_documents(token: Optional[str] = None, max_docs: int = 15):
    """Backfill insight extraction for documents uploaded before this feature
    existed (or any that failed the first time). Kicks off background
    extraction threads and returns immediately — the admin refreshes the
    Insights tab a little later to see results land."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)

        c.execute(
            """
            SELECT d.id, d.filename, d.content
            FROM documents d
            LEFT JOIN document_insights di ON di.document_id = d.id
            WHERE di.id IS NULL
            ORDER BY d.upload_timestamp DESC
            LIMIT %s
            """,
            (max(1, min(max_docs, 50)),),
        )
        pending = c.fetchall()
        conn.close()

        from threading import Thread
        for doc_id, filename, content in pending:
            Thread(target=_extract_document_insights, args=(doc_id, filename, content), daemon=True).start()

        return {"queued": len(pending)}
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        raise HTTPException(500, str(e))


@app.get("/admin/system_status")
async def admin_system_status(token: Optional[str] = None):
    """Operational health for the admin dashboard: is the LLM backend reachable,
    which models are installed, and is Postgres up."""
    conn = connect_to_postgres()
    try:
        c = conn.cursor()
        _require_admin(c, token)
        conn.close()
    except HTTPException:
        conn.close()
        raise
    except Exception:
        conn.close()

    db_ok = False
    try:
        probe = connect_to_postgres()
        if probe is not None:
            db_ok = True
            probe.close()
    except Exception:
        db_ok = False

    ollama_ok = False
    ollama_url = None
    models = []
    try:
        ollama_url = get_active_ollama_base_url()
        resp = requests.get(f"{ollama_url}/api/tags", timeout=(OLLAMA_CONNECT_TIMEOUT, 5))
        resp.raise_for_status()
        data = resp.json()
        models = [
            {"name": m.get("name"), "size_gb": round((m.get("size") or 0) / 1e9, 1)}
            for m in data.get("models", [])
        ]
        ollama_ok = True
    except Exception:
        ollama_ok = False

    return {
        "database": {"ok": db_ok},
        "llm_backend": {"ok": ollama_ok, "url": ollama_url, "models": models},
    }


@app.get("/test")
async def test_endpoint():
    return {"message": "SynergeReader API is running successfully!"}


# ------------------- Startup -------------------



@app.post("/convert-docx")
async def convert_docx_to_pdf(file: UploadFile = File(...)):
    """Convert a DOCX file to PDF using LibreOffice headless."""
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(400, "Only .docx files are supported")
    tmp_dir = tempfile.mkdtemp()
    try:
        docx_path = os.path.join(tmp_dir, sanitize_filename(file.filename))
        content   = await file.read()
        with open(docx_path, "wb") as f:
            f.write(content)
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", tmp_dir, docx_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Conversion failed: {result.stderr}")
        pdf_name = os.path.splitext(file.filename)[0] + ".pdf"
        pdf_path = os.path.join(tmp_dir, pdf_name)
        if not os.path.exists(pdf_path):
            raise HTTPException(500, "PDF output not found")
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={pdf_name}",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
