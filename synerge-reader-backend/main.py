from contextlib import asynccontextmanager
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
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
from fastapi.responses import JSONResponse
from schemas import AskRequest, AskResponse, CorrectionRequest, RatingRequest,GoogleLoginRequest,LoginRequest,RegisterRequest,ResetPasswordRequest,ResendVerificationRequest
from schemas import HistoryItem,HistoryRequest, KnowledgeItem,KnowledgeInsertRequest,ForgotPasswordRequest,KnowledgeUrlImportRequest
from schemas import GenerationSelectRequest
import asyncio
import concurrent.futures
import threading
import uuid
import os
import string
import datetime
import re
from typing import List, Optional
from dbSetup import VectorSchemaError,init_db,connect_to_postgres,test_postgres_connection
from rag_model_profiles import resolve_embedding_profile
from ollama_embedding_provider import EmbeddingProviderError, OllamaEmbeddingProvider
from generation_profiles import (
    COMPAT_RUNTIME,
    PRIMARY_RUNTIME,
    AnswerGenerationSnapshot,
    ExperimentalProfileDisabledError,
    GenerationProfileError,
    ModelIdentityMismatchError,
    ModelIdentityUnavailableError,
    ModelRuntimeUnavailableError,
    ModelSelectionError,
    ProfileNotQualifiedError,
    UnknownModelProfileError,
    check_installed_identity,
    default_model_registry,
    select_shared_profile,
    shared_generation_snapshot,
)
from ollama_generation import (
    FAILURE_OUTPUT_LIMIT,
    OllamaGenerateStream,
    completed_generate_response,
)
from answer_provenance import (
    AnswerTimer,
    answer_provenance_json,
    build_answer_provenance,
    new_trace_id,
)
from release_metadata import API_CONTRACT_VERSION, release_metadata_from_env
from generation_admission import (
    AdmissionRefusal,
    AdmissionRefused,
    AdmissionSettings,
    DispatchRefused,
    GenerationLedger,
    GenerationTicket,
    HelperPool,
    SelectionPolicy,
    SharedSelection,
    current_dispatch_permitted,
    current_generation_ticket,
)
from helper_jobs import EnqueueOutcome, HelperJobQueue, HelperQueueSettings
from enhanced_extraction import EXTRACTION_MODES, ExtractionSettings, build_document_extractor
from upload_jobs import SubmitOutcome, UploadJobRegistry, UploadJobSettings
from starlette.concurrency import run_in_threadpool
from answer_jobs import (
    ANSWER_TIMEOUT_ERROR,
    NOT_DISPATCHED_ERROR,
    AdmittedAnswerResponse,
    AnswerJob,
    DeliverySettings,
    acquire_answer,
)
from generation_selection import GenerationOwner, OwnerSettings, PostgresGenerationStore
from generation_coordinator import (
    CoordinatorSettings,
    GenerationRuntime,
    RuntimeInventoryUnavailable,
    SwitchCoordinator,
    loaded_model_entries,
)
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
import socket
import ipaddress
from urllib.parse import urlparse, urljoin, urlsplit
from lxml import html as lxml_html
from email_validator import validate_email, EmailNotValidError
from fastapi.responses import Response

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize required database state before accepting requests.

    Initialization ends by starting generation ownership (see
    _initialize_application): until this process holds it, every generation
    request is refused. The background-insights workers start only after
    initialization has succeeded. A failed worker start leaves that queue
    unavailable while the application still serves. Shutdown closes the
    queue's intake and signals its workers, then fences generation and lets
    ownership go, without waiting for either; it proves nothing about model
    work already submitted -- unobserved calls stay durably unresolved for the
    next owner -- and the queue's in-memory accounting is kept for a later
    lifespan in the same process.
    """
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
    insights_queue = _INSIGHTS_QUEUE
    _start_insights_queue(insights_queue)
    try:
        yield
    finally:
        _close_insights_queue(insights_queue)
        _stop_generation_runtime(_GENERATION_RUNTIME)


app = FastAPI(title="SynergeReader API", version=API_CONTRACT_VERSION, lifespan=lifespan)

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
# Embedding keep-alive only. Generation calls take theirs from the model
# registry below, never from the environment or a request.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Generation runtimes (backend-owned keys; see generation_profiles.py). The
# PRIMARY runtime is the existing host Ollama resolved above (OLLAMA_BASE_URL
# or the host fallbacks); it keeps serving Llama, Gemma and the configured
# embedder, and is never repointed. The COMPATIBILITY runtime serves only the
# experimental profiles. Its endpoint comes from server configuration alone:
# SYNERGE_COMPAT_OLLAMA_URL, a plain http://<private-alias>:<port> on the
# backend's private runtime network. It has no fallback host, and a request
# can never name it. Unset or malformed: the runtime is not configured and
# every profile it serves is refused, while the primary runtime is unaffected.
_COMPAT_HOST_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$")


def _compat_runtime_url(value: str) -> Optional[str]:
    """``http://host:port`` from a server setting, or None. No credentials,
    path, query or fragment; an explicit port is required. The value itself
    is never logged."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname
    if (
        parts.scheme != "http" or not host or not _COMPAT_HOST_PATTERN.match(host) or port is None
        or parts.username is not None or parts.password is not None
        or parts.path not in ("", "/") or parts.query or parts.fragment
    ):
        return None
    return f"http://{host}:{port}"


_COMPAT_RUNTIME_SETTING = os.getenv("SYNERGE_COMPAT_OLLAMA_URL", "")
_COMPAT_RUNTIME_URL = _compat_runtime_url(_COMPAT_RUNTIME_SETTING)
if _COMPAT_RUNTIME_SETTING.strip() and _COMPAT_RUNTIME_URL is None:
    print("[Generation] SYNERGE_COMPAT_OLLAMA_URL is not a valid private runtime URL; "
          "the compatibility runtime is not configured")
# Server-side enablement of the experimental profiles. Default off; only
# 1/true/yes enable them.
_EXPERIMENTAL_MODELS_ENABLED = os.getenv("SYNERGE_EXPERIMENTAL_MODELS", "").strip().lower() in ("1", "true", "yes")
_CONFIGURED_RUNTIMES = frozenset({PRIMARY_RUNTIME} | ({COMPAT_RUNTIME} if _COMPAT_RUNTIME_URL else set()))
print(
    "[Generation] experimental profiles "
    f"{'enabled' if _EXPERIMENTAL_MODELS_ENABLED else 'disabled'}; compatibility runtime "
    f"{'configured' if _COMPAT_RUNTIME_URL else 'not configured'}"
)

# The one source of answer, verifier and helper model settings. Pure data,
# built at import: no request can add a tag, change an option, or redirect a
# call. See generation_profiles.py.
_MODEL_REGISTRY = default_model_registry(
    experimental_enabled=_EXPERIMENTAL_MODELS_ENABLED,
    configured_runtimes=_CONFIGURED_RUNTIMES,
)
# This deployment's release identity: the app version, the prompt versions,
# and the build commit the release operator injected as SYNERGE_BUILD_SHA.
# Read once at import. Git is never run here; a missing or malformed SHA is
# reported as unknown. See release_metadata.py.
_RELEASE = release_metadata_from_env(os.environ)
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


_COMPAT_RUNTIME_CHECKED_AT = 0.0
_COMPAT_RUNTIME_REACHABLE = False


def _compat_runtime_base_url() -> str:
    """The compatibility runtime's configured endpoint once it answered a
    read-only ``/api/version`` within the last 30 s (like the primary's
    health cache). Raises when it is not configured or not reachable, BEFORE
    any dispatch intent is recorded, so an unreachable runtime never leaves a
    call unresolved. There is no fallback host."""
    global _COMPAT_RUNTIME_CHECKED_AT, _COMPAT_RUNTIME_REACHABLE
    if not _COMPAT_RUNTIME_URL:
        raise RuntimeError("the compatibility runtime is not configured")
    now = time.time()
    if _COMPAT_RUNTIME_REACHABLE and now - _COMPAT_RUNTIME_CHECKED_AT < 30:
        return _COMPAT_RUNTIME_URL
    try:
        resp = requests.get(f"{_COMPAT_RUNTIME_URL}/api/version", timeout=(OLLAMA_CONNECT_TIMEOUT, 5))
        resp.raise_for_status()
    except Exception:
        _COMPAT_RUNTIME_REACHABLE = False
        _COMPAT_RUNTIME_CHECKED_AT = now
        raise RuntimeError("the compatibility runtime is not reachable") from None
    _COMPAT_RUNTIME_REACHABLE = True
    _COMPAT_RUNTIME_CHECKED_AT = now
    return _COMPAT_RUNTIME_URL


def _runtime_base_url(runtime_key: str) -> str:
    """The endpoint for one backend-owned runtime key. The primary runtime
    keeps its existing resolution; any other key is resolved only from
    server configuration. Raises (never falls back) when unavailable."""
    if runtime_key == PRIMARY_RUNTIME:
        return get_active_ollama_base_url()
    if runtime_key == COMPAT_RUNTIME:
        return _compat_runtime_base_url()
    raise RuntimeError("unknown generation runtime")


def post_ollama(endpoint: str, payload: dict, *, stream: bool = False, timeout: float = 60):
    """POST to a local Ollama endpoint.

    ``timeout`` is the Requests READ timeout: the longest wait for any one
    read, including the first response byte (which covers a cold model load).
    The connect timeout stays the short OLLAMA_CONNECT_TIMEOUT. It bounds each
    wait, not a whole job.

    Anything other than ``/api/generate`` (the embedder) goes to the primary
    runtime, exactly as before. A native generation call is sent only for the
    admitted job of the calling thread (an /ask answer job; see
    answer_jobs.py), and only while that job may dispatch. It goes to the
    runtime its job's snapshot holds -- never one a payload or request names
    -- and only with that snapshot's own model tag. Its read timeout is clamped
    to the job's remaining wall-clock budget, and a durable dispatch intent is
    recorded immediately before the POST -- after the endpoint was resolved,
    so an unreachable server never counts as a dispatch. Otherwise nothing is
    sent (DispatchRefused). A non-streamed reply shows its decoded body to the
    job; a streamed reply is observed by the job once it has been read.
    """
    if endpoint != "/api/generate":
        base_url = get_active_ollama_base_url()
        return requests.post(
            f"{base_url}{endpoint}",
            json=payload,
            stream=stream,
            timeout=(OLLAMA_CONNECT_TIMEOUT, timeout or OLLAMA_READ_TIMEOUT),
        )
    ticket = current_generation_ticket()
    if ticket is None or not current_dispatch_permitted():
        raise DispatchRefused("no admitted job may dispatch this generation call")
    snapshot = ticket.snapshot
    if not isinstance(snapshot, AnswerGenerationSnapshot) or not isinstance(payload, dict) \
            or payload.get("model") != snapshot.model_tag:
        raise DispatchRefused("a generation call must use its own job's model")
    base_url = _runtime_base_url(snapshot.runtime_key)
    read_timeout = ticket.read_timeout(timeout or OLLAMA_READ_TIMEOUT)
    ticket.dispatching("answer" if stream else "verifier")
    response = requests.post(
        f"{base_url}{endpoint}",
        json=payload,
        stream=stream,
        timeout=(OLLAMA_CONNECT_TIMEOUT, read_timeout),
    )
    return response if stream else _ObservedReply(response, ticket)


class _ObservedReply:
    """A non-streamed generation reply whose decoded body is shown to the job
    that dispatched it, so a terminal reply returns the job's capacity and
    anything else (an error status, an undecodable or unfinished body) leaves
    the call unresolved."""

    def __init__(self, response, ticket: GenerationTicket) -> None:
        self._response = response
        self._ticket = ticket

    def raise_for_status(self):
        return self._response.raise_for_status()

    def json(self):
        body = self._response.json()
        self._ticket.observed(body)
        return body

    def __getattr__(self, name):
        return getattr(self._response, name)


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
    """Initialize database state using the resolved embedding dimension, then
    start generation ownership. Ownership is acquired on its own thread (see
    generation_selection.GenerationOwner): this never waits for it, and no
    generation is admitted until it is held."""
    init_db(expected_dimension=_EMBEDDING_PROFILE.dimension)
    _start_generation_runtime(_GENERATION_RUNTIME)


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


# Background helper calls are bounded calls derived from the shared selection
# their job captured when it was admitted. They used to probe a list of tags
# (several of them never approved) with a bare "Say OK" generation each --
# every probe could load a model at its default, unbounded context -- and then
# send their temperature at the top level, where Ollama ignores it.
_KB_HELPER_TEMPERATURE = 0.2
_INSIGHTS_HELPER_TEMPERATURE = 0.1


def _helper_generation(ticket: GenerationTicket) -> AnswerGenerationSnapshot:
    """The snapshot for one admitted helper job, built once from the shared
    selection its ticket captured at admission -- never from a request, never
    re-resolved per excerpt, and never a fallback model.

    The unqualified control makes no inventory call. A profile that requires
    an identity check (qualified or experimental) has its installed identity
    read first, on its own runtime, and an unknown, ineligible, missing,
    unreadable or mismatched one raises, so the job generates nothing.
    """
    if ticket.snapshot is not None:
        return ticket.snapshot
    profile = select_shared_profile(_MODEL_REGISTRY, ticket.selection.profile_id)
    observed = (
        check_installed_identity(profile, _identity_lookup(profile.runtime_key))
        if profile.requires_identity_check else None
    )
    snapshot = shared_generation_snapshot(
        profile, epoch=ticket.selection.epoch, observed_digest=observed, registry=_MODEL_REGISTRY
    )
    ticket.bind_snapshot(snapshot)
    return snapshot


def _admitted_helper_ticket(recorder):
    """(ticket, owned): the caller's admitted helper ticket, or one admitted
    here without waiting -- owned, so the caller releases it. (None, False)
    when no helper slot is free or generation is not owned here."""
    if recorder is not None:
        return (recorder, False) if isinstance(recorder, GenerationTicket) else (None, False)
    ticket = _GENERATION_LEDGER.try_acquire_helper()
    return (ticket, True) if ticket is not None else (None, False)


def _generate_qa_pairs_from_text(text: str, label: str, *, recorder: Optional[GenerationTicket] = None) -> List[dict]:
    """Ask the LLM to write Q:/A: pairs from arbitrary text (a document excerpt
    or scraped external page). Shared by document-upload seeding and the
    external URL importer — the only difference between those two source
    kinds is what gets stored alongside the pairs, not how they're generated.

    Each excerpt's reply must be a finished generation (see
    completed_generate_response) before it is parsed: an unfinished, cut-off,
    blank or error reply contributes no pairs, and pairs from the other
    excerpt are kept. Log lines carry fixed categories and counts only --
    never ``label``, the text, the model output, or exception text.

    The job runs under one admitted helper ticket (``recorder``, or one taken
    here without waiting): its model is built once from the shared selection
    captured at admission (see _helper_generation), so every excerpt uses the
    same snapshot. Each call records a durable intent first; once a call's end
    was not observed, no further excerpt is sent in that slot."""
    if len(text.strip()) < 200:
        return []
    ticket, owned = _admitted_helper_ticket(recorder)
    if ticket is None:
        print("[KB] Helper model capacity unavailable; Q&A generation skipped")
        return []
    try:
        return _qa_pairs_for_ticket(text, ticket)
    finally:
        if owned:
            _GENERATION_LEDGER.release(ticket)


def _qa_pairs_for_ticket(text: str, ticket: GenerationTicket) -> List[dict]:
    try:
        generation = _helper_generation(ticket)
    except Exception as exc:
        print(f"[KB] Helper model unavailable ({type(exc).__name__}); Q&A generation skipped")
        return []
    try:
        # The helper's own runtime, bound in its snapshot; never a fallback.
        base_url = _runtime_base_url(generation.runtime_key)
    except Exception:
        print("[KB] Ollama not reachable; Q&A generation skipped")
        return []

    helper = generation.helper_call(_KB_HELPER_TEMPERATURE)

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

        if not ticket.may_dispatch:
            print(f"[KB] Excerpt {i+1}: no further model call permitted for this job; no pairs taken")
            continue

        try:
            payload = helper.ollama_payload(prompt, stream=False)
            ticket.dispatching("helper")
            resp = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=(5, ticket.read_timeout(helper.read_timeout_s)),
            )
            resp.raise_for_status()
            body = resp.json()
            ticket.observed(body)
            print(helper.describe_completion("qa_pairs", body))
            reply = completed_generate_response(body)
            if reply is None:
                print(f"[KB] Excerpt {i+1}: reply was not a completed response; no pairs taken")
                continue
            raw = reply.strip()
            print(f"[KB] Excerpt {i+1}: completed reply of {len(raw)} chars")

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
            print(f"[KB] Excerpt {i+1}: model call failed ({type(e).__name__}); no pairs taken")
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


def _extract_document_insights(doc_id: int, filename: str, text: str, *, recorder: Optional[GenerationTicket] = None):
    """Ask the LLM to read a document and pull out a type classification, key
    facts, keywords, and named entities/parties — then store one row per
    document for the admin dashboard's Insights tab. Runs on a background
    insights-queue worker (see _run_insight_job); never blocks the upload
    response, and any failure here is non-critical (the document itself is
    already saved and usable).

    The reply must be a finished generation (see completed_generate_response)
    before anything is parsed or stored: an unfinished, cut-off, blank or error
    reply writes nothing, so an existing insights row is never overwritten
    from it. Log lines carry fixed categories and counts only -- never the
    filename, the text, the model output or values parsed from it, or
    exception text. The insights connection and cursor are closed on every
    path, and an uncommitted write is rolled back.

    The job runs under one admitted helper ticket: ``recorder`` (the queue
    worker's, keyword-only), or one taken here without waiting and released
    here. Its model is built once from the shared selection the ticket
    captured at admission (see _helper_generation); if it cannot be built,
    nothing is generated or stored. The ticket records a durable intent
    immediately before the POST and is shown the decoded reply body, so a call
    whose termination was never observed stays accounted for -- across a
    restart too -- even though this function swallows its own errors."""
    if len(text.strip()) < 200:
        return
    ticket, owned = _admitted_helper_ticket(recorder)
    if ticket is None:
        print("[Insights] Helper model capacity unavailable; insight extraction skipped")
        return
    try:
        _extract_insights_for_ticket(doc_id, filename, text, ticket)
    finally:
        if owned:
            _GENERATION_LEDGER.release(ticket)


def _extract_insights_for_ticket(doc_id: int, filename: str, text: str, recorder: GenerationTicket) -> None:
    try:
        generation = _helper_generation(recorder)
    except Exception as exc:
        print(f"[Insights] Helper model unavailable ({type(exc).__name__}); insight extraction skipped")
        return
    try:
        # The helper's own runtime, bound in its snapshot; never a fallback.
        base_url = _runtime_base_url(generation.runtime_key)
    except Exception:
        print("[Insights] Ollama not reachable; insight extraction skipped")
        return

    try:
        # Insights' own evidenced policy where the job's profile has one,
        # otherwise the helper policy; refused before anything is dispatched.
        helper = generation.insights_call(_INSIGHTS_HELPER_TEMPERATURE)
    except GenerationProfileError:
        print("[Insights] Helper call refused by its profile policy; insight extraction skipped")
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
        payload = helper.ollama_payload(prompt, stream=False)
        recorder.dispatching("helper")
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=(5, recorder.read_timeout(helper.read_timeout_s)),
        )
        resp.raise_for_status()
        body = resp.json()
        recorder.observed(body)
        print(helper.describe_completion("insights", body))
        reply = completed_generate_response(body)
        if reply is None:
            print("[Insights] Reply was not a completed response; nothing stored")
            return
        raw = reply.strip()
        parsed = _extract_json_block(raw)
        if not parsed:
            print(f"[Insights] No JSON object in a {len(raw)}-char reply; nothing stored")
            return

        doc_type = str(parsed.get("doc_type") or "Other").strip()[:60]
        keywords = [str(k).strip() for k in (parsed.get("keywords") or []) if str(k).strip()][:10]
        facts = [str(f).strip() for f in (parsed.get("facts") or []) if str(f).strip()][:8]
        entities = [str(e).strip() for e in (parsed.get("entities") or []) if str(e).strip()][:10]

        conn = connect_to_postgres()
        if conn is None:
            print("[Insights] Extraction failed (non-critical): connection unavailable")
            return
        c = None
        committed = False
        try:
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
            committed = True
        finally:
            if not committed:
                try:
                    conn.rollback()
                except Exception:
                    pass
            _close_db_quietly(c, conn)
        print(f"[Insights] Stored: {len(keywords)} keywords, {len(facts)} facts, {len(entities)} entities")
    except Exception as e:
        print(f"[Insights] Extraction failed (non-critical): {type(e).__name__}")


def _close_db_quietly(cursor, connection) -> bool:
    """Close an acquired cursor and connection, each at most once; ``None`` is
    never closed. Returns False if either close raised. Nothing about the
    failure is logged or kept."""
    closed = True
    if cursor is not None:
        try:
            cursor.close()
        except Exception:
            closed = False
    if connection is not None:
        try:
            connection.close()
        except Exception:
            closed = False
    return closed


class _InsightSourceUnavailable(Exception):
    """The queued document's current row could not be read. Carries no detail."""


# The CURRENT row of one document that still has no insights row, by its exact
# id: the same eligibility the pending-analysis selection uses. Insights stay
# a per-document row derived only from that document's own stored text.
_PENDING_INSIGHT_SOURCE_SQL = """
    SELECT d.filename, d.content
    FROM documents d
    LEFT JOIN document_insights di ON di.document_id = d.id
    WHERE d.id = %s AND di.id IS NULL
"""


def _load_pending_insight_source(document_id: int):
    """``(filename, content)`` for a document still awaiting insights, or None
    when it was deleted or already analysed. Opens its own connection and
    closes cursor and connection before returning; any database failure raises
    the detail-free _InsightSourceUnavailable."""
    conn = None
    cursor = None
    row = None
    failed = False
    try:
        conn = connect_to_postgres()
        if conn is None:
            failed = True
        else:
            cursor = conn.cursor()
            cursor.execute(_PENDING_INSIGHT_SOURCE_SQL, (document_id,))
            row = cursor.fetchone()
    except Exception:
        failed = True
    _close_db_quietly(cursor, conn)
    if failed:
        # Raised outside the except block: nothing chains onto it.
        raise _InsightSourceUnavailable()
    return row


def _run_insight_job(document_id: int, call: GenerationTicket) -> None:
    """One queued insights job, run by an insights-queue worker that already
    holds an admitted helper ticket. Reloads the current row by its exact id
    when the job actually starts -- nothing private ever waited in the queue --
    and skips a document that was deleted or analysed meanwhile. ``call`` is
    the ticket: it captured the shared selection when the worker took the job,
    and records whether the model was invoked and whether its termination was
    observed."""
    try:
        row = _load_pending_insight_source(document_id)
    except _InsightSourceUnavailable:
        print("[Insights] Queued document could not be loaded; skipped")
        return
    if row is None:
        print("[Insights] Queued document is no longer pending; skipped")
        return
    filename, content = row[0], row[1]
    if not isinstance(filename, str) or not isinstance(content, str):
        print("[Insights] Queued document is not readable text; skipped")
        return
    _extract_document_insights(document_id, filename, content, recorder=call)


async def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode()



# A session token identifies a user only while that account is active.
# Suspending an account (is_active = 0) must invalidate the tokens it was
# already issued -- tokens never expire -- not just block its next login, so
# every protected identity check uses one of these lookups. NULL counts as
# active, matching /login and /me: the column predates suspension.
_ACTIVE_USER_ID_BY_TOKEN = (
    "SELECT id FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
)
_ACTIVE_ADMIN_FLAG_BY_TOKEN = (
    "SELECT is_admin FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0"
)


def _resolve_caller_id(auth_token: Optional[str]):
    """The signed-in caller of /upload or /ask, from the users.token column.

    Both routes create persistent rows (documents, chat history) that belong
    to whoever made the request, so a caller must be identified before any
    work starts -- a request can never degrade into an ownerless record:

    * no token is refused with 401, without opening a connection;
    * a token that matches a user gives that user's id;
    * a token that matches no user is refused with 401;
    * a lookup that cannot complete -- no connection, or a connection, cursor
      or query failure -- is refused with a generic 503.

    Every refusal carries a fixed message and is raised outside any ``except``
    block, so neither the token, the SQL, nor the database exception reaches
    the response, even by exception chaining. A lookup failure is logged by
    exception class name only, and the cursor and connection are closed on
    every path.
    """
    if not auth_token:
        raise HTTPException(401, "Unauthorized")

    lookup_row = None
    lookup_failed = False
    try:
        lookup_conn = connect_to_postgres()
        if lookup_conn is None:
            print("[Upload] uploader lookup failed: connection unavailable")
            lookup_failed = True
        else:
            lookup_cursor = None
            try:
                lookup_cursor = lookup_conn.cursor()
                lookup_cursor.execute(_ACTIVE_USER_ID_BY_TOKEN, (auth_token,))
                lookup_row = lookup_cursor.fetchone()
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
    except Exception as exc:
        print(f"[Upload] uploader lookup failed: {type(exc).__name__}")
        lookup_failed = True

    if lookup_failed:
        raise HTTPException(503, "Could not verify your session. Please try again shortly.")
    if not lookup_row or lookup_row[0] is None:
        raise HTTPException(401, "Invalid session")
    return lookup_row[0]


class _PostCommitDispatchError(RuntimeError):
    """At least one post-commit follow-up could not be started.

    Raised with a fixed message and nothing else: no exception text, no
    document or chunk content, no filename, no auth token, no SQL, and no
    connection detail. DocumentIngestionService catches it and converts it into
    its existing fixed warning on an already-indexed result.
    """


# ------------------- Shared generation: admission, ownership, switching -------------------
#
# ONE ledger admits every generation job in this process: /ask answers (with
# their verifier calls), background insights and URL-import Q&A. Only the
# process holding generation ownership (a PostgreSQL session advisory lock,
# see generation_selection.py) installs its durable dispatch-intent store into
# it; until then, and after ownership is lost or the application shuts down,
# nothing is admitted or dispatched. Every value below is PROVISIONAL -- chosen
# for a single-host first release and not measured on the production host.
# Constructing any of this starts no thread and does no I/O.
_ADMISSION_SETTINGS = AdmissionSettings()
_DELIVERY_SETTINGS = DeliverySettings()


def _selection_policy(selection: SharedSelection) -> Optional[SelectionPolicy]:
    """The admission policy of the profile a shared selection names (pure:
    a registry lookup). An experimental profile admits one combined job and
    may carry its own job budgets; every other profile keeps the shared
    settings, so Gemma's and Llama's admission is unchanged."""
    policy = _MODEL_REGISTRY.admission_policy(selection.profile_id)
    if policy is None:
        return None
    max_jobs, answer_budget_s, helper_budget_s = policy
    if max_jobs is None and answer_budget_s is None and helper_budget_s is None:
        return None
    return SelectionPolicy(max_jobs=max_jobs, answer_job_budget_s=answer_budget_s,
                           helper_job_budget_s=helper_budget_s)


_GENERATION_LEDGER = GenerationLedger(_ADMISSION_SETTINGS, clock=time.monotonic, selection_policy=_selection_policy)
# A fresh id per process start: the owner row and every dispatch intent record
# which boot wrote them.
_BOOT_ID = str(uuid.uuid4())


def _new_generation_store() -> PostgresGenerationStore:
    """One owner-session store; its connection opens only when it tries to
    acquire ownership."""
    return PostgresGenerationStore(lambda: connect_to_postgres(), boot_id=_BOOT_ID)


def _resident_models(runtime_key: str = PRIMARY_RUNTIME) -> list:
    """Read-only: the models one runtime currently reports as loaded
    (/api/ps). Used only as the switch coordinator's residency barrier; it is
    a point-in-time view and never proves exclusive GPU use. A successful
    response alone is not a readable inventory: a reply without an explicit
    models list, or with an entry that names no model, raises.

    Two kinds of failure are kept apart, because the coordinator may excuse
    only the first, and only for the previous compatibility runtime on a
    return to Gemma or Llama. RuntimeInventoryUnavailable is raised ONLY for
    a compatibility runtime that is not configured here, or for a CONFIRMED
    connection-establishment failure: a connect timeout, or a new connection
    that could not be made (refused, unreachable, name not resolved). Every
    other failure fails closed -- the runtime may have answered and may
    still hold weights: a timeout or protocol error while reading the status
    line, headers or body (Requests can report a body read timeout as a
    ConnectionError), any other or ambiguous ConnectionError, an HTTP error
    status, a body that is not JSON, or a malformed list.

    The response is streamed so the HTTP status is checked BEFORE the body is
    read, and it is always closed. Redirects are never followed, and only
    HTTP 200 is accepted: any other status, a 3xx included (which
    ``raise_for_status`` lets through), fails closed without its body being
    read. The compatibility list is read directly from its configured
    endpoint (no reachability cache) so a failure is classified by this read
    itself."""
    if runtime_key == COMPAT_RUNTIME:
        if not _COMPAT_RUNTIME_URL:
            raise RuntimeInventoryUnavailable("the compatibility runtime is not configured")
        base_url = _COMPAT_RUNTIME_URL
    else:
        base_url = _runtime_base_url(runtime_key)
    try:
        response = requests.get(
            f"{base_url}/api/ps", timeout=(OLLAMA_CONNECT_TIMEOUT, 5), stream=True, allow_redirects=False,
        )
    except requests.exceptions.ConnectTimeout:
        raise RuntimeInventoryUnavailable("the runtime did not accept a connection") from None
    except requests.exceptions.ConnectionError as exc:
        if _connection_never_established(exc):
            raise RuntimeInventoryUnavailable("the runtime did not accept a connection") from None
        raise RuntimeError("the runtime's loaded-model list could not be read") from None
    except Exception:
        # A read timeout on the status line or headers, or anything else: fail closed.
        raise RuntimeError("the runtime's loaded-model list could not be read") from None
    try:
        if response.status_code != 200:
            raise ValueError("not HTTP 200")  # a redirect or any other status: fail closed
        payload = response.json()
    except Exception:
        raise RuntimeError("the runtime's loaded-model list could not be read") from None
    finally:
        response.close()
    return loaded_model_entries(payload)


def _connection_never_established(exc: BaseException) -> bool:
    """Whether a Requests ConnectionError is a confirmed failure to OPEN the
    connection: urllib3's NewConnectionError (refused, unreachable, name not
    resolved) as the reason of the MaxRetryError Requests wraps. Anything
    else -- a protocol error, a dropped connection, a body read timeout that
    Requests reports as a ConnectionError, or an error without that exact
    cause -- is not confirmed and returns False."""
    from urllib3.exceptions import MaxRetryError, NewConnectionError

    cause = exc.args[0] if exc.args else None
    return isinstance(cause, MaxRetryError) and isinstance(cause.reason, NewConnectionError)


def _coordinator_settings() -> CoordinatorSettings:
    """The drain wait must outlast the longest whole-job budget a running job
    may have been granted (an experimental profile's may exceed the shared
    420 s), or a switch would abort while that job is still legitimately
    running. Residency keeps its provisional bound: the previous model leaves
    by its own five-minute keep-alive. Provisional, unmeasured."""
    longest = max(
        [_ADMISSION_SETTINGS.answer_job_budget_s, _ADMISSION_SETTINGS.helper_job_budget_s]
        + [budget for profile in _MODEL_REGISTRY.profiles
           for budget in (profile.answer_job_budget_s, profile.helper_job_budget_s) if budget is not None]
    )
    defaults = CoordinatorSettings()
    return CoordinatorSettings(drain_timeout_s=max(defaults.drain_timeout_s, float(longest) + 60.0))


_GENERATION_OWNER = GenerationOwner(
    store_factory=lambda: _new_generation_store(),
    ledger=_GENERATION_LEDGER,
    # Seeds the persisted selection only when it does not exist yet.
    default_profile_id=lambda: _MODEL_REGISTRY.default_profile_id,
    settings=OwnerSettings(),
)
_GENERATION_RUNTIME = GenerationRuntime(
    ledger=_GENERATION_LEDGER,
    owner=_GENERATION_OWNER,
    coordinator=SwitchCoordinator(
        ledger=_GENERATION_LEDGER,
        owner=_GENERATION_OWNER,
        registry_provider=lambda: _MODEL_REGISTRY,
        # Both are read-only, and always called with the runtime the profile
        # in question actually uses (see SwitchCoordinator).
        installed_digest=lambda tag, runtime_key: _identity_lookup(runtime_key)(tag),
        resident_models=lambda runtime_key: _resident_models(runtime_key),
        settings=_coordinator_settings(),
    ),
)


def _start_generation_runtime(runtime) -> None:
    """Start ownership acquisition and the switch coordinator; never waits,
    never raises. Generation stays unavailable until ownership is held."""
    try:
        runtime.start()
    except Exception:
        print("[Generation] runtime could not start; generation unavailable")


def _stop_generation_runtime(runtime) -> None:
    """Fence generation and let ownership go; never waits, never raises."""
    try:
        runtime.stop()
    except Exception:
        print("[Generation] runtime could not be stopped cleanly")


# Background insights run through ONE bounded, id-only queue per process:
# ``workers`` threads and at most ``max_queued`` waiting document ids (see
# helper_jobs.py). A worker takes a job only when the shared ledger grants it a
# helper ticket, which captures the shared model selection at that moment.
# These are provisional local values, not measured production capacity.
# Constructing the queue starts no thread and does no I/O; the lifespan starts
# its workers.
_INSIGHTS_QUEUE_SETTINGS = HelperQueueSettings(workers=1, max_queued=100)
_INSIGHTS_QUEUE = HelperJobQueue(
    settings=_INSIGHTS_QUEUE_SETTINGS,
    # Looked up when each job runs, never captured per item.
    run_job=lambda document_id, call: _run_insight_job(document_id, call),
    capacity=HelperPool(_GENERATION_LEDGER),
)

_ENQUEUE_FAILURE_LABELS = {
    EnqueueOutcome.FULL: "queue full",
    EnqueueOutcome.UNAVAILABLE: "queue unavailable",
}

# --- Enhanced extraction (OCR / tables) and background upload jobs ----------
#
# OCR and table extraction run in a separate private worker service reached
# through EXTRACTOR_URL with the deployment-supplied EXTRACTOR_TOKEN (never
# logged or returned). With no worker configured, PDFs are processed exactly
# as before. Background jobs keep slow uploads off the request loop; the
# registry is bounded and in-memory (see upload_jobs.py). Constructing it
# starts no thread; the worker starts on the first accepted job.
_UPLOAD_JOBS = UploadJobRegistry(UploadJobSettings())
_UPLOAD_PROCESSING_MODES = ("sync", "background")
_UPLOAD_JOB_REFUSALS = {
    SubmitOutcome.QUEUE_FULL: (429, "The processing queue is full. Please try again in a few minutes."),
    SubmitOutcome.OWNER_LIMIT: (429, "You already have uploads processing. Please wait for them to finish."),
    SubmitOutcome.TOO_LARGE: (413, "The upload batch is too large to process in the background."),
    SubmitOutcome.UNAVAILABLE: (503, "Background processing is temporarily unavailable."),
}


def _build_document_extractor(extraction_mode: str = "standard"):
    """The per-request PDF extractor: native text, plus OCR/tables via the worker.

    Settings are read from the environment at call time, so a deployment's
    EXTRACTOR_URL / EXTRACTOR_TOKEN apply without import-order coupling.
    """
    return build_document_extractor(ExtractionSettings.from_env(), mode=extraction_mode)


def _start_insights_queue(queue) -> None:
    """Start the background-insights workers; a failure leaves the queue
    unavailable (its own fixed log line) and never stops the application."""
    try:
        queue.start()
    except Exception:
        print("[Insights] Queue worker could not start; background insights unavailable")


def _close_insights_queue(queue) -> None:
    """Close intake and signal the workers; never waits and never raises."""
    try:
        queue.close()
    except Exception:
        print("[Insights] Queue could not be closed cleanly")


def _dispatch_upload_followups(committed: CommittedDocument) -> None:
    """Post-commit work for one ingested document: queue its insights.

    DocumentIngestionService calls this only after the document's transaction
    has committed and its connection has been closed, and treats any failure
    here as a warning on an already-indexed result -- so a follow-up can never
    turn a committed document into a reported ingestion failure.

    Only the committed document id is queued. The filename and server-extracted
    text stay out of the queue: the worker reloads the current row by that id
    when the job actually starts (see _run_insight_job). An id that is already
    queued, running or unresolved is a no-op.

    Security containment (PR #54): generate_kb_from_document is deliberately
    NOT dispatched. It writes Q&A pairs derived from the uploaded text into
    the knowledge base, which has no owner column and is shared by every
    user, so a private document's content would reach other users. It stays
    disabled until the knowledge base is user-owned. Document insights are a
    per-document row and are unaffected.

    A full or unavailable queue, or any admission failure, raises one generic
    _PostCommitDispatchError; only a fixed category is logged -- never the
    exception, its text, or anything about the document. Once admitted, the
    job's own outcome cannot be observed here or turned into a warning on the
    ingestion result.
    """
    outcome = None
    try:
        outcome = _INSIGHTS_QUEUE.enqueue(committed.document_id)
    except Exception:
        # Not bound, logged, re-raised or chained: only the fact is kept.
        outcome = None
    if outcome in (EnqueueOutcome.ACCEPTED, EnqueueOutcome.DUPLICATE):
        return
    # Raised outside the except block on purpose: with no active exception
    # context there is no implicit chaining, so the original error cannot
    # ride along on __context__ into any caller's log.
    print(f"[Upload] post-commit insights not queued: {_ENQUEUE_FAILURE_LABELS.get(outcome, 'admission failed')}")
    raise _PostCommitDispatchError("post-commit follow-up could not be started")


def _build_ingestion_service(extraction_mode: str = "standard") -> DocumentIngestionService:
    """Compose the verified ingestion service with this application's parts.

    connect_to_postgres is passed as the connection factory specifically
    because it already applies pgvector's register_vector() to every
    connection it returns; the service inserts each embedding as a plain
    list[float], which psycopg2 can only adapt to the `vector` type on a
    registered connection. No second, unregistered psycopg2 connection path
    is introduced anywhere in the upload flow.

    The extractor is the enhanced PDF extractor for the requested mode; DOCX,
    text and (without a configured worker) PDFs keep the standard parser.
    """
    return DocumentIngestionService(
        connection_factory=connect_to_postgres,
        embedding_provider=_EMBEDDING_PROVIDER,
        embedding_profile=_EMBEDDING_PROFILE,
        dispatch_after_commit=_dispatch_upload_followups,
        extractor=_build_document_extractor(extraction_mode),
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
    extraction: Optional[str] = Form(None),
    processing: Optional[str] = Form(None),
):
    """Thin transport adapter over DocumentIngestionService.

    This route no longer parses, chunks, embeds, or writes anything itself. It
    reads each uploaded file's ORIGINAL bytes exactly once, maps the form
    fields onto the ingestion contract, hands the batch to the service in the
    order the client sent it, and returns the service's batch envelope under
    the service's own aggregated HTTP status (200 / 207 / 422 / 503). Each
    document owns a separate transaction inside the service, so one file's
    failure can never erase a sibling that already committed.

    ``extraction`` is "standard" (default: native text, OCR for pages without
    usable text when the worker is configured) or "enhanced" (adds table
    structure extraction). ``processing`` is "sync" (default: the envelope is
    returned when the batch finishes; the work runs off the event loop) or
    "background" (HTTP 202 with a job id; poll GET /upload/jobs/{job_id}).
    """
    extraction_mode = (extraction or "standard").strip().lower()
    processing_mode = (processing or "sync").strip().lower()
    if extraction_mode not in EXTRACTION_MODES:
        raise HTTPException(422, "extraction must be 'standard' or 'enhanced'")
    if processing_mode not in _UPLOAD_PROCESSING_MODES:
        raise HTTPException(422, "processing must be 'sync' or 'background'")
    # Table extraction stays unavailable until its qualification passes and an operator
    # enables it (EXTRACTOR_ENABLE_TABLES=1). Scanned-page OCR does not depend on this.
    if extraction_mode == "enhanced" and not ExtractionSettings.from_env().tables_enabled:
        raise HTTPException(422, "Table extraction is not available on this server yet.")
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

    # Raises 401 for a missing token or one that matches no user and 503 when
    # the lookup cannot complete, so none of them reaches a file read or the
    # service: every stored document has an owner.
    uploader_id = _resolve_caller_id(auth_token)

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

    service = _build_ingestion_service(extraction_mode)

    if processing_mode == "background":
        def _run_batch():
            batch = service.ingest_batch(uploads)
            return batch.http_status, batch.to_dict()

        submitted = _UPLOAD_JOBS.submit(
            uploader_id,
            _run_batch,
            size_bytes=sum(len(u.content) for u in uploads if isinstance(u.content, bytes)),
            file_count=len(uploads),
        )
        if submitted.outcome is not SubmitOutcome.ACCEPTED:
            status_code, message = _UPLOAD_JOB_REFUSALS[submitted.outcome]
            headers = {"Retry-After": "60"} if status_code == 429 else None
            raise HTTPException(status_code, message, headers=headers)
        return JSONResponse(
            status_code=202,
            content={
                "job_id": submitted.job_id,
                "status": "queued",
                "status_url": f"/upload/jobs/{submitted.job_id}",
                "extraction": extraction_mode,
                "file_count": len(uploads),
            },
        )

    # Parsing, OCR/table calls and embedding are blocking; run them off the
    # event loop so other requests (status polls, /ask streams) stay responsive.
    batch = await run_in_threadpool(service.ingest_batch, uploads)
    return JSONResponse(status_code=batch.http_status, content=batch.to_dict())


@app.get("/upload/jobs/{job_id}")
async def upload_job_status(job_id: str, authorization: Optional[str] = Header(None)):
    """Status and, once finished, the batch envelope of the caller's own upload job.

    The caller is identified from ``Authorization: Bearer <token>``. A job that
    belongs to someone else is reported exactly like one that does not exist,
    so job ids cannot be probed. Jobs are kept in memory only: after a backend
    restart an unfinished or old job reads as not found, and the documents
    list is the source of truth for what was actually indexed.
    """
    caller_id = _resolve_caller_id(_bearer_token(authorization))
    view = _UPLOAD_JOBS.get(job_id, caller_id)
    if view is None:
        raise HTTPException(404, "Upload job not found or expired")
    return JSONResponse(status_code=200, content=view.to_dict())



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

    Fails closed. If no token is supplied, the connection cannot be opened, the
    lookup raises, or the token matches no user, the caller gets an unresolved
    scope and therefore no server-loaded evidence at all.

    There is no anonymous scope: documents stored without an owner (legacy
    rows) are not served to anyone through /ask, and are never attributed to a
    caller. The route also refuses a missing or unknown token up front
    (_resolve_caller_id); this check is the evidence path's own boundary.
    """
    if not auth_token:
        return AuthorizedScope.unresolved()

    connection = connect_to_postgres()
    if connection is None:
        return AuthorizedScope.unresolved()

    cursor = None
    try:
        cursor = connection.cursor()
        cursor.execute(_ACTIVE_USER_ID_BY_TOKEN, (auth_token,))
        row = cursor.fetchone()
        if not row or row[0] is None:
            return AuthorizedScope.unresolved()
        user_id = row[0]

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
            anonymous=False,
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
    caller cannot turn into a cross-user read. A scope without an owner never
    loads anything: ownerless (legacy) documents are served to no one.
    """
    if not scope.authorizes(document_id) or scope.user_id is None:
        return None

    connection = connect_to_postgres()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor()
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


# Versioned as CLAIM_SUPPORT_PROMPT_VERSION in release_metadata.py, separately
# from the answer prompt; tests/test_release_metadata.py pins this template's
# fingerprint, so editing it means bumping that version.
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


def _ollama_claim_verifier(claim: str, records, verifier) -> Optional[str]:
    """Local-only claim verification against the cited evidence.

    Reads each record's INTERNAL ``evidence_text`` -- the same text the answer
    model saw -- not the bounded public excerpt. Judging against the excerpt
    would mark a claim unsupported whenever its support happens to sit past the
    excerpt bound, which is exactly what a long selection or a complete short
    document looks like. The verifier's own result stays public-safe: it
    returns one status word, and the claim payload carries only citation ids.

    ``verifier`` is the answer snapshot's verifier call, captured before the
    answer streamed: the answer profile's exact tag, context window,
    keep_alive and think policy with the bounded verifier options and the
    profile's verifier read timeout. A request cannot change it, and it is
    never rebuilt from the registry here.
    Runs through the same local Ollama transport as generation and nothing
    else. A transport or model failure raises (``verifier_unavailable``). A
    reply that did not finish -- not ``done``, cut off at the output limit, or
    any completion reason other than ``"stop"`` -- or that is not exactly one
    status returns None (``verifier_malformed``). citation_generation turns
    either into ``unverified``, never ``supported``, and neither affects
    whether the answer itself completed. Only the ``response`` field is read,
    so separate thinking text is ignored. Each reply's completion is logged as
    counts and fixed words only (``describe_completion``), never its text.
    """
    evidence = "\n\n".join(
        f"[{record.citation_id}] {record.evidence_text}" for record in records
    )
    prompt = _CLAIM_VERIFIER_PROMPT.format(evidence=evidence, claim=claim.strip())
    response = post_ollama("/api/generate", verifier.ollama_payload(prompt), timeout=verifier.read_timeout_s)
    response.raise_for_status()
    body = response.json()
    print(verifier.describe_completion("verifier", body))
    # Raises with fixed text on a model error; None for an unfinished reply.
    return _claim_verifier_status(completed_generate_response(body))


def _installed_model_digest(model_tag: str, runtime_key: str = PRIMARY_RUNTIME) -> Optional[str]:
    """The manifest digest one runtime's Ollama lists for exactly this tag,
    or None.

    Read-only (``/api/tags`` on ``runtime_key``'s endpoint). Consulted per
    answer and once per helper job only for a profile that requires an
    identity check -- so ordinary control answers make no extra request --
    and by the switch coordinator for a switch's target, always on that
    profile's own runtime.
    """
    base_url = _runtime_base_url(runtime_key)
    response = requests.get(f"{base_url}/api/tags", timeout=(OLLAMA_CONNECT_TIMEOUT, 5))
    response.raise_for_status()
    for model in response.json().get("models", []):
        if model.get("name") == model_tag or model.get("model") == model_tag:
            return model.get("digest")
    return None


def _identity_lookup(runtime_key: str):
    """A one-tag installed-identity lookup bound to ``runtime_key``. The
    primary runtime keeps the original one-argument call; every other
    runtime is named explicitly, so a tag installed on one runtime can never
    vouch for another."""
    if runtime_key == PRIMARY_RUNTIME:
        return lambda tag: _installed_model_digest(tag)
    return lambda tag: _installed_model_digest(tag, runtime_key)


# Per-request model selection is retired: every answer uses the one shared
# selection an administrator switches (/admin/generation). Any non-null
# model_profile_id is refused for every caller -- an administrator too --
# before any lookup; the legacy ``model`` string is never read at all.
_MODEL_OVERRIDE_RETIRED = (
    "Per-request model selection is not supported. An administrator selects the shared answer model."
)

# Fixed refusals when the shared selection cannot produce this answer's
# snapshot. The client chose nothing, so each is a server-side 503, and no
# other model is ever used instead. Only the class of the error is used.
_MODEL_SELECTION_REFUSALS = {
    UnknownModelProfileError: (503, "The selected answer model is not available on this server."),
    ProfileNotQualifiedError: (503, "The selected answer model is not qualified for use."),
    ExperimentalProfileDisabledError: (503, "The selected experimental answer model is disabled on this server."),
    ModelRuntimeUnavailableError: (503, "The selected answer model's runtime is not configured on this server."),
    ModelIdentityMismatchError: (503, "The selected answer model is not installed with its qualified identity."),
    ModelIdentityUnavailableError: (503, "Could not verify the model. Please try again shortly."),
}

# Fixed admission refusals: busy before any evidence, model or database work.
_ADMISSION_REFUSALS = {
    AdmissionRefusal.BUSY: "The answer service is busy. Please retry in a moment.",
    AdmissionRefusal.UNRESOLVED: (
        "The answer service is holding capacity for model work whose outcome is unknown. Please retry later."
    ),
    AdmissionRefusal.SWITCHING: "The answer model is being switched. Please retry shortly.",
    AdmissionRefusal.UNAVAILABLE: "Answer generation is not available on this server right now.",
}

# Read-only identity lookups for qualified profiles run here: one dedicated
# thread, never the shared request pool, and never while a slot is waited for.
_IDENTITY_LOOKUPS = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-identity")


async def _admit_answer() -> GenerationTicket:
    """An admitted answer ticket, or a fixed 503 with ``Retry-After``.

    Waits on the event loop for at most the agreed two seconds, holding no
    thread. A refusal happens before scope SQL, planning, embedding, the
    health probe or any model call."""
    refusal = None
    try:
        return await acquire_answer(_GENERATION_LEDGER)
    except AdmissionRefused as exc:
        refusal = exc.reason
    print(f"[Ask] admission refused: {refusal.value}")
    # Raised outside the except block so nothing chains onto the HTTP error.
    raise HTTPException(503, _ADMISSION_REFUSALS[refusal], headers={"Retry-After": "2"})


async def _answer_generation(ticket: GenerationTicket) -> AnswerGenerationSnapshot:
    """The model, options and verifier for one admitted answer, fixed before
    it starts, from the shared selection its ticket captured.

    The unqualified control needs no lookup (observed digest None). A
    profile that requires an identity check (qualified or experimental) has
    its installed identity read for THIS answer, on its own runtime, on the
    dedicated identity thread. Every refusal is a fixed HTTP error raised
    before any evidence is read or any model is called; the caller returns
    the ticket's slot. No other model or runtime is ever tried instead."""
    refusal = category = None
    try:
        profile = select_shared_profile(_MODEL_REGISTRY, ticket.selection.profile_id)
        observed = None
        if profile.requires_identity_check:
            observed = await asyncio.wrap_future(
                _IDENTITY_LOOKUPS.submit(check_installed_identity, profile, _identity_lookup(profile.runtime_key))
            )
        snapshot = shared_generation_snapshot(
            profile, epoch=ticket.selection.epoch, observed_digest=observed, registry=_MODEL_REGISTRY
        )
        ticket.bind_snapshot(snapshot)
        return snapshot
    except ModelSelectionError as exc:
        category = type(exc).__name__
        refusal = _MODEL_SELECTION_REFUSALS.get(type(exc), (503, "Model selection is unavailable."))
    print(f"[Ask] shared model unavailable for this answer ({category})")
    raise HTTPException(*refusal)


def _new_answer_job(ticket: GenerationTicket, body) -> AnswerJob:
    """The answer's owned job, with its start lease armed on this loop."""
    job = AnswerJob(ledger=_GENERATION_LEDGER, ticket=ticket, body=body, settings=_DELIVERY_SETTINGS)
    job.arm_start_lease(asyncio.get_running_loop(), _ADMISSION_SETTINGS.unstarted_lease_s)
    return job


def _release_unstarted_answer(ticket: GenerationTicket) -> None:
    """Return an admitted answer's slot when no response will start it."""
    _GENERATION_LEDGER.release_unstarted(ticket)


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
    # Server-measured milestones for this answer's provenance, all relative
    # to this moment (monotonic, so wall-clock changes cannot skew them).
    timer = AnswerTimer()
    answer_parts = []
    entry_id = None
    # Decides the prompt rules and whether claims are graded; evidence
    # planning and authorization below are the same in every mode.
    answer_mode = AnswerMode(request.mode)
    # Retired: no request chooses a model, an administrator's included.
    if request.model_profile_id is not None:
        raise HTTPException(400, _MODEL_OVERRIDE_RETIRED)
    # Identified before anything streams: a missing or unknown token is a 401
    # and a failed lookup a 503, so an unidentified request reads no evidence,
    # calls no model, and writes no history row. The answer's history row
    # belongs to exactly this caller.
    caller_id = _resolve_caller_id(request.auth_token)
    # Admission: one ticket from the shared ledger, waited for on the event
    # loop (no thread held) for at most two seconds, else a fixed 503. The
    # ticket captured the shared selection -- profile and epoch -- when it
    # was granted, and from here on this answer owns it: its generation,
    # every verifier call and its persistence all run under it.
    ticket = await _admit_answer()
    # The answer model, its native options and the verifier are fixed once,
    # here, from the captured selection, before anything streams: nothing that
    # happens later -- a switch, a registry change -- can change what this
    # answer is generated or verified with. A refusal returns the slot.
    try:
        generation = await _answer_generation(ticket)
    except BaseException:
        _release_unstarted_answer(ticket)
        raise
    # Fixed with the model snapshot, for the same reason: the provenance of
    # this answer describes the release and embedding configuration it started
    # under. The trace id is generated here, server-side, for this attempt; it
    # is stored only with a successfully saved answer.
    trace_id = new_trace_id()
    release = _RELEASE
    embedding_profile = _EMBEDDING_PROFILE
    verifier_calls = 0

    def verify_claim(claim, records):
        nonlocal verifier_calls
        # After known abandonment, past the job's budget, or once an earlier
        # call of this answer was never observed to finish, no verifier call
        # is dispatched: the claim stays unverified.
        if not current_dispatch_permitted():
            raise DispatchRefused("no further model call is permitted for this answer")
        verifier_calls += 1
        return _ollama_claim_verifier(claim, records, generation.verifier)

    def build_evidence():
        """Plan the evidence and register its citations. Authorization-scoped."""
        scope = _resolve_authorized_scope(request.auth_token)
        planner = _build_answer_evidence_planner(scope)
        planning = planner.plan(_evidence_request_from_ask(request, scope), scope)
        registry = CitationRegistry.from_bundle(planning.bundle, _CITATION_LIMITS)
        return planning, registry

    def stream_generate(job):
        # Runs on this answer's own job thread (answer_jobs.AnswerJob), never
        # on the event loop or the shared request pool. ``job`` reports known
        # delivery abandonment and the whole-job budget.
        nonlocal answer_parts, entry_id
        yield '{"type": "delta", "text": ""}\n'
        if job.abandoned:
            # Delivery ended before any evidence was read: nothing is planned,
            # generated or saved, and the slot returns as never dispatched.
            return

        stream_error = None
        try:
            planning, registry = build_evidence()

            # Security containment (PR #54): no answer mode reads the
            # knowledge base or injects it into the prompt, for any caller.
            # Its rows have no owner column and include answers and Q&A
            # derived from users' private documents, so neither a valid login
            # nor admin status gives a reliable boundary for serving them.
            # The prompt is built from the caller's authorized evidence only.
            if answer_mode is AnswerMode.DOCUMENT_QA:
                prompt = build_generation_prompt(request.question, registry)
            else:
                prompt = build_mode_prompt(answer_mode, request.question, registry)

            yield evidence_event(
                registry.to_dict()["citations"],
                mode=planning.mode.value,
                truncated=planning.bundle.truncated,
                warnings=planning.warnings,
                generation=generation.public_metadata(),
            )

            payload = generation.ollama_payload(prompt, stream=True)
        except EmbeddingProviderError:
            yield '{"type": "error", "code": "search_unavailable", "message": "Search is temporarily unavailable. Please try again shortly."}\n'
            yield '{"type": "done", "ok": false}\n'
            return
        except Exception as exc:
            print(f"[Ask] evidence planning failed: {type(exc).__name__}")
            yield '{"type": "error", "code": "evidence_unavailable", "message": "Document evidence is temporarily unavailable."}\n'
            yield '{"type": "done", "ok": false}\n'
            return

        if job.abandoned:
            # Abandoned before dispatch: no model call is sent.
            return
        if job.expired:
            yield error_event(**ANSWER_TIMEOUT_ERROR)
            yield done_event(ok=False)
            return

        # Only "response" text becomes the answer; thinking, error lines and
        # the final metrics record are handled by the reader (see
        # ollama_generation.py), which also survives split lines and split
        # multi-byte characters.
        generated = OllamaGenerateStream()
        try:
            timer.mark("generation_request_sent")
            # Per-profile, finite read timeout from the snapshot: a cold Gemma
            # load alone exceeded the 60 s Llama keeps. post_ollama clamps it to
            # the job's remaining budget and records the dispatch intent first.
            with post_ollama(
                "/api/generate", payload, stream=True, timeout=generation.answer_read_timeout_s
            ) as r:
                r.raise_for_status()
                for chunk in r.iter_content(decode_unicode=True, chunk_size=32):
                    for token in generated.feed(chunk):
                        timer.mark("first_answer_token")
                        answer_parts.append(token)
                        yield delta_event(token)
                    # A client that went away does not stop this read: the
                    # stream is drained to its terminal record so the call can
                    # be observed as finished. Only the job's budget stops it,
                    # and then the call's end is unobserved (its slot is held).
                    if job.expired:
                        break
                else:
                    for token in generated.close():
                        timer.mark("first_answer_token")
                        answer_parts.append(token)
                        yield delta_event(token)
                    job.observed_stream(generated)
            timer.mark("generation_complete")
        except DispatchRefused:
            # Nothing was sent: no admitted job may dispatch, or the durable
            # dispatch intent could not be recorded first.
            stream_error = True
            yield error_event(**NOT_DISPATCHED_ERROR)
        except Exception as e:
            stream_error = True
            response = getattr(e, "response", None)
            if response is not None:
                yield '{"type": "error", "code": "generation_http_error", "message": "The local model request failed. Check that the requested model is installed in Ollama."}\n'
            elif generation.runtime_key != PRIMARY_RUNTIME:
                # No fallback: the experimental model's own runtime failed.
                yield '{"type": "error", "code": "generation_unreachable", "message": "The experimental model\'s Ollama runtime is not reachable. Please retry later or ask an administrator."}\n'
            else:
                yield '{"type": "error", "code": "generation_unreachable", "message": "The local LLM server is not reachable. Start Ollama or update OLLAMA_BASE_URL / OLLAMA_PORT in .env."}\n'

        # Counts only, never content: lets an operator compare the prompt the
        # server evaluated with the requested context and spot output limits.
        print(
            f"[Ask] trace_id={trace_id} generation profile={generation.profile_id} "
            f"failure={generated.failure} done_reason={generated.done_reason} "
            f"prompt_eval_count={generated.metrics.get('prompt_eval_count')} "
            f"eval_count={generated.metrics.get('eval_count')} "
            f"thinking_chars={generated.thinking_chars}"
        )

        if job.expired and not stream_error:
            # The whole-job budget ran out while the answer streamed.
            stream_error = True
            yield error_event(**ANSWER_TIMEOUT_ERROR)

        if not stream_error and generated.failure is not None:
            # The server reported an error, a line was corrupt, the final
            # record never came, or the output limit cut the answer off. Text
            # may have streamed, but it is not a finished answer.
            stream_error = True
            code = (
                "generation_output_limit"
                if generated.failure == FAILURE_OUTPUT_LIMIT
                else "generation_incomplete"
            )
            yield error_event(**safe_error(code))

        if stream_error or not "".join(answer_parts).strip():
            # An interrupted or empty generation must never terminate as a
            # success: done carries ok=false and no entry id is written. A
            # reply with no answer text (for example thinking only, or only
            # whitespace) says so.
            if not stream_error:
                yield error_event(**safe_error("generation_empty"))
            yield done_event(ok=False)
            return

        if job.abandoned:
            # Delivery was abandoned (the client disconnected or stalled, or
            # the delivery bound overflowed) after a complete generation: no
            # verifier call is dispatched and no answer row is begun.
            return

        full_answer = "".join(answer_parts)

        citation_result = _resolve_citations(
            full_answer, registry, verify_claim, mode=answer_mode
        )
        timer.mark("verification_complete")
        yield verification_event(
            [claim.to_dict() for claim in citation_result.claims],
            invalid_citation_ids=citation_result.invalid_citation_ids,
            used_citation_ids=citation_result.used_citation_ids,
        )

        # The commit point: decided once. After known abandonment or past the
        # budget, no INSERT is begun; an INSERT begun here is completed and
        # kept even if the client goes away meanwhile.
        if not job.begin_persistence():
            if job.abandoned:
                return
            yield error_event(**ANSWER_TIMEOUT_ERROR)
            yield done_event(ok=False)
            return

        conn = None
        committed = False
        try:
            # Built only from the snapshot, the server's release metadata, the
            # configured embedding profile, server timing and the finished
            # stream -- nothing from the request. Built inside this try and
            # written in the same INSERT as the answer, so either both are
            # saved or neither is: a provenance failure saves no answer row.
            provenance_json = answer_provenance_json(build_answer_provenance(
                trace_id=trace_id,
                release=release,
                generation=generation,
                answer_mode=answer_mode,
                evidence_mode=planning.mode,
                embedding_profile=embedding_profile,
                timer=timer,
                stream=generated,
                verifier_calls=verifier_calls,
            ))

            conn = connect_to_postgres()
            c = conn.cursor()

            c.execute(
                """
                INSERT INTO chat_history
                    (ts, selected_text, question, answer, user_id, trace_id, answer_provenance)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    datetime.datetime.now().isoformat(),
                    request.selected_text or "",
                    request.question,
                    full_answer,
                    caller_id,
                    trace_id,
                    provenance_json,
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

        # Security containment (PR #54): the committed answer is NOT
        # auto-saved to the knowledge base. That table has no owner column and
        # is shared by every user, so an answer built from one user's private
        # documents would be served to others. auto_save_to_kb stays disabled
        # until the knowledge base is user-owned.

        yield done_event(ok=True)

    # The response owns the ticket from here: its ASGI call starts the job
    # thread (a start lease reclaims an answer whose response never starts),
    # and each NDJSON line it sends is bounded in bytes and in send time.
    try:
        return AdmittedAnswerResponse(
            _new_answer_job(ticket, stream_generate),
            media_type=STREAM_MEDIA_TYPE,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    except BaseException:
        _release_unstarted_answer(ticket)
        raise



@app.post("/history", response_model=List[HistoryItem])
async def get_history(request: HistoryRequest):
    """The caller's own most recent chat history.

    A missing or invalid token is a 401. There is no anonymous history: rows
    stored without an owner (legacy anonymous chats) are returned to no one.
    """
    conn = _open_db()
    try:
        c = conn.cursor()
        user_id = _user_id_for_token(c, request.token)
        c.execute(
            "SELECT id, ts, selected_text, question, answer FROM chat_history WHERE user_id = %s ORDER BY id DESC LIMIT 20",
            (user_id,),
        )
        rows = c.fetchall()
        return [
            HistoryItem(
                id=r[0], timestamp=r[1], selected_text=r[2], question=r[3], answer=r[4]
            )
            for r in rows
        ]
    except HTTPException:
        raise
    except Exception as e:
        print(f"[get_history] failed: {type(e).__name__}")
        raise HTTPException(500, "Could not load history")
    finally:
        conn.close()


@app.get("/documents")
async def get_documents(authorization: Optional[str] = Header(None)):
    """The caller's own documents' metadata.

    The caller comes from the Authorization header, like the other document
    routes; a missing, malformed, or invalid token is a 401. Other users'
    documents and ownerless (legacy) documents are never listed.
    """
    token = _bearer_token(authorization)
    conn = _open_db()
    try:
        c = conn.cursor()
        user_id = _user_id_for_token(c, token)
        c.execute("""SELECT id, filename, upload_timestamp, author, title, publication_date, source, doi_url,
                     (SELECT COUNT(*) FROM document_chunks WHERE document_id = documents.id)
                     FROM documents WHERE user_id = %s ORDER BY upload_timestamp DESC""",
                  (user_id,))
        rows = c.fetchall()
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"[get_documents] failed: {type(e).__name__}")
        raise HTTPException(500, "Could not list documents")
    finally:
        conn.close()


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
    cursor.execute(_ACTIVE_USER_ID_BY_TOKEN, (token,))
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
async def put_ratings(request: RatingRequest, authorization: Optional[str] = Header(None)):
    """Rate one of the caller's own answers.

    The caller comes from the Authorization header, and the UPDATE is scoped
    ``WHERE id = ... AND user_id = ...``, so a missing or invalid token is a
    401 and another user's (or an ownerless) history row is indistinguishable
    from a missing one (404) -- nothing is written in any of those cases.
    """
    token = _bearer_token(authorization)
    conn = _open_db()
    try:
        c = conn.cursor()
        user_id = _user_id_for_token(c, token)
        c.execute(
            "UPDATE chat_history SET rating = %s, comment = %s WHERE id = %s AND user_id = %s RETURNING id",
            (request.rating, request.comment, request.id, user_id),
        )
        if not c.fetchone():
            conn.rollback()
            raise HTTPException(404, "Chat ID not found")
        conn.commit()
        return {"message": "Rating updated", "id": request.id}
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"[put_ratings] failed: {type(e).__name__}")
        raise HTTPException(500, "Could not save the rating")
    finally:
        conn.close()


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
async def google_login():
    """Google sign-in is refused for this release, before any token check.

    The previous flow verified the Google ID token and then returned the
    token of whichever account had ``username`` equal to the Google email.
    Registration accepts any username, so an account pre-registered with
    someone else's email address as its username would receive that
    person's Google sign-ins. Sign-in stays refused until accounts are bound
    to a verified Google identity (the token's ``sub``). Username/password
    login is unaffected. No Google token is verified and no account is read
    or created here.
    """
    raise HTTPException(
        403,
        "Google sign-in is temporarily unavailable. Please sign in with your username and password.",
    )


@app.post("/submit_correction")
async def submit_correction(request: CorrectionRequest, authorization: Optional[str] = Header(None)):
    """Correct one of the caller's own answers, in the caller's own history.

    The caller comes from the Authorization header, and both the SELECT and
    the UPDATE are scoped ``WHERE id = ... AND user_id = ...``, so a missing
    or invalid token is a 401 and another user's (or an ownerless) history row
    is indistinguishable from a missing one (404) -- nothing is written in any
    of those cases.

    Security containment (PR #54): the correction is no longer copied into
    the knowledge base, for any caller. That table has no owner column and is
    shared by every user, so the question and original answer -- built from
    the caller's private documents -- would be served to others. Admins curate
    the knowledge base through its explicit admin endpoints instead.
    """
    token = _bearer_token(authorization)
    conn = _open_db()
    try:
        c = conn.cursor()
        user_id = _user_id_for_token(c, token)
        c.execute(
            "SELECT id FROM chat_history WHERE id = %s AND user_id = %s",
            (request.chat_id, user_id),
        )
        if not c.fetchone():
            conn.rollback()
            raise HTTPException(404, "Chat ID not found")
        c.execute(
            "UPDATE chat_history SET answer = %s, comment = %s WHERE id = %s AND user_id = %s RETURNING id",
            (request.corrected_answer, request.comment, request.chat_id, user_id),
        )
        if not c.fetchone():
            conn.rollback()
            raise HTTPException(404, "Chat ID not found")
        conn.commit()
        return {
            "message": "Correction saved to your history",
            "chat_id": request.chat_id,
        }
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        print(f"[submit_correction] failed: {type(e).__name__}")
        raise HTTPException(500, "Could not save the correction")
    finally:
        conn.close()


@app.get("/knowledge_base")
async def knowledge_base(authorization: Optional[str] = Header(None)):
    """Every knowledge-base entry, for the admin knowledge-base page.

    Security containment (PR #54): admin-only, like the knowledge base's write
    endpoints. The table has no owner column and holds answers and Q&A derived
    from users' private documents, so it must not be readable by every signed-
    in user. The token comes from an ``Authorization: Bearer <token>`` header,
    not a ``?token=`` query parameter that would land in browser history and
    access logs. The admin check runs before any knowledge-base query: a
    missing or malformed header is a 401 and an invalid or non-admin token a
    403, with no row read.
    """
    token = _bearer_token(authorization)
    conn = _open_db()
    try:
        c = conn.cursor()
        _require_admin(c, token)
        c.execute("""SELECT id, question, original_answer, corrected_answer, created_at,
                            chat_history_id, corrected_by, COALESCE(usage_count,0), context_text,
                            COALESCE(source_type, 'document')
                     FROM knowledge_base ORDER BY COALESCE(usage_count,0) DESC, id DESC""")
        rows = c.fetchall()
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
    except HTTPException:
        raise
    except Exception as e:
        print(f"[knowledge_base] failed: {type(e).__name__}")
        raise HTTPException(500, "Could not load the knowledge base")
    finally:
        conn.close()


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


_HELPER_BUSY = "Model capacity is busy or unavailable. Please retry shortly."


async def _run_interactive_helper(ticket: GenerationTicket, work):
    """Run ``work`` on a dedicated thread that owns ``ticket`` and releases it
    exactly once, by what its model calls showed; await the result without
    holding a request-pool thread. If the caller stops waiting, the job still
    finishes and releases its own ticket."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def settle(value) -> None:
        if not future.done():
            future.set_result(value)

    def run() -> None:
        result = []
        try:
            result = work()
        except Exception:
            print("[KB] Interactive helper job failed")
        finally:
            _GENERATION_LEDGER.release(ticket)
            try:
                loop.call_soon_threadsafe(settle, result)
            except RuntimeError:
                pass

    try:
        threading.Thread(target=run, name="kb-import-helper", daemon=True).start()
    except Exception:
        # The thread never ran, so nothing was dispatched.
        _GENERATION_LEDGER.release(ticket)
        raise HTTPException(503, _HELPER_BUSY, headers={"Retry-After": "2"}) from None
    return await future


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

    # The Q&A model calls are an interactive helper job: admitted at once or
    # refused at once (never an open-ended wait), then run on a dedicated
    # thread that owns the ticket -- never on the event loop.
    ticket = _GENERATION_LEDGER.try_acquire_helper()
    if ticket is None:
        raise HTTPException(503, _HELPER_BUSY, headers={"Retry-After": "2"})
    pairs = await _run_interactive_helper(
        ticket, lambda: _generate_qa_pairs_from_text(text, domain, recorder=ticket)
    )
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
        c.execute(_ACTIVE_ADMIN_FLAG_BY_TOKEN, (token,))
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
        c.execute(_ACTIVE_ADMIN_FLAG_BY_TOKEN, (token,))
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
        c.execute(_ACTIVE_ADMIN_FLAG_BY_TOKEN, (token,))
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
        c.execute("SELECT id, username FROM users WHERE token = %s AND COALESCE(is_active, 1) <> 0", (token,))
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
    cursor.execute(_ACTIVE_ADMIN_FLAG_BY_TOKEN, (token,))
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
            SELECT ch.id, ch.user_id, u.username, ch.ts, ch.question, ch.answer, ch.rating, ch.comment,
                   ch.trace_id, ch.answer_provenance
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
                    # Admin-only. Null on rows saved before provenance was
                    # recorded: "not recorded", never the current release,
                    # model, or prompt.
                    "trace_id": str(r[8]) if r[8] is not None else None,
                    "answer_provenance": r[9],
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
    """Delete a user account that owns no documents or chats.

    A user who still owns records is refused with 409 and nothing changes.
    Clearing the ownership link (the previous behaviour) moved that user's
    private documents and chats into the ownerless pool, and deleting them is
    a data-retention decision this route does not make. Suspending the account
    keeps the records attached to their owner."""
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
        if chat_count or doc_count:
            conn.close()
            raise HTTPException(
                409,
                f"This user still owns {doc_count} document(s) and {chat_count} chat(s). "
                "Suspend the account instead; deleting it would detach those records from their owner.",
            )
        c.execute("DELETE FROM users WHERE id = %s", (user_id,))
        _log_admin_action(c, token, "delete_user", target_id=user_id, target_username=target_row[1],
                           detail="account removed; it owned no documents or chats")
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


# Documents still without an insights row, newest first. Ids only: the queue
# never holds content or filenames, and each job reloads its own row.
_PENDING_INSIGHT_IDS_SQL = """
    SELECT d.id
    FROM documents d
    LEFT JOIN document_insights di ON di.document_id = d.id
    WHERE di.id IS NULL
    ORDER BY d.upload_timestamp DESC
    LIMIT %s
"""
_INSIGHTS_BACKFILL_UNAVAILABLE = "Insight analysis is temporarily unavailable. Please try again shortly."


@app.post("/admin/document_insights/analyze_pending")
async def admin_analyze_pending_documents(token: Optional[str] = None, max_docs: int = 15):
    """Backfill insight extraction for documents uploaded before this feature
    existed (or any that failed the first time). Queues the ids of up to
    ``max_docs`` (1..50) documents that still have no insights row and returns
    immediately -- the admin refreshes the Insights tab a little later to see
    results land.

    ``queued`` counts only ids newly accepted now; ``already_pending`` counts
    ids already queued, running or unresolved, and ``capacity_rejected`` ids
    turned away because the queue was full. A database or queue that cannot
    serve the request is a fixed 503 with nothing queued; 401/403 are kept."""
    conn = None
    cursor = None
    denied = None
    pending_ids = None
    try:
        conn = connect_to_postgres()
        if conn is not None:
            cursor = conn.cursor()
            _require_admin(cursor, token)
            cursor.execute(_PENDING_INSIGHT_IDS_SQL, (max(1, min(max_docs, 50)),))
            pending_ids = [row[0] for row in cursor.fetchall()]
    except HTTPException as exc:
        denied = exc
    except Exception:
        pending_ids = None
    _close_db_quietly(cursor, conn)
    if denied is not None:
        raise denied
    admission = None
    if pending_ids is not None:
        try:
            admission = _INSIGHTS_QUEUE.enqueue_batch(pending_ids)
        except Exception:
            admission = None
    if admission is None:
        print("[Insights] Pending analysis could not be queued")
        raise HTTPException(503, _INSIGHTS_BACKFILL_UNAVAILABLE)
    return {
        "queued": admission.accepted,
        "already_pending": admission.already_pending,
        "capacity_rejected": admission.capacity_rejected,
    }


@app.get("/admin/system_status")
async def admin_system_status(token: Optional[str] = None):
    """Operational health for the admin dashboard: is the LLM backend reachable,
    which models are installed, and is Postgres up.

    Only a verified admin whose authorization connection also closed cleanly
    reaches the health checks. A missing token is 401 and an untrusted one
    403; any failure of the authorization lookup itself -- including a close
    failure after it -- is a fixed 503 that never reaches the probes, the
    endpoint lookup or a model HTTP call, and never hides an established
    401/403."""
    conn = None
    denied = None
    verified = False
    try:
        conn = connect_to_postgres()
        if conn is not None:
            _require_admin(conn.cursor(), token)
            verified = True
    except HTTPException as exc:
        denied = exc
    except Exception:
        verified = False
    closed = True
    if conn is not None:
        try:
            conn.close()
        except Exception:
            closed = False
    if denied is not None:
        raise denied
    if not (verified and closed):
        print("[Admin] System status authorization could not be verified")
        raise HTTPException(503, "Could not verify your session. Please try again shortly.")

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


def _verify_generation_admin(authorization: Optional[str]) -> None:
    """Admin authorization for the shared-model routes, before any inventory,
    coordinator or other privileged work.

    The token comes from ``Authorization: Bearer`` (never a query string,
    which would reach browser history and access logs) and is checked with
    the same active-account admin lookup every admin route uses; nothing the
    browser says about its role is read. A missing token is 401, a non-admin
    403, and a lookup that cannot complete -- including a failed close -- a
    fixed 503."""
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(401, "Unauthorized")
    conn = None
    denied = None
    verified = False
    try:
        conn = connect_to_postgres()
        if conn is not None:
            _require_admin(conn.cursor(), token)
            verified = True
    except HTTPException as exc:
        denied = exc
    except Exception:
        verified = False
    closed = True
    if conn is not None:
        try:
            conn.close()
        except Exception:
            closed = False
    if denied is not None:
        raise denied
    if not (verified and closed):
        print("[Admin] Generation authorization could not be verified")
        raise HTTPException(503, "Could not verify your session. Please try again shortly.")


@app.get("/admin/generation")
async def admin_generation_status(authorization: Optional[str] = Header(None)):
    """The shared answer model as this process knows it: ownership, the
    committed active profile and epoch, any switch in progress or the last
    outcome, the registered choices with their eligibility, and admission
    counts. In-memory only -- no database read, inventory or model call -- so
    it stays responsive while generation is busy."""
    _verify_generation_admin(authorization)
    return JSONResponse(_GENERATION_RUNTIME.status(), headers={"Cache-Control": "no-store"})


@app.post("/admin/generation/select")
async def admin_generation_select(request: GenerationSelectRequest, authorization: Optional[str] = Header(None)):
    """Ask the coordinator to switch the shared answer model to ``profile_id``.
    202 when queued (the switch then runs on the coordinator's own thread;
    poll /admin/generation for its outcome); 400 unknown profile; 409 not
    qualified, already active, a switch already pending, or unresolved model
    work; 503 when this process does not own generation. Never falls back to
    another model."""
    _verify_generation_admin(authorization)
    outcome = _GENERATION_RUNTIME.request_switch(request.profile_id)
    if not outcome.accepted:
        raise HTTPException(outcome.status_code, outcome.detail)
    return JSONResponse(
        status_code=202,
        content={"switch_id": outcome.switch_id, "status": _GENERATION_RUNTIME.status()},
        headers={"Cache-Control": "no-store"},
    )


@app.post("/admin/generation/abort")
async def admin_generation_abort(authorization: Optional[str] = Header(None)):
    """Abort the switch in progress (or one a previous process left
    interrupted). 202 when the abort is requested; the coordinator then moves
    the persisted selection back to the unchanged active profile with a new
    epoch, which fences any late commit. 409 when there is nothing to abort,
    the switch already committed, or it is committing right now."""
    _verify_generation_admin(authorization)
    outcome = _GENERATION_RUNTIME.abort_switch()
    if not outcome.accepted:
        raise HTTPException(outcome.status_code, outcome.detail)
    return JSONResponse(
        status_code=202,
        content={"switch_id": outcome.switch_id, "status": _GENERATION_RUNTIME.status()},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/test")
async def test_endpoint():
    return {"message": "SynergeReader API is running successfully!"}


@app.get("/version")
async def get_version():
    """The deployed release identity, for the UI and the post-deploy check.

    Public and read-only (served as /api/version behind nginx). The body is
    exactly ReleaseMetadata.public_version(): version and build commit only --
    no model inventory, profile names, paths, environment values, database
    details, or endpoints. ``no-store`` keeps a browser or proxy from showing
    a previous deployment's version after a release.
    """
    return JSONResponse(_RELEASE.public_version(), headers={"Cache-Control": "no-store"})


# ------------------- Startup -------------------



@app.post("/convert-docx")
async def convert_docx_to_pdf():
    """DOCX-to-PDF conversion is disabled for this release.

    The route had no authentication and no frontend caller, ran LibreOffice
    for anyone, and read its output from a path built from the raw client
    filename -- which a crafted name could point at an existing ``.pdf``
    outside the temporary directory. The request body is not read: no file
    is accepted, written, converted, or opened here.
    """
    raise HTTPException(403, "DOCX conversion is disabled.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
