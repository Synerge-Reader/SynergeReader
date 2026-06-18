from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from document_parser import extract_text_from_upload, ExtractionError, sanitize_filename
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from schemas import AskRequest, AskResponse, CorrectionRequest, RatingRequest,GoogleLoginRequest,LoginRequest,RegisterRequest
from schemas import HistoryItem,HistoryRequest, KnowledgeItem,KnowledgeInsertRequest,ForgotPasswordRequest
import os
import string
import datetime
import re
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from dbSetup import init_db,connect_to_postgres,test_postgres_connection
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
from fastapi.responses import Response

load_dotenv()

app = FastAPI(title="SynergeReader API", version="2.0.0")

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
OLLAMA_EMBED_CONCURRENCY = int(os.getenv("OLLAMA_EMBED_CONCURRENCY", "4"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto").strip()
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "SynergeReader")
_ACTIVE_OLLAMA_BASE_URL = None
_OLLAMA_HEALTH_CHECKED_AT = 0
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
resend.api_key = os.getenv("EMAIL_KEY")
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


def stream_openrouter_chat(messages: list[dict], model: str = OPENROUTER_MODEL):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_TITLE,
    }
    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    with requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=(10, 90),
    ) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if raw_line.startswith("data: "):
                data = raw_line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except Exception:
                    continue


def chunk_text(text: str, max_chunk_size: int = 500) -> list:
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    current = []

    for word in words:
        current_size = sum(len(w) for w in current) + len(current) - 1
        if current_size + len(word) + 1 <= max_chunk_size:
            current.append(word)
        else:
            if current:
                chunks.append(" ".join(current))
            current = [word]
    if current:
        chunks.append(" ".join(current))
    return chunks


def embed_chunks(
    chunks: List[str], model: str = "nomic-embed-text:v1.5"
) -> List[List[float]]:
    if not chunks:
        return []

    def fetch_embeddings(endpoint: str, payload: dict):
        resp = post_ollama(endpoint, payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data:
            return data["embeddings"]
        if "embedding" in data:
            return [data["embedding"]]
        return []

    try:
        embeddings = fetch_embeddings(
            "/api/embed",
            {"model": model, "input": chunks, "keep_alive": OLLAMA_KEEP_ALIVE},
        )
        if len(embeddings) == len(chunks):
            return embeddings
    except Exception:
        pass

    def embed_one(chunk: str) -> List[float]:
        try:
            embeddings = fetch_embeddings(
                "/api/embed",
                {"model": model, "input": chunk, "keep_alive": OLLAMA_KEEP_ALIVE},
            )
            if embeddings:
                return embeddings[0]
        except Exception:
            pass

        try:
            resp = post_ollama(
                "/api/embeddings",
                {"model": model, "prompt": chunk, "keep_alive": OLLAMA_KEEP_ALIVE},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding", data.get("embeddings", [[0.0] * 768])[0])
        except Exception:
            return [0.0] * 768

    if len(chunks) == 1:
        return [embed_one(chunks[0])]

    max_workers = max(1, min(OLLAMA_EMBED_CONCURRENCY, len(chunks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(embed_one, chunks))


def get_relevant_chunks(
    question: str, top_k: int = 3, document_names: Optional[List[str]] = None
) -> List[dict]:
    """Get relevant chunks ranked by embedding similarity."""
    conn = None
    try:
        conn = connect_to_postgres()
        if conn is None:
            return []
        c = conn.cursor()
        question_embedding = embed_chunks([question])[0]
        if document_names:
            c.execute(
                """
                SELECT
                    dc.chunk_text,
                    1 - (dc.embedding <=> %s::vector) AS similarity
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE dc.embedding IS NOT NULL
                  AND d.filename = ANY(%s)
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
                """,
                (question_embedding, document_names, question_embedding, top_k),
            )
        else:
            c.execute(
                """
                SELECT
                    dc.chunk_text,
                    1 - (dc.embedding <=> %s::vector) AS similarity
                FROM document_chunks dc
                WHERE dc.embedding IS NOT NULL
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
                """,
                (question_embedding, question_embedding, top_k),
            )
        rows = c.fetchall()

        scored = []
        for text, similarity in rows:
            scored.append({"text": text, "similarity": float(similarity)})

        return scored
    except Exception as e:
        print(f"Error in get_relevant_chunks: {e}")
        return []
    finally:
        if conn is not None:
            conn.close()


def get_documents_by_filenames(document_names: List[str]) -> List[dict]:
    if not document_names:
        return []

    conn = None
    try:
        conn = connect_to_postgres()
        if conn is None:
            return []
        c = conn.cursor()
        c.execute(
            """
            SELECT filename, title, content
            FROM documents
            WHERE filename = ANY(%s)
            """,
            (document_names,),
        )
        rows = c.fetchall()
        return [
            {"filename": r[0], "title": r[1], "content": r[2]}
            for r in rows
        ]
    except Exception as e:
        print(f"Error retrieving documents by filename: {e}")
        return []
    finally:
        if conn is not None:
            conn.close()


def is_summary_question(question: str) -> bool:
    normalized = question.lower()
    summary_markers = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "briefly",
        "in short",
        "1-2 sentences",
        "one or two sentences",
        "write the summary",
    ]
    return any(marker in normalized for marker in summary_markers)


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
        # Embed the incoming question
        q_emb = embed_chunks([question])
        if not q_emb or not q_emb[0]:
            return []
        q_vec = q_emb[0]

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

        # Embed the question
        q_emb = embed_chunks([question])
        if not q_emb or not q_emb[0]:
            return
        q_vec = q_emb[0]

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


def generate_kb_from_document(doc_id: int, filename: str, text: str):
    """
    After a document is uploaded, use the LLM to generate Q&A pairs
    from the document text and seed them into the Knowledge Base.
    Called in a background daemon thread — never blocks the upload response.
    Uses Q:/A: plain-text format instead of JSON for reliable parsing.
    """
    try:
        if len(text.strip()) < 200:
            return

        try:
            base_url = get_active_ollama_base_url()
        except Exception:
            print(f"[KB] Ollama not reachable — skipping KB generation for {filename}")
            return

        # Find first available text-generation model
        generation_models = [
            "saul-instruct:latest", "llama3.1:latest", "llama3.1:8b",
            "llama3:latest", "mistral:latest", "qwen2.5:latest",
            "qwen2.5:7b", "phi3:latest", "gemma2:latest", "gemma:latest",
        ]
        active_model = None
        for model in generation_models:
            try:
                test = requests.post(
                    f"{base_url}/api/generate",
                    json={"model": model, "prompt": "Say OK", "stream": False},
                    timeout=(3, 20),
                )
                if test.status_code == 200 and test.json().get("response"):
                    active_model = model
                    print(f"[KB] Using model '{model}' for KB generation")
                    break
            except Exception:
                continue

        if not active_model:
            print(f"[KB] No text-generation model available for {filename}")
            return

        # Split document into 2 segments for broader coverage
        mid = min(len(text) // 2, 4000)
        excerpts = [
            text[:4000].strip(),
            text[mid:mid + 4000].strip(),
        ]

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
                    json={
                        "model": active_model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.2,
                    },
                    timeout=(5, 120),
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()
                print(f"[KB] Raw output part {i+1} (first 300 chars): {raw[:300]}")

                # Parse Q:/A: pairs
                # Split on lines starting with Q:
                current_q = None
                current_a_lines = []
                for line in raw.splitlines():
                    line = line.strip()
                    if line.lower().startswith("q:"):
                        # Save previous pair if any
                        if current_q and current_a_lines:
                            all_pairs.append({
                                "question": current_q,
                                "answer": " ".join(current_a_lines).strip()
                            })
                        current_q = line[2:].strip()
                        current_a_lines = []
                    elif line.lower().startswith("a:") and current_q:
                        current_a_lines = [line[2:].strip()]
                    elif current_a_lines and line:
                        # Continuation of the answer
                        current_a_lines.append(line)

                # Don't forget last pair
                if current_q and current_a_lines:
                    all_pairs.append({
                        "question": current_q,
                        "answer": " ".join(current_a_lines).strip()
                    })

                print(f"[KB] Parsed {len(all_pairs)} pairs so far after part {i+1}")

            except Exception as e:
                print(f"[KB] LLM call failed for excerpt {i+1}: {e}")
                continue

        if not all_pairs:
            print(f"[KB] No Q&A pairs extracted from {filename}")
            return

        conn = connect_to_postgres()
        c = conn.cursor()
        saved = 0
        seen = set()

        for pair in all_pairs:
            q = pair.get("question", "").strip()
            a = pair.get("answer", "").strip()

            if not q or not a or len(a) < 20:
                print(f"[KB] Skip (too short/empty): q={q[:40]}")
                continue
            if q.lower() in seen:
                print(f"[KB] Skip (duplicate in batch): {q[:50]}")
                continue
            seen.add(q.lower())

            try:
                q_emb = embed_chunks([q])
                q_vec = q_emb[0] if q_emb else None
            except Exception:
                q_vec = None

            c.execute(
                """
                INSERT INTO knowledge_base
                  (question, original_answer, corrected_answer, created_at,
                   context_text, corrected_by, usage_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, 0, %s)
                """,
                (q, a, a,
                 datetime.datetime.now().isoformat(),
                 filename, f"Auto-generated from: {filename}", q_vec)
            )
            saved += 1
            print(f"[KB] ✓ Inserted pair {saved}: {q[:60]}")

        conn.commit()
        conn.close()
        print(f"[KB] ✓ Final: saved {saved}/{len(all_pairs)} KB entries from: {filename}")

    except Exception as e:
        print(f"[KB] Document KB generation failed (non-critical): {e}")


async def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode()



@app.post("/upload")
async def upload_documents(
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    author: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    publication_date: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    doi_url: Optional[str] = Form(None),
):
    if file and files:
        upload_list = [file] + files
    elif files:
        upload_list = files
    elif file:
        upload_list = [file]
    else:
        raise HTTPException(400, "No files provided")

    results = []
    for f in upload_list:
        try:
            content = await f.read()
            safe_filename = sanitize_filename(f.filename)
            try:
                result = extract_text_from_upload(safe_filename, content)
                text = result.text
            except ExtractionError as e:
                raise HTTPException(status_code=e.http_status, detail=e.user_message)

            if not text.strip():
                results.append({"error": "Empty file", "filename": f.filename})
                continue

            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)

            conn = connect_to_postgres()
            if conn is None:
                raise HTTPException(500, "Failed to connect to PostgreSQL")
            try:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO documents
                    (filename, upload_timestamp, content, author, title, publication_date, source, doi_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        f.filename,
                        datetime.datetime.now().isoformat(),
                        text,
                        author,
                        title,
                        publication_date,
                        source,
                        doi_url,
                    ),
                )

                doc_id = c.fetchone()[0]

                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    c.execute(
                        """
                        INSERT INTO document_chunks
                        (document_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (doc_id, chunk, i, emb),
                    )

                conn.commit()
            finally:
                conn.close()

            results.append(
                {
                    "message": "Uploaded",
                    "filename": f.filename,
                    "document_id": doc_id,
                    "chunks_count": len(chunks),
                }
            )

            # Auto-generate KB entries from this document in the background
            try:
                from threading import Thread
                Thread(
                    target=generate_kb_from_document,
                    args=(doc_id, f.filename, text),
                    daemon=True
                ).start()
            except Exception:
                pass

        except Exception as e:
            results.append({"error": str(e), "filename": f.filename})

    return results



@app.post("/ask")
async def ask_question(request: AskRequest):
    system_prompt = (
        "You are SynergeReader, a document assistant. "
        "Answer only from the provided context when possible. "
        "If the context is insufficient, say what is missing instead of guessing. "
        "Do not include internal tags, JSON, or the words CONTEXT/QUESTION in the answer."
    )

    answer_parts = []
    entry_id = None
    selected_items = list(request.selections or [])
    raw_selected_text = (request.selected_text or "").strip()
    selected_document_names = []

    for selection in selected_items:
        if selection.document_name and selection.document_name not in selected_document_names:
            selected_document_names.append(selection.document_name)

    if request.active_document_name and request.active_document_name not in selected_document_names:
        selected_document_names.append(request.active_document_name)

    def build_context() -> tuple[str, List[dict], float, str]:
        if selected_items:
            prompt_text = "\n\n---\n\n".join(
                f"[Selection {index + 1} from: {selection.document_name}]\n{selection.text}"
                for index, selection in enumerate(selected_items)
            )
            return prompt_text, [{"text": prompt_text, "similarity": 1.0}], 1.0, "selected_text"

        if raw_selected_text:
            return raw_selected_text, [{"text": raw_selected_text, "similarity": 1.0}], 1.0, "selected_text"

        if request.active_document_name:
            documents = get_documents_by_filenames([request.active_document_name])
            if documents:
                document = documents[0]
                document_text = document["content"] or ""
                if len(document_text) > 14000:
                    document_text = (
                        document_text[:14000]
                        + "\n\n[Truncated to keep the prompt responsive.]"
                    )
                display_name = document["title"] or document["filename"]
                prompt_text = (
                    f"Document title: {display_name}\n"
                    f"Document file: {document['filename']}\n\n"
                    f"{document_text}"
                )
                return prompt_text, [{"text": prompt_text, "similarity": 1.0}], 1.0, "active_document"

        scoped_names = [
            name for name in selected_document_names if name and name != request.active_document_name
        ]
        if request.active_document_name and request.active_document_name not in scoped_names:
            scoped_names.insert(0, request.active_document_name)

        if scoped_names and is_summary_question(request.question):
            scoped_documents = get_documents_by_filenames(scoped_names)
            if scoped_documents:
                parts = []
                for document in scoped_documents:
                    document_text = document["content"] or ""
                    if len(document_text) > 12000:
                        document_text = (
                            document_text[:12000]
                            + "\n\n[Truncated to keep the prompt responsive.]"
                        )
                    display_name = document["title"] or document["filename"]
                    parts.append(
                        f"Document title: {display_name}\n"
                        f"Document file: {document['filename']}\n\n"
                        f"{document_text}"
                    )
                prompt_text = "\n\n---\n\n".join(parts)
                return prompt_text, [{"text": part, "similarity": 1.0} for part in parts], 1.0, "summary_document"

        if scoped_names:
            context_chunks = get_relevant_chunks(
                request.question, top_k=4, document_names=scoped_names
            )
        else:
            context_chunks = get_relevant_chunks(request.question, top_k=4)

        prompt_text = ""
        for chunk_data in context_chunks:
            prompt_text += f"\n\n{chunk_data['text']}"

        best_similarity = max(
            (chunk["similarity"] for chunk in context_chunks), default=0.0
        )
        return prompt_text, context_chunks, best_similarity, "retrieval"

    def build_prompt(prompt_text: str) -> str:
        return f"""<context>
{prompt_text}
</context>

<question>
{request.question}
</question>

Answer using only the provided context when it contains relevant information.
If the context is insufficient, say what is missing instead of guessing.
Do not include internal tags, metadata, JSON, or the words CONTEXT/QUESTION in the answer.
If a specific document or highlighted excerpt was provided, treat it as the primary source and do not mix in unrelated documents.
Keep the answer concise, structured, and directly responsive to the question."""

    def stream_generate():
        nonlocal answer_parts, entry_id
        yield "__SEARCHING__\n"

        stream_error = None
        try:
            prompt_text, context_chunks, best_similarity, context_source = build_context()

            # ── Knowledge Base injection ──────────────────────────────────
            kb_entries = get_relevant_knowledge_base(request.question, limit=3)
            kb_ids_fired = [e["id"] for e in kb_entries]
            if kb_entries:
                kb_block = "\n\n<knowledge_base_corrections>\n"
                for e in kb_entries:
                    kb_block += f"Q: {e['question']}\nA: {e['answer']}\n---\n"
                kb_block += "</knowledge_base_corrections>"
                prompt_text = prompt_text + kb_block
            # ─────────────────────────────────────────────────────────────

            prompt = build_prompt(prompt_text)
            fallback_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            context_data = {
                "context_chunks": [chunk_data["text"] for chunk_data in context_chunks],
                "similarity_score": best_similarity,
                "context_source": context_source,
                "active_document_name": request.active_document_name,
            }

            yield f"__CONTEXT__{json.dumps(context_data)}__\n\n"
            yield "__READY__\n"

            payload = {
                "model": request.model,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": True,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
        except Exception as e:
            yield f"__ERROR__Failed to build document context: {e}__"
            return

        try:
            with post_ollama("/api/generate", payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                buffer = ""
                token_count = 0
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
                                        token_count += 1
                                        answer_parts.append(token)
                                        yield token
                                except Exception as e:
                                    print(f"DEBUG: JSON parse error: {e}")
                                    continue
                # Handle any remaining buffered data
                if buffer:
                    try:
                        data = json.loads(buffer)
                        token = data.get("response", "")
                        if token:
                            token_count += 1
                            answer_parts.append(token)
                            yield token
                    except Exception:
                        pass
                print(f"DEBUG: Streaming complete. Total tokens: {token_count}")
        except Exception as e:
            stream_error = str(e)
            if OPENROUTER_API_KEY:
                try:
                    answer_parts.clear()
                    for token in stream_openrouter_chat(
                        fallback_messages, model=OPENROUTER_MODEL
                    ):
                        answer_parts.append(token)
                        yield token
                    stream_error = None
                except Exception as fallback_error:
                    print(f"DEBUG: OpenRouter fallback failed: {fallback_error}")
                    response = getattr(fallback_error, "response", None)
                    if response is not None:
                        yield f"__ERROR__LLM request failed with HTTP {response.status_code}. Check model access in OpenRouter.__"
                    else:
                        yield "__ERROR__The local LLM server is not reachable and OpenRouter fallback is unavailable.__"
            else:
                response = getattr(e, "response", None)
                if response is not None:
                    yield f"__ERROR__LLM request failed with HTTP {response.status_code}. Check that model '{request.model}' is installed in Ollama.__"
                else:
                    yield "__ERROR__The local LLM server is not reachable. Start Ollama or update OLLAMA_BASE_URL / OLLAMA_PORT in .env.__"

        # Increment KB usage counts for entries that fired this query
        if kb_ids_fired:
            increment_kb_usage(kb_ids_fired)

        if stream_error or not answer_parts:
            return

        full_answer = "".join(answer_parts)
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
            conn.close()

            yield f"\n\n__ENTRY_ID__{entry_id}__"

            # Auto-save this Q&A to the Knowledge Base in the background
            try:
                from threading import Thread
                Thread(
                    target=auto_save_to_kb,
                    args=(request.question, full_answer, "auto-query"),
                    daemon=True
                ).start()
            except Exception:
                pass

        except Exception:
            yield "__ERROR__Database error__"

    return StreamingResponse(
        stream_generate(),
        media_type="text/plain",
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


class DeleteDocumentRequest(BaseModel):
    filename: str


@app.delete("/documents/delete")
async def delete_document(request: DeleteDocumentRequest):
    """Delete a document and its chunks from the database"""
    conn = None
    try:
        conn = connect_to_postgres()
        if conn is None:
            raise HTTPException(500, "Failed to connect to PostgreSQL")
        c = conn.cursor()
        
        # First, find the document by filename
        c.execute("SELECT id FROM documents WHERE filename = %s", (request.filename,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            raise HTTPException(404, f"Document '{request.filename}' not found")
        
        doc_id = row[0]
        
        # Delete associated chunks first (foreign key constraint)
        c.execute("DELETE FROM document_chunks WHERE document_id = %s", (doc_id,))
        chunks_deleted = c.rowcount
        
        # Delete the document
        c.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        
        conn.commit()
        
        return {
            "message": "Document deleted successfully",
            "filename": request.filename,
            "document_id": doc_id,
            "chunks_deleted": chunks_deleted
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if conn is not None:
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
    conn = connect_to_postgres()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = %s OR email = %s", (request.username,request.email))
    if c.fetchone():
        conn.close()
        raise HTTPException(400, "Username or email already exists")
    hashed = await hash_password(request.password)
    token = secrets.token_hex(32)
    c.execute("INSERT INTO users (username, password, token,email) VALUES (%s, %s, %s)", (request.username, hashed, token,request.email))
    conn.commit()
    conn.close()

    params = { # improve on this message later
    "from": "Synerge <no-reply@synergereader.ai>",
    "to": [request.email],
    "subject": "Welcome to SynergeReader!",
    "html": """
        <h2>Welcome to SynergeReader!</h2>

        <p>Your account has been successfully created.</p>

        <p>SynergeReader helps you read, analyze, and interact with documents more intelligently — all in one place.</p>

        <p><b>What you can do next:</b></p>
        <ul>
            <li>Upload and read documents</li>
            <li>Ask questions and get contextual answers</li>
        </ul>

        <p>A user guide will be available inside the app.</p>

        <p>— The SynergeReader Team</p>
    """
    }

    email = resend.Emails.send(params)

    return {"message": "Registered", "token": token}


@app.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    conn = connect_to_postgres()
    c = conn.cursor()

    c.execute(
        "SELECT id, email FROM users WHERE email = %s",
        (request.email,)
    )
    user = c.fetchone()

    if not user:
        conn.close()
        raise HTTPException(400, "User not found")

    user_id, email = user

    new_password = ''.join(
        secrets.choice(string.ascii_letters + string.digits)
        for _ in range(12)
    )

    hashed = await hash_password(new_password)

    c.execute(
        "UPDATE users SET password = %s WHERE id = %s",
        (hashed, user_id)
    )
    conn.commit()
    conn.close()

    params = {
    "from": "Synerge <no-reply@synergereader.ai>",
    "to": [email],
    "subject": "Your new password",
    "html": f"""
        <p>Your password has been reset.</p>
        <p><b>New password:</b> {new_password}</p>
        <p>Please log in and change it immediately.</p>
    """
}

    email_response = resend.Emails.send(params)

    return {"message": "New password sent to your email"}


@app.post("/login")
async def login(request: LoginRequest):
    conn = connect_to_postgres()
    c = conn.cursor()
    c.execute("SELECT password, token FROM users WHERE username = %s", (request.username,))
    row = c.fetchone()
    conn.close()
    if not row or not bcrypt.checkpw(request.password.encode(), row[0].encode()):
        raise HTTPException(400, "Invalid username or password")
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
        # Embed the question for semantic matching
        try:
            q_emb = embed_chunks([question])
            q_vec = q_emb[0] if q_emb else None
        except Exception:
            q_vec = None

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
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/knowledge_base")
async def knowledge_base():
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("""SELECT id, question, original_answer, corrected_answer, created_at,
                            chat_history_id, corrected_by, COALESCE(usage_count,0)
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
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/knowledge_base")
async def add_knowledge(request: KnowledgeInsertRequest):
    """Add knowledge items directly to knowledge base (admin/manual entry)"""
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        for item in request.items:
            try:
                q_emb = embed_chunks([item.question])
                q_vec = q_emb[0] if q_emb else None
            except Exception:
                q_vec = None
            c.execute(
                "INSERT INTO knowledge_base (question, original_answer, corrected_answer, created_at, context_text, corrected_by, usage_count, embedding) VALUES (%s, %s, %s, %s, %s, %s, 0, %s)",
                (item.question, "", item.answer, datetime.datetime.now().isoformat(), item.source or "", "Manual Entry", q_vec)
            )
        conn.commit()
        conn.close()
        return {"message": f"{len(request.items)} knowledge items added"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/knowledge_base/{entry_id}")
async def delete_knowledge(entry_id: int):
    """Delete a knowledge base entry by ID"""
    try:
        conn = connect_to_postgres()
        c = conn.cursor()
        c.execute("DELETE FROM knowledge_base WHERE id = %s RETURNING id", (entry_id,))
        deleted = c.fetchone()
        conn.commit()
        conn.close()
        if not deleted:
            raise HTTPException(404, "Entry not found")
        return {"message": f"Entry {entry_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


class KBUpdateRequest(BaseModel):
    question: str
    answer: str


@app.put("/knowledge_base/{entry_id}")
async def update_knowledge(entry_id: int, request: KBUpdateRequest):
    """Edit a knowledge base entry"""
    try:
        # Re-embed if question changed
        try:
            q_emb = embed_chunks([request.question])
            q_vec = q_emb[0] if q_emb else None
        except Exception:
            q_vec = None
        conn = connect_to_postgres()
        c = conn.cursor()
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
        raise
    except Exception as e:
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
        docx_path = os.path.join(tmp_dir, file.filename)
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

init_db()
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)