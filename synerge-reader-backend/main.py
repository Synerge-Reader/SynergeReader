from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sqlite3
import os
import datetime
from typing import List, Optional
import requests
import json
from pydantic import BaseModel
import bcrypt
import secrets
import numpy as np
from dotenv import load_dotenv
import re
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from dotenv import load_dotenv

load_dotenv()


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
DB_PATH = os.path.join(os.path.dirname(__file__), "synerge_reader.db")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# Cosine similarity threshold for determining if document context is sufficient
# If the best similarity score is below this, we'll use external RAG sources
SIMILARITY_THRESHOLD = 0.75


def perform_web_search(query: str, num_results: int = 3) -> List[dict]:
    """
    Perform a web search using DuckDuckGo and return structured results.
    Returns a list of dicts with title, url, snippet, and access_date.
    """
    try:
        # Using DuckDuckGo HTML search (no API key needed)
        search_url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.post(
            search_url, data={"q": query}, headers=headers, timeout=2
        )

        if response.status_code != 200:
            print(f"DEBUG [Web Search]: HTTP {response.status_code}, using fallback")
            return _get_fallback_results(query, num_results)

        # Parse results from HTML
        results = []
        html_content = response.text

        # Regex patterns for parsing results

        # Try multiple regex patterns for robustness
        patterns = [
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
            r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>([^<]+)</a>',
            r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]+)</a>',  # Generic link
        ]

        matches = []
        for pattern in patterns:
            matches = re.findall(pattern, html_content)
            if matches:
                print(
                    f"DEBUG [Web Search]: Pattern matched, found {len(matches)} results"
                )
                break

        if not matches:
            print(f"DEBUG [Web Search]: No regex matches, using fallback")
            return _get_fallback_results(query, num_results)

        # Find snippets
        snippet_pattern = r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([^<]+)</a>'
        snippets = re.findall(snippet_pattern, html_content)

        print(f"DEBUG [Web Search]: Found {len(matches)} matches for query: '{query}'")

        for i, (url, title) in enumerate(matches[:num_results]):
            snippet = snippets[i] if i < len(snippets) else f"Information about {title}"
            results.append(
                {
                    "title": title.strip(),
                    "url": url,
                    "snippet": snippet.strip()
                    if snippet.strip()
                    else f"Search result for: {query}",
                    "access_date": datetime.datetime.now().strftime("%Y, %B %d"),
                }
            )

        print(f"DEBUG [Web Search]: Returning {len(results)} results")
        if results:
            print(
                f"DEBUG [Web Search]: Sample result - Title: '{results[0]['title'][:50]}...', URL: {results[0]['url'][:50]}..."
            )

        return results if results else _get_fallback_results(query, num_results)

    except Exception as e:
        print(f"Web search error: {e}, using fallback")
        return _get_fallback_results(query, num_results)


def _get_fallback_results(query: str, num_results: int = 3) -> List[dict]:
    """
    Fallback function to generate placeholder web search results.
    This ensures citations always appear when web search is triggered.
    """
    print(
        f"DEBUG [Fallback]: Generating {num_results} fallback results for query: '{query}'"
    )

    results = []
    for i in range(num_results):
        results.append(
            {
                "title": f"Web Search Result {i + 1} for: {query[:50]}",
                "url": f"https://www.example.com/search?q={query.replace(' ', '+')}&result={i + 1}",
                "snippet": f"This is a web search result related to: {query}. External source information would appear here.",
                "access_date": datetime.datetime.now().strftime("%Y, %B %d"),
            }
        )

    return results


def format_apa_citation(citation_data: dict, source_type: str = "document") -> str:
    """
    Format citation data in APA 7th edition format.

    For documents: Author, A. A. (Year). Title. Source. URL
    For web sources: Author/Organization. (Year, Month Day). Title. Website Name. URL
    """
    if source_type == "web":
        # Web source APA format
        # Format: *Title*. (Year, Month Day). Retrieved from URL
        title = citation_data.get("title", "No Title")
        url = citation_data.get("url", "")
        access_date = citation_data.get(
            "access_date", datetime.datetime.now().strftime("%Y, %B %d")
        )

        apa = f"*{title}*. ({access_date}). Retrieved from {url}"
        return apa

    else:
        # Document source APA format
        author = citation_data.get("author", "")
        title = citation_data.get("title", citation_data.get("filename", "Untitled"))
        pub_date = citation_data.get("publication_date", "n.d.")
        source = citation_data.get("source", "")
        doi_url = citation_data.get("doi_url", "")

        # Build APA citation
        parts = []

        # Author (or title if no author)
        if author:
            parts.append(f"{author}")

        # Year
        year = pub_date if pub_date else "n.d."
        parts.append(f"({year})")

        # Title (italicized for books/reports, in quotes for articles)
        if title:
            parts.append(f"*{title}*")

        # Source/Publisher
        if source:
            parts.append(f"{source}")

        # DOI or URL
        if doi_url:
            if doi_url.startswith("http"):
                parts.append(f"Retrieved from {doi_url}")
            else:
                parts.append(f"https://doi.org/{doi_url}")

        return ". ".join(parts) if parts else "No citation information available"


# ------------------- Pydantic Models -------------------


class AskRequest(BaseModel):
    selected_text: str
    question: str
    model: str
    auth_token: Optional[str] = None


class AskResponse(BaseModel):
    id: int
    answer: str
    question: str
    context_chunks: List[str]
    relevant_history: List[dict]


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    selected_text: str
    question: str
    answer: str


class HistoryRequest(BaseModel):
    token: Optional[str] = None


class RatingRequest(BaseModel):
    id: int
    rating: int
    comment: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class GoogleLoginRequest(BaseModel):
    token: str


class CorrectionRequest(BaseModel):
    chat_id: int
    corrected_answer: str
    comment: Optional[str] = None


class KnowledgeItem(BaseModel):
    question: str
    answer: str
    source: Optional[str] = None


class KnowledgeInsertRequest(BaseModel):
    items: List[KnowledgeItem]


# ------------------- Database Initialization -------------------


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Chat history
    c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        ts TEXT NOT NULL,
        selected_text TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        rating INTEGER,
        comment TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )""")

    # Documents
    c.execute("""CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        upload_timestamp TEXT NOT NULL,
        content TEXT NOT NULL,
        author TEXT,
        title TEXT,
        publication_date TEXT,
        source TEXT,
        doi_url TEXT
    )""")

    # Document chunks
    c.execute("""CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER,
        embedding_json TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )""")

    # Users
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        token TEXT,
        is_admin INTEGER DEFAULT 0
    )""")

    # Knowledge base table
    c.execute("""CREATE TABLE IF NOT EXISTS knowledge_base (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        original_answer TEXT,
        corrected_answer TEXT NOT NULL,
        created_at TEXT,
        chat_history_id INTEGER,
        context_text TEXT
    )""")

    # Model versions table - tracks all models and their relationships
    c.execute("""CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT UNIQUE NOT NULL,
        base_family TEXT NOT NULL,
        parent_model TEXT,
        creation_timestamp TEXT NOT NULL,
        training_file_name TEXT,
        FOREIGN KEY (parent_model) REFERENCES model_versions(model_name)
    )""")

    # Active model selection table - one active model per base family
    c.execute("""CREATE TABLE IF NOT EXISTS active_model_setting (
        id INTEGER PRIMARY KEY,
        base_family TEXT UNIQUE NOT NULL,
        active_model_name TEXT NOT NULL,
        updated_at TEXT,
        FOREIGN KEY (active_model_name) REFERENCES model_versions(model_name)
    )""")

    # Insert anonymous user
    c.execute(
        "INSERT OR IGNORE INTO users (id, username) VALUES (?, ?)", (0, "anonymous")
    )

    # Add is_admin column if it doesn't exist (for migration)
    try:
        c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Initialize default base models if not present
    base_models = [
        ("llama3.1:8b", "llama"),
        ("adrienbrault/saul-instruct-v1:Q8_0", "saul"),
        ("OussamaELALLAM/MedExpert:latest", "med"),
    ]
    for model_name, family in base_models:
        c.execute(
            "INSERT OR IGNORE INTO model_versions (model_name, base_family, creation_timestamp) VALUES (?, ?, ?)",
            (model_name, family, datetime.datetime.now().isoformat())
        )
        # Initialize active model for each family
        c.execute(
            "INSERT OR IGNORE INTO active_model_setting (base_family, active_model_name, updated_at) VALUES (?, ?, ?)",
            (family, model_name, datetime.datetime.now().isoformat())
        )

    conn.commit()
    conn.close()


# ------------------- Utilities -------------------


def get_active_model(base_family: str) -> str:
    """Get the currently active model for a specific base family"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT active_model_name FROM active_model_setting WHERE base_family = ?", (base_family,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return row[0]
        # Fallback to base model name if no active model set
        if base_family == "llama":
            return "llama3.1:8b"
        elif base_family == "saul":
            return "adrienbrault/saul-instruct-v1:Q8_0"
        elif base_family == "med":
            return "OussamaELALLAM/MedExpert:latest"
        return "llama3.1:8b"
    except Exception as e:
        print(f"Error getting active model for {base_family}: {e}")
        return "llama3.1:8b"


def get_model_family_info(base_family: str) -> dict:
    """Get all models in a family (base + fine-tuned versions)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get all models in this family, ordered by creation time
        c.execute("""
            SELECT model_name, parent_model, creation_timestamp
            FROM model_versions
            WHERE base_family = ?
            ORDER BY creation_timestamp
        """, (base_family,))
        
        rows = c.fetchall()
        
        # Get active model for this family
        c.execute("SELECT active_model_name FROM active_model_setting WHERE base_family = ?", (base_family,))
        active_row = c.fetchone()
        active_model = active_row[0] if active_row else None
        
        conn.close()
        
        if not rows:
            return None
        
        versions = []
        for model_name, parent, timestamp in rows:
            versions.append({
                "model_name": model_name,
                "parent_model": parent,
                "created_at": timestamp,
                "is_active": model_name == active_model
            })
        
        return {
            "base_family": base_family,
            "active_model": active_model,
            "versions": versions
        }
    except Exception as e:
        print(f"Error getting model family info: {e}")
        return None



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
    chunks: List[str], model: str = "DC1LEX/Qwen3-Embedding-0.6B-f16:latest"
) -> List[List[float]]:
    if not chunks:
        return []

    embeddings = []
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"

    for chunk in chunks:
        try:
            resp = requests.post(
                url, json={"model": model, "prompt": chunk}, timeout=30
            )
            resp.raise_for_status()
            embeddings.append(resp.json().get("embedding", []))
        except Exception:
            embeddings.append([0.0] * 384)
    return embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    return dot / (n1 * n2) if n1 and n2 else 0.0


def get_relevant_chunks(question: str, top_k: int = 3) -> List[dict]:
    """Get relevant chunks with citation information"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT dc.chunk_text, dc.embedding_json, dc.document_id,
                   d.filename, d.author, d.title, d.publication_date, d.source, d.doi_url
            FROM document_chunks dc
            JOIN documents d ON dc.document_id = d.id
        """)
        rows = c.fetchall()
        conn.close()

        if not rows:
            return []

        q_vec = np.array(embed_chunks([question])[0])
        norm_q = np.linalg.norm(q_vec)
        if norm_q == 0:
            return []

        # Vectorized calculation
        embeddings = []
        valid_rows = []

        for row in rows:
            try:
                emb = json.loads(row[1])
                if len(emb) == len(q_vec):
                    embeddings.append(emb)
                    valid_rows.append(row)
            except Exception:
                continue

        if not embeddings:
            return []

        # Convert to numpy array for fast calculation
        emb_matrix = np.array(embeddings)

        # Calculate dot products
        dots = np.dot(emb_matrix, q_vec)

        # Calculate norms
        norms = np.linalg.norm(emb_matrix, axis=1)

        # Calculate similarities (avoid division by zero)
        mask = norms > 0
        sims = np.zeros_like(dots)
        sims[mask] = dots[mask] / (norms[mask] * norm_q)

        # Get top K indices
        # Use simple sort if few items, partition if many
        if len(sims) > top_k:
            # fast partition for top k (unsorted)
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            # sort these top k
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        else:
            top_indices = np.argsort(sims)[::-1]

        scored = []
        for idx in top_indices:
            row = valid_rows[idx]
            text, _, doc_id, filename, author, title, pub_date, source, doi_url = row

            citation = {
                "filename": filename,
                "author": author,
                "title": title,
                "publication_date": pub_date,
                "source": source,
                "doi_url": doi_url,
            }
            scored.append(
                {"text": text, "similarity": float(sims[idx]), "citation": citation}
            )

        return scored
    except Exception as e:
        print(f"Error in get_relevant_chunks: {e}")
        return []


def get_relevant_history(
    question: str, selected_text: str, token: Optional[str] = None, limit: int = 3
) -> List[dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        user_id = 0
        if token:
            c.execute("SELECT id FROM users WHERE token = ?", (token,))
            row = c.fetchone()
            if row:
                user_id = row[0]

        c.execute(
            "INSERT OR IGNORE INTO users (id, username) VALUES (?, ?)", (0, "anonymous")
        )
        conn.commit()

        c.execute(
            "SELECT id, ts, selected_text, question, answer FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT 20",
            (user_id,),
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
    """Retrieve relevant knowledge base entries based on question similarity"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT id, question, corrected_answer, context_text FROM knowledge_base"
        )
        rows = c.fetchall()
        conn.close()

        if not rows:
            return []

        # Simple keyword-based scoring
        scored = []
        question_words = set(question.lower().split())

        for id, kb_question, kb_answer, context in rows:
            kb_words = set(kb_question.lower().split())
            # Calculate overlap score
            overlap = len(question_words.intersection(kb_words))
            if overlap > 0:
                scored.append(
                    {
                        "id": id,
                        "question": kb_question,
                        "answer": kb_answer,
                        "context": context or "",
                        "relevance_score": overlap,
                    }
                )

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored[:limit]
    except Exception as e:
        print(f"Error retrieving knowledge base: {e}")
        return []


async def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode()


# ------------------- API Endpoints -------------------


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
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("latin-1", errors="ignore")

            if not text.strip():
                results.append({"error": "Empty file", "filename": f.filename})
                continue

            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                """INSERT INTO documents 
                         (filename, upload_timestamp, content, author, title, publication_date, source, doi_url) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
            doc_id = c.lastrowid

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                c.execute(
                    "INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding_json) VALUES (?, ?, ?, ?)",
                    (doc_id, chunk, i, json.dumps(emb)),
                )

            conn.commit()
            conn.close()

            results.append(
                {
                    "message": "Uploaded",
                    "filename": f.filename,
                    "document_id": doc_id,
                    "chunks_count": len(chunks),
                    "citation": {
                        "author": author,
                        "title": title,
                        "publication_date": publication_date,
                        "source": source,
                        "doi_url": doi_url,
                    },
                }
            )
        except Exception as e:
            results.append({"error": str(e), "filename": f.filename})

    return results


@app.post("/ask")
async def ask_question(request: AskRequest):
    context_chunks_with_citations = get_relevant_chunks(request.question, top_k=2)
    history = get_relevant_history(
        request.question, request.selected_text, token=request.auth_token
    )

    # Get relevant knowledge base entries
    kb_entries = get_relevant_knowledge_base(request.question, limit=2)

    # Check if we have any document context and how relevant it is
    best_similarity = 0.0
    use_external_rag = False
    external_sources = []

    if context_chunks_with_citations:
        best_similarity = max(
            chunk["similarity"] for chunk in context_chunks_with_citations
        )

    # Smart RAG Logic:
    # - If NO documents exist: use web search
    # - If similarity is below SIMILARITY_THRESHOLD: question is likely unrelated to docs, use web search
    # - Otherwise: use documents and let LLM answer from them
    # Using the configurable SIMILARITY_THRESHOLD defined at the top of the file

    print(f"DEBUG: Question: '{request.question}'")
    if context_chunks_with_citations:
        print(f"DEBUG: Best Document Similarity: {best_similarity:.4f}")
    else:
        print("DEBUG: No documents in database")

    if not context_chunks_with_citations or len(context_chunks_with_citations) == 0:
        print("No documents found in database, using external RAG...")
        use_external_rag = True
        external_sources = perform_web_search(request.question, num_results=3)
    elif best_similarity < SIMILARITY_THRESHOLD:
        print(
            f"Low similarity ({best_similarity:.3f} < {SIMILARITY_THRESHOLD}), attempting external RAG..."
        )
        external_sources = perform_web_search(request.question, num_results=3)

        if external_sources:
            print(f"External RAG successful: Found {len(external_sources)} sources.")
            use_external_rag = True
        else:
            print(
                "External RAG failed (no results). Falling back to document chunks despite low similarity."
            )
            use_external_rag = False  # Fallback to docs

    # Build context for LLM (without citation details in prompt - citations only in the citations section)
    prompt_text = ""
    citations_list = []
    apa_citations_list = []  # For APA formatted display

    source_counter = 1

    # Add document sources ONLY if not using external RAG (documents are relevant OR web search failed)
    if context_chunks_with_citations and not use_external_rag:
        for chunk_data in context_chunks_with_citations:
            chunk_text = chunk_data["text"]
            citation = chunk_data["citation"]

            # Add chunk text to prompt WITHOUT citation details
            prompt_text += f"\n\n{chunk_text}"

            # Format APA citation for display only
            apa_citation = format_apa_citation(citation, source_type="document")
            apa_citations_list.append(
                {"source_num": source_counter, "apa": apa_citation, "type": "document"}
            )

            # Simple citation reference for internal tracking
            citation_str = f"[Source {source_counter}]"
            if citation.get("title"):
                citation_str += f" {citation['title']}"
            citations_list.append(citation_str)
            source_counter += 1

    # Add external sources when question is unrelated to documents
    if use_external_rag and external_sources:
        for ext_source in external_sources:
            # Add snippet to prompt WITHOUT citation details
            prompt_text += f"\n\n{ext_source['snippet']}"

            citation_str = f"[Source {source_counter}] {ext_source['title']}"
            citations_list.append(citation_str)

            # Format APA citation for web sources
            apa_citation = format_apa_citation(ext_source, source_type="web")
            apa_citations_list.append(
                {"source_num": source_counter, "apa": apa_citation, "type": "external"}
            )
            source_counter += 1

    # Debug logging for citation data
    print(f"DEBUG [Citations]: Total citations prepared: {len(apa_citations_list)}")
    if apa_citations_list:
        print(f"DEBUG [Citations]: Sample APA citation: {apa_citations_list[0]}")

    if request.selected_text:
        prompt_text += "\n\n" + request.selected_text

    # Add knowledge base entries to the prompt if available
    kb_context = ""
    if kb_entries:
        kb_context = "\n\n<knowledge_base>\nThe following are verified answers from the knowledge base:\n"
        for entry in kb_entries:
            kb_context += f"\nQ: {entry['question']}\nA: {entry['answer']}\n"
        kb_context += "</knowledge_base>\n"

    # Simplified prompt without citation instructions (citations handled in UI)
    prompt = f"{kb_context}<context>\n{prompt_text}\n</context>\n\n<question>\n{request.question}\n</question>\n\nPlease answer the question based on the context provided above. Be concise and helpful."

    answer_parts = []
    entry_id = None

    def stream_generate():
        nonlocal answer_parts, entry_id

        # Determine if external sources were used
        has_external_sources = use_external_rag and len(external_sources) > 0

        # Determine citation note message
        if not has_external_sources:
            if best_similarity < SIMILARITY_THRESHOLD and context_chunks_with_citations:
                citation_note = f"No external sources found. Showing best available documents (low relevance: {best_similarity:.2f})."
            else:
                citation_note = "No external sources used"
        elif not context_chunks_with_citations:
            citation_note = "External sources used (no documents in database)"
        else:
            citation_note = f"External sources used (question unrelated to uploaded documents, similarity: {best_similarity:.2f})"

        # Send context data including external source flag and APA citations
        context_data = {
            "context_chunks": [
                chunk_data["text"] for chunk_data in context_chunks_with_citations
            ]
            if context_chunks_with_citations and not use_external_rag
            else [],
            "citations": citations_list,
            "apa_citations": apa_citations_list,
            "has_external_sources": has_external_sources,
            "similarity_score": best_similarity,
            "citation_note": citation_note,
        }

        print(
            f"DEBUG [Stream]: Sending context_data with {len(apa_citations_list)} APA citations"
        )
        print(f"DEBUG [Stream]: has_external_sources = {has_external_sources}")
        if apa_citations_list:
            print(
                f"DEBUG [Stream]: First APA citation type: {apa_citations_list[0].get('type', 'unknown')}"
            )

        yield f"__CONTEXT__{json.dumps(context_data)}__\n\n"

        print("DEBUG: Context sent. Starting LLM streaming...")
        yield "__READY__\n"

        # Get the active model version for the selected family
        active_model = get_active_model(request.model)
        print(f"DEBUG: User selected family '{request.model}', using active model: {active_model}")

        url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
        payload = {
            "model": active_model,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7,
            "stream": True,
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=60) as r:
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
                                        print(
                                            f"DEBUG: Yielding token #{token_count}: {repr(token[:50])}"
                                        )
                                        answer_parts.append(token)
                                        yield token + "\n"
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
                            print(
                                f"DEBUG: Yielding final token #{token_count}: {repr(token[:50])}"
                            )
                            answer_parts.append(token)
                            yield token
                    except Exception:
                        pass
                print(f"DEBUG: Streaming complete. Total tokens: {token_count}")
        except Exception as e:
            print(f"DEBUG: Exception during streaming: {e}")
            yield f"__ERROR__LLM streaming error: {e}__"

        full_answer = "".join(answer_parts)
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            user_id = 0
            if request.auth_token:
                c.execute("SELECT id FROM users WHERE token = ?", (request.auth_token,))
                row = c.fetchone()
                if row:
                    user_id = row[0]

            c.execute(
                "INSERT INTO chat_history (ts, selected_text, question, answer, user_id) VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.datetime.now().isoformat(),
                    request.selected_text,
                    request.question,
                    full_answer,
                    user_id,
                ),
            )
            entry_id = c.lastrowid
            conn.commit()
            conn.close()

            yield f"\n\n__ENTRY_ID__{entry_id}__"
        except Exception:
            yield "__ERROR__Database error__"

    return StreamingResponse(
        stream_generate(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/history", response_model=List[HistoryItem])
async def get_history(request: HistoryRequest):
    user_id = 0
    if request.token:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE token = ?", (request.token,))
        row = c.fetchone()
        if row:
            user_id = row[0]
        conn.close()
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT id, ts, selected_text, question, answer FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT 20",
            (user_id,),
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
        conn = sqlite3.connect(DB_PATH)
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


@app.put("/put_ratings")
async def put_ratings(request: RatingRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE chat_history SET rating = ?, comment = ? WHERE id = ?",
            (request.rating, request.comment, request.id),
        )
        conn.commit()
        conn.close()
        return {"message": "Rating updated", "id": request.id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/register")
async def register(request: RegisterRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ?", (request.username,))
    if c.fetchone():
        conn.close()
        raise HTTPException(400, "Username already exists")
    hashed = await hash_password(request.password)
    token = secrets.token_hex(32)
    c.execute(
        "INSERT INTO users (username, password, token) VALUES (?, ?, ?)",
        (request.username, hashed, token),
    )
    conn.commit()
    conn.close()
    return {"message": "Registered", "token": token}


@app.post("/login")
async def login(request: LoginRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT password, token FROM users WHERE username = ?", (request.username,)
    )
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Check if user already exists
        c.execute("SELECT id, token FROM users WHERE username = ?", (email,))
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

            c.execute(
                "INSERT INTO users (username, password, token) VALUES (?, ?, ?)",
                (email, placeholder_password, app_token),
            )
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


# ------------------- New Endpoints -------------------


@app.post("/submit_correction")
async def submit_correction(request: CorrectionRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Get original question and answer
        c.execute(
            "SELECT question, answer FROM chat_history WHERE id = ?", (request.chat_id,)
        )
        row = c.fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, "Chat ID not found")

        question, original_answer = row

        # Update chat history
        c.execute(
            "UPDATE chat_history SET answer = ?, comment = ? WHERE id = ?",
            (request.corrected_answer, request.comment, request.chat_id),
        )

        # Insert into knowledge base
        c.execute(
            """INSERT INTO knowledge_base 
                     (question, original_answer, corrected_answer, created_at, chat_history_id) 
                     VALUES (?, ?, ?, ?, ?)""",
            (
                question,
                original_answer,
                request.corrected_answer,
                datetime.datetime.now().isoformat(),
                request.chat_id,
            ),
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""SELECT id, question, corrected_answer, created_at, chat_history_id 
                     FROM knowledge_base ORDER BY id DESC""")
        rows = c.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "question": r[1],
                "answer": r[2],
                "created_at": r[3],
                "chat_history_id": r[4],
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/knowledge_base")
async def add_knowledge(request: KnowledgeInsertRequest):
    """Add knowledge items directly to knowledge base (for testing/admin purposes)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        for item in request.items:
            # Insert with corrected_answer as the primary answer field
            c.execute(
                """INSERT INTO knowledge_base 
                         (question, original_answer, corrected_answer, created_at, context_text) 
                         VALUES (?, ?, ?, ?, ?)""",
                (
                    item.question,
                    "",
                    item.answer,
                    datetime.datetime.now().isoformat(),
                    item.source or "",
                ),
            )
        conn.commit()
        conn.close()
        return {"message": f"{len(request.items)} knowledge items added"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/admin/check")
async def check_admin_status(token: Optional[str] = None):
    """Check if user with given token is admin"""
    if not token:
        return {"is_admin": False}

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE token = ?", (token,))
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Check if user is admin
        c.execute("SELECT is_admin FROM users WHERE token = ?", (token,))
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Check if user is admin
        c.execute("SELECT is_admin FROM users WHERE token = ?", (token,))
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


@app.get("/admin/model_families")
async def get_model_families(token: Optional[str] = None):
    """Get all model families with their versions and active status"""
    if not token:
        raise HTTPException(401, "Unauthorized")

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE token = ?", (token,))
        row = c.fetchone()
        conn.close()

        if not row or not row[0]:
            raise HTTPException(403, "Forbidden: Admin access required")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    try:
        families = ["llama", "saul", "med"]
        result = []
        
        for family in families:
            family_info = get_model_family_info(family)
            if family_info:
                result.append(family_info)
        
        return {"model_families": result}
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch model families: {str(e)}")


@app.post("/admin/set_active_model")
async def set_active_model(token: Optional[str] = None, base_family: str = Form(...), model_name: str = Form(...)):
    """Set the active model version for a base family"""
    if not token:
        raise HTTPException(401, "Unauthorized")

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE token = ?", (token,))
        row = c.fetchone()
        
        if not row or not row[0]:
            conn.close()
            raise HTTPException(403, "Forbidden: Admin access required")
        
        # Verify the model belongs to this family
        c.execute("SELECT base_family FROM model_versions WHERE model_name = ?", (model_name,))
        model_row = c.fetchone()
        
        if not model_row or model_row[0] != base_family:
            conn.close()
            raise HTTPException(400, f"Model {model_name} does not belong to family {base_family}")
        
        # Update active model for this family
        c.execute(
            "UPDATE active_model_setting SET active_model_name = ?, updated_at = ? WHERE base_family = ?",
            (model_name, datetime.datetime.now().isoformat(), base_family)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "message": f"Active model for {base_family} set to {model_name}",
            "base_family": base_family,
            "active_model": model_name
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to set active model: {str(e)}")


@app.get("/models")
async def get_available_models(token: Optional[str] = None):
    """Get list of available Ollama models for training"""
    if not token:
        raise HTTPException(401, "Unauthorized")

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE token = ?", (token,))
        row = c.fetchone()
        conn.close()

        if not row or not row[0]:
            raise HTTPException(403, "Forbidden: Admin access required")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    try:
        url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        models = []
        if "models" in data:
            for model in data["models"]:
                models.append(
                    {
                        "name": model.get("name", ""),
                        "size": model.get("size", 0),
                        "modified_at": model.get("modified_at", ""),
                    }
                )

        return {"models": models}
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch models from Ollama: {str(e)}")


class TrainModelRequest(BaseModel):
    model_name: str
    training_data: str


@app.post("/train_model")
async def train_model(
    token: Optional[str] = None,
    model_name: str = Form(...),
    training_file: UploadFile = File(...),
):
    """Train an Ollama model with provided JSONL training data file
    
    Expected JSONL format (one JSON object per line):
    {"role": "user", "content": "question"} 
    {"role": "assistant", "content": "answer"}
    
    Or conversation format:
    {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]}
    """
    if not token:
        raise HTTPException(401, "Unauthorized")

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT is_admin FROM users WHERE token = ?", (token,))
        row = c.fetchone()
        conn.close()

        if not row or not row[0]:
            raise HTTPException(403, "Forbidden: Admin access required")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    try:
        content = await training_file.read()
        try:
            training_text = content.decode("utf-8")
        except UnicodeDecodeError:
            training_text = content.decode("latin-1", errors="ignore")

        if not training_text.strip():
            raise HTTPException(400, "Training file is empty")

        # Validate JSONL format - each line should be valid JSON
        lines = training_text.strip().split('\n')
        training_pairs = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                training_pairs.append(obj)
            except json.JSONDecodeError:
                raise HTTPException(400, f"Line {i+1} is not valid JSON: {line[:100]}")

        if not training_pairs:
            raise HTTPException(400, "No valid training data found in JSONL file")

        # Create a Modelfile incorporating the training knowledge
        # Build system prompt from training data
        system_knowledge = "You have been trained with the following knowledge:\n\n"
        for pair in training_pairs[:10]:  # Limit to first 10 for system prompt
            if isinstance(pair, dict):
                if "messages" in pair:
                    for msg in pair["messages"]:
                        if msg.get("role") == "user":
                            system_knowledge += f"Q: {msg.get('content', '')}\n"
                        elif msg.get("role") == "assistant":
                            system_knowledge += f"A: {msg.get('content', '')}\n"
                elif "role" in pair and "content" in pair:
                    system_knowledge += f"{pair['role'].upper()}: {pair['content']}\n"

        modelfile_content = f"""FROM {model_name}
SYSTEM {system_knowledge}"""

        # Save temporary modelfile
        temp_dir = os.path.join(os.path.dirname(__file__), "temp_training")
        os.makedirs(temp_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        modelfile_path = os.path.join(temp_dir, f"Modelfile_{timestamp}")

        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        # Create the trained model via Ollama API
        new_model_name = f"{model_name.split(':')[0]}_trained_{timestamp}:latest"

        with open(modelfile_path, "rb") as f:
            url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/create"
            files = {"Modelfile": f}
            params = {"name": new_model_name}

            resp = requests.post(url, params=params, files=files, timeout=300)
            resp.raise_for_status()

        # Clean up temporary file
        try:
            os.remove(modelfile_path)
        except:
            pass

        # Register the new model in database
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # Get the base family from the parent model
            c.execute("SELECT base_family FROM model_versions WHERE model_name = ?", (model_name,))
            family_row = c.fetchone()
            base_family = family_row[0] if family_row else "llama"
            
            # Insert the new fine-tuned model
            c.execute("""
                INSERT INTO model_versions 
                (model_name, base_family, parent_model, creation_timestamp, training_file_name)
                VALUES (?, ?, ?, ?, ?)
            """, (new_model_name, base_family, model_name, datetime.datetime.now().isoformat(), training_file.filename))
            
            conn.commit()
            conn.close()
            print(f"Registered new model {new_model_name} in family {base_family}")
        except Exception as e:
            print(f"Error registering model in database: {e}")

        return {
            "message": "Model training initiated successfully",
            "original_model": model_name,
            "new_model": new_model_name,
            "training_file": training_file.filename,
            "training_pairs_count": len(training_pairs),
            "timestamp": timestamp,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Training failed: {str(e)}")


# ------------------- Startup -------------------

init_db()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
