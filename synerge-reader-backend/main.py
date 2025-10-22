from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import sqlite3
import os
import datetime
from typing import List
import requests
import json
from pydantic import BaseModel
import bcrypt
import secrets

# System instruction and few-shot prompt for LLM
# Removed system instruction and few-shot prompt literals to avoid embedding model-level instructions in prompts

app = FastAPI(title="SynergeReader API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "172.18.0.1")  # Use the gateway IP as default
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), "synerge_reader.db")


# Pydantic models
class AskRequest(BaseModel):
    selected_text: str
    question: str
    model: str
    auth_token: str


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
    token: str


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


def init_db():
    """Initialize SQLite database with proper schema"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Existing chat_history table
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

    # New documents table
    c.execute("""CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        upload_timestamp TEXT NOT NULL,
        content TEXT NOT NULL
    )""")

    # New document_chunks table
    c.execute("""CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER,
        embedding_json TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )""")

    # New users table
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT,
        token TEXT
    )""")

    conn.commit()
    conn.close()


def chunk_text(text: str, max_chunk_size: int = 500) -> list:
    """
    Split text into chunks based on word count and character limit.

    Args:
        text: The input text to chunk
        max_chunk_size: Maximum character size per chunk

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        # Calculate current chunk size if we add this word
        current_size = (
            sum(len(w) for w in current_chunk) + len(current_chunk) - 1
        )  # -1 for spaces

        if current_size + len(word) + 1 <= max_chunk_size:  # +1 for space
            current_chunk.append(word)
        else:
            if current_chunk:  # Only append if there's content
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def embed_chunks(
    chunks: List[str], model: str = "DC1LEX/Qwen3-Embedding-0.6B-f16:latest"
) -> List[List[float]]:
    """
    Generate embeddings for text chunks using Ollama API.

    Args:
        chunks: List of text chunks to embed
        model: Ollama model to use for embeddings

    Returns:
        List of embedding vectors
    """
    if not chunks:
        return []

    embeddings = []
    ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"

    for chunk in chunks:
        try:
            payload = {"model": model, "prompt": chunk}

            response = requests.post(ollama_url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            # Ollama returns embeddings in 'embedding' field
            embedding = data.get("embedding", [])
            embeddings.append(embedding)

        except Exception as e:
            print(f"Error generating embedding for chunk: {str(e)}")
            # Return zero vector as fallback
            embeddings.append([0.0] * 384)  # Default embedding size

    return embeddings


def analyze_question(question: str, selected_text: str) -> str:
    """Simple question analysis"""
    analysis_parts = []

    # Check question type
    if any(
        word in question.lower()
        for word in ["what", "how", "why", "when", "where", "who"]
    ):
        analysis_parts.append("Question type: Information seeking")

    if any(word in question.lower() for word in ["compare", "difference", "similar"]):
        analysis_parts.append("Question type: Comparison")

    if any(word in question.lower() for word in ["explain", "describe", "define"]):
        analysis_parts.append("Question type: Explanation")

    # Extract key terms
    key_terms = [word for word in question.lower().split() if len(word) > 3]
    analysis_parts.append(f"Key terms: {', '.join(key_terms[:5])}")

    # Context analysis
    if selected_text:
        analysis_parts.append(f"Context length: {len(selected_text)} characters")

    return "; ".join(analysis_parts)


def get_relevant_history(
    question: str, selected_text: str, limit: int = 3
) -> List[dict]:
    """Retrieve relevant chat history based on similarity"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, ts, selected_text, question, answer 
        FROM chat_history 
        ORDER BY id DESC 
        LIMIT 20
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        return []

    # Simple keyword-based relevance scoring
    question_lower = question.lower()
    selected_lower = selected_text.lower()

    relevant_history = []
    for row in rows:
        id, ts, sel_text, q, a = row
        score = 0

        # Score based on question similarity
        if any(word in q.lower() for word in question_lower.split()):
            score += 1

        # Score based on selected text similarity
        if any(word in sel_text.lower() for word in selected_lower.split()):
            score += 2

        if score > 0:
            relevant_history.append(
                {
                    "id": id,
                    "timestamp": ts,
                    "selected_text": sel_text,
                    "question": q,
                    "answer": a,
                    "relevance_score": score,
                }
            )

    # Sort by relevance and return top results
    relevant_history.sort(key=lambda x: x["relevance_score"], reverse=True)
    return relevant_history[:limit]


def get_relevant_chunks(question: str, top_k: int = 3) -> List[str]:
    """
    Get relevant document chunks based on question similarity.
    This is a simple implementation - you might want to use vector similarity later.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Simple keyword matching for now
    question_words = set(question.lower().split())

    c.execute("""
        SELECT chunk_text, embedding_json 
        FROM document_chunks
    """)

    chunks_data = c.fetchall()
    conn.close()

    if not chunks_data:
        return []

    # Score chunks based on keyword overlap
    scored_chunks = []
    for chunk_text, embedding_json in chunks_data:
        chunk_words = set(chunk_text.lower().split())
        overlap_score = len(question_words.intersection(chunk_words))
        if overlap_score > 0:
            scored_chunks.append((chunk_text, overlap_score))

    # Sort by score and return top results
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk[0] for chunk in scored_chunks[:top_k]]


def call_llm(
    question: str,
    context_chunks: List[str],
    selected_text: str,
    relevant_history: List[dict],
    model: str,
) -> str:
    """Call Ollama LLM and return the full answer after completion using streaming"""

    # Build prompt
    prompt_parts = []

    if relevant_history:
        history_text = "\n".join(
            [
                f"Previous Q: {h['question']}\nPrevious A: {h['answer']}"
                for h in relevant_history
            ]
        )
        prompt_parts.append(f"Relevant History:\n{history_text}")

    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        prompt_parts.append(f"Document Context:\n{context_text}")

    if selected_text:
        prompt_parts.append(f"Selected Text:\n{selected_text}")

    prompt_parts.append(f"Question:\n{question}")

    prompt = (
        "\n\n".join(prompt_parts)
        + "\n\nPlease provide a comprehensive answer based on the context provided."
    )

    # Ollama API call with streaming
    api_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    payload = {"model": model, "prompt": prompt, "max_tokens": 1000, "temperature": 0.7}

    try:
        response = requests.post(api_url, json=payload, timeout=60)
        response.raise_for_status()

        # Process streaming response
        full_answer = ""
        for line in response.text.splitlines():
            if line.strip():
                try:
                    parsed = json.loads(line)
                    full_answer += parsed.get("response", "")
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

        return full_answer if full_answer else "No response received from LLM"

    except Exception as e:
        return f"Error calling Ollama LLM: {str(e)}"


# API Endpoints


@app.post("/upload")
async def upload_documents(
    file: UploadFile = File(None), files: List[UploadFile] = File(None)
):
    """Upload and process one or multiple documents for embedding storage"""
    # Handle single file or multiple files
    if file and files:
        # If both provided, combine
        all_files = [file] + files
    elif files:
        all_files = files
    elif file:
        all_files = [file]
    else:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []
    for f in all_files:
        try:
            # Read file content
            content = await f.read()

            # Decode text (handle different file types as needed)
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                # Try other encodings if UTF-8 fails
                try:
                    text = content.decode("latin-1")
                except UnicodeDecodeError:
                    results.append(
                        {
                            "error": f"Could not decode file content for {f.filename}",
                            "filename": f.filename,
                        }
                    )
                    continue

            if not text.strip():
                results.append(
                    {
                        "error": f"File {f.filename} appears to be empty",
                        "filename": f.filename,
                    }
                )
                continue

            # Chunk the text
            chunks = chunk_text(text, max_chunk_size=500)

            if not chunks:
                results.append(
                    {
                        "error": f"No valid chunks generated from document {f.filename}",
                        "filename": f.filename,
                    }
                )
                continue

            # Generate embeddings
            embeddings = embed_chunks(chunks)

            # Store in database
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            # Insert document
            c.execute(
                """
                INSERT INTO documents (filename, upload_timestamp, content) 
                VALUES (?, ?, ?)
            """,
                (f.filename, datetime.datetime.now().isoformat(), text),
            )

            document_id = c.lastrowid

            # Insert chunks with embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                c.execute(
                    """
                    INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding_json) 
                    VALUES (?, ?, ?, ?)
                """,
                    (document_id, chunk, i, json.dumps(embedding)),
                )

            conn.commit()
            conn.close()

            results.append(
                {
                    "message": f"Document {f.filename} uploaded and processed successfully",
                    "filename": f.filename,
                    "document_id": document_id,
                    "chunks_count": len(chunks),
                    "embeddings_count": len(embeddings),
                }
            )

        except Exception as e:
            results.append(
                {
                    "error": f"Error processing document {f.filename}: {str(e)}",
                    "filename": f.filename,
                }
            )

    return results


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Ask a question and stream LLM answer with context from uploaded documents"""
    try:
        # Get relevant chunks from uploaded documents
        context_chunks = get_relevant_chunks(request.question, top_k=3)
        if not context_chunks and request.selected_text:
            context_chunks = [request.selected_text]
        elif not context_chunks:
            context_chunks = ["No relevant context found in uploaded documents."]

        # Get relevant history
        relevant_history = get_relevant_history(request.question, request.selected_text)

        # Build prompt
        combined_text = ""
        if context_chunks:
            combined_text += "\n\n".join(context_chunks) + "\n\n"
        if request.selected_text:
            combined_text += request.selected_text
        combined_text = combined_text.strip()
        if not combined_text:
            combined_text = "No context provided."

        # Build a neutral prompt containing only the provided context and question.
        # System-level instructions and few-shot examples have been removed to avoid
        # embedding model instructions in user-visible prompts.
        prompt = f"<text_snippet>\n{combined_text}\n</text_snippet>\n<question>\n{request.question}\n</question>"

        # Store answer chunks and entry ID for database insertion
        answer_chunks = []
        entry_id = None

        # Generator to stream from Ollama
        def stream_generate():
            nonlocal answer_chunks, entry_id
            api_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
            payload = {
                "model": request.model,
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": True,
            }

            # Stream tokens from Ollama
            try:
                with requests.post(api_url, json=payload, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            try:
                                parsed = json.loads(line.decode("utf-8"))
                                token = parsed.get("response", "")
                                if token:
                                    answer_chunks.append(token)
                                    yield token
                            except Exception:
                                continue
            except Exception as api_error:
                yield f"\n\n__ERROR__LLM streaming error: {api_error}__"

            # Save full answer to DB after streaming
            full_answer = "".join(answer_chunks)
            if full_answer:
                try:
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()

                    # Get user_id from token
                    c.execute(
                        "SELECT id FROM users WHERE token = ?", (request.auth_token,)
                    )
                    row = c.fetchone()
                    if not row:
                        return
                    user_id = row[0]

                    # Insert into chat_history
                    c.execute(
                        """
                        INSERT INTO chat_history (ts, selected_text, question, answer, user_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
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

                    # Send the entry ID at the end
                    yield f"\n\n__ENTRY_ID__{entry_id}__"

                except Exception as db_error:
                    print(f"Database error: {db_error}")
                    yield f"\n\n__ERROR__Database error occurred__"

        return StreamingResponse(stream_generate(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming answer: {str(e)}")


@app.post("/history", response_model=List[HistoryItem])
async def get_history(request: HistoryRequest):
    """Get chat history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            SELECT id, ts, selected_text, question, answer 
            FROM chat_history
            WHERE user_id IN (
                SELECT id
                FROM users
                WHERE token = ?
            )
            ORDER BY id 
            DESC LIMIT 20
        """,
            (request.token,),
        )
        rows = c.fetchall()
        conn.close()

        return [
            HistoryItem(
                id=row[0],
                timestamp=row[1],
                selected_text=row[2],
                question=row[3],
                answer=row[4],
            )
            for row in rows
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving history: {str(e)}"
        )


@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, filename, upload_timestamp,
                   (SELECT COUNT(*) FROM document_chunks WHERE document_id = documents.id) as chunks_count
            FROM documents 
            ORDER BY upload_timestamp DESC
        """)
        rows = c.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "filename": row[1],
                "upload_timestamp": row[2],
                "chunks_count": row[3],
            }
            for row in rows
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )


@app.put("/put_ratings")  # Changed to PUT since you're updating
async def put_ratings(request: RatingRequest):
    """Update rating and comment for a chat history entry"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Update the rating and comment for the specified ID
        c.execute(
            """
            UPDATE chat_history 
            SET rating = ?, comment = ?
            WHERE id = ?
        """,
            (request.rating, request.comment, request.id),
        )

        # Check if any row was actually updated
        if c.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Chat history entry not found")

        conn.commit()
        conn.close()

        return {
            "success": True,
            "message": "Rating updated successfully",
            "id": request.id,
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating rating: {str(e)}")


async def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed_password.decode("utf-8")


@app.post("/register")
async def register(request: RegisterRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        select id
        from users
        where username = ?
    """,
        (request.username,),
    )
    already_registered = c.fetchone()

    if already_registered:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = await hash_password(request.password)
    session_token = secrets.token_hex(32)
    c.execute(
        """
        INSERT INTO users (username, password, token) 
        VALUES (?, ?, ?)
    """,
        (request.username, hashed_password, session_token),
    )
    conn.commit()
    conn.close()

    return {"message": "Registered Successfully", "token": session_token}


@app.post("/login")
async def login(request: LoginRequest):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "SELECT password, token FROM users WHERE username = ?", (request.username,)
    )
    row = c.fetchone()
    if not row:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    stored_hashed_password, token = row
    if not bcrypt.checkpw(request.password.encode(), stored_hashed_password.encode()):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    return {"message": "Login Successful", "token": token}


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "SynergeReader API is working!"}


# Initialize database when app starts
init_db()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
