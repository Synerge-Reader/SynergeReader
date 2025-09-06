from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import datetime
from typing import List
import requests
import json
from pydantic import BaseModel

app = FastAPI(title="SynergeReader API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.getenv('OLLAMA_HOST', '172.18.0.1')  # Use the gateway IP as default
OLLAMA_PORT = os.getenv('OLLAMA_PORT', '11434')

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'synerge_reader.db')

# Pydantic models
class AskRequest(BaseModel):
    selected_text: str
    question: str
    model: str

class AskResponse(BaseModel):
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

def init_db():
    """Initialize SQLite database with proper schema"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Existing chat_history table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        selected_text TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL
    )''')
    
    # New documents table
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        upload_timestamp TEXT NOT NULL,
        content TEXT NOT NULL
    )''')
    
    # New document_chunks table
    c.execute('''CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER,
        embedding_json TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )''')
    
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
        current_size = sum(len(w) for w in current_chunk) + len(current_chunk) - 1  # -1 for spaces
        
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

def embed_chunks(chunks: List[str], model: str = "DC1LEX/Qwen3-Embedding-0.6B-f16:latest") -> List[List[float]]:
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
            payload = {
                "model": model,
                "prompt": chunk
            }
            
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
    if any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
        analysis_parts.append("Question type: Information seeking")
    
    if any(word in question.lower() for word in ['compare', 'difference', 'similar']):
        analysis_parts.append("Question type: Comparison")
    
    if any(word in question.lower() for word in ['explain', 'describe', 'define']):
        analysis_parts.append("Question type: Explanation")
    
    # Extract key terms
    key_terms = [word for word in question.lower().split() if len(word) > 3]
    analysis_parts.append(f"Key terms: {', '.join(key_terms[:5])}")
    
    # Context analysis
    if selected_text:
        analysis_parts.append(f"Context length: {len(selected_text)} characters")
    
    return "; ".join(analysis_parts)

def get_relevant_history(question: str, selected_text: str, limit: int = 3) -> List[dict]:
    """Retrieve relevant chat history based on similarity"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, ts, selected_text, question, answer 
        FROM chat_history 
        ORDER BY id DESC 
        LIMIT 20
    ''')
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
            relevant_history.append({
                "id": id,
                "timestamp": ts,
                "selected_text": sel_text,
                "question": q,
                "answer": a,
                "relevance_score": score
            })
    
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
    
    c.execute('''
        SELECT chunk_text, embedding_json 
        FROM document_chunks
    ''')
    
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
    model: str
) -> str:
    """Call Ollama LLM and return the full answer after completion using streaming"""

    # Build prompt
    prompt_parts = []

    if relevant_history:
        history_text = "\n".join([
            f"Previous Q: {h['question']}\nPrevious A: {h['answer']}"
            for h in relevant_history
        ])
        prompt_parts.append(f"Relevant History:\n{history_text}")

    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        prompt_parts.append(f"Document Context:\n{context_text}")

    if selected_text:
        prompt_parts.append(f"Selected Text:\n{selected_text}")

    prompt_parts.append(f"Question:\n{question}")

    prompt = "\n\n".join(prompt_parts) + "\n\nPlease provide a comprehensive answer based on the context provided."

    # Ollama API call with streaming
    api_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.7
    }

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
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for embedding storage"""
    try:
        # Read file content
        content = await file.read()
        
        # Decode text (handle different file types as needed)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            # Try other encodings if UTF-8 fails
            try:
                text = content.decode("latin-1")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Could not decode file content")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty")
        
        # Chunk the text
        chunks = chunk_text(text, max_chunk_size=500)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks generated from document")
        
        # Generate embeddings
        embeddings = embed_chunks(chunks)
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Insert document
        c.execute('''
            INSERT INTO documents (filename, upload_timestamp, content) 
            VALUES (?, ?, ?)
        ''', (file.filename, datetime.datetime.now().isoformat(), text))
        
        document_id = c.lastrowid
        
        # Insert chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            c.execute('''
                INSERT INTO document_chunks (document_id, chunk_text, chunk_index, embedding_json) 
                VALUES (?, ?, ?, ?)
            ''', (document_id, chunk, i, json.dumps(embedding)))
        
        conn.commit()
        conn.close()
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "document_id": document_id,
            "chunks_count": len(chunks),
            "embeddings_count": len(embeddings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question and get LLM answer with context from uploaded documents"""
    try:
        # Analyze the question
        question_analysis = analyze_question(request.question, request.selected_text)
        
        # Get relevant chunks from uploaded documents
        context_chunks = get_relevant_chunks(request.question, top_k=3)
        
        # If no uploaded documents, fall back to selected text as context
        if not context_chunks and request.selected_text:
            context_chunks = [request.selected_text]
        elif not context_chunks:
            context_chunks = ["No relevant context found in uploaded documents."]
        
        # Get relevant history
        relevant_history = get_relevant_history(request.question, request.selected_text)
        
        # Call LLM with enhanced context
        answer = call_llm(
            request.question, 
            context_chunks, 
            request.selected_text, 
            relevant_history,
            request.model
        )
        
        # Store in SQLite history
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_history (ts, selected_text, question, answer) 
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            request.selected_text,
            request.question,
            answer
        ))
        conn.commit()
        conn.close()
        
        return AskResponse(
            answer=answer,
            context_chunks=context_chunks,
            relevant_history=relevant_history,
            question=request.question
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/history", response_model=List[HistoryItem])
async def get_history():
    """Get chat history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT id, ts, selected_text, question, answer 
            FROM chat_history 
            ORDER BY id DESC 
            LIMIT 20
        ''')
        rows = c.fetchall()
        conn.close()
        
        return [
            HistoryItem(
                id=row[0],
                timestamp=row[1],
                selected_text=row[2],
                question=row[3],
                answer=row[4]
            )
            for row in rows
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/documents")
async def get_documents():
    """Get list of uploaded documents"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT id, filename, upload_timestamp,
                   (SELECT COUNT(*) FROM document_chunks WHERE document_id = documents.id) as chunks_count
            FROM documents 
            ORDER BY upload_timestamp DESC
        ''')
        rows = c.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "filename": row[1],
                "upload_timestamp": row[2],
                "chunks_count": row[3]
            }
            for row in rows
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "SynergeReader API is working!"}

# Initialize database when app starts
init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)