from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import os
import datetime
from typing import List
import requests
from pydantic import BaseModel

app = FastAPI(title="SynergeReader API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow any origin
    allow_credentials=False,  # must be False if using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'synerge_reader.db')

# Pydantic models
class AskRequest(BaseModel):
    selected_text: str
    question: str

class AskResponse(BaseModel):
    answer: str
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
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        selected_text TEXT NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

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

def call_llm(question: str, context_chunks: List[str], selected_text: str, relevant_history: List[dict]) -> str:
    """Call LLM with enhanced prompt including context and history"""
    # Build comprehensive prompt
    prompt_parts = []
    
    # Add relevant history
    if relevant_history:
        history_text = "\n".join([
            f"Previous Q: {h['question']}\nPrevious A: {h['answer']}"
            for h in relevant_history
        ])
        prompt_parts.append(f"Relevant History:\n{history_text}")
    
    # Add context chunks
    if context_chunks:
        context_text = "\n\n".join(context_chunks)
        prompt_parts.append(f"Document Context:\n{context_text}")
    
    # Add selected text
    if selected_text:
        prompt_parts.append(f"Selected Text:\n{selected_text}")
    
    # Add question
    prompt_parts.append(f"Question:\n{question}")
    
    # Build final prompt
    prompt = "\n\n".join(prompt_parts) + "\n\nPlease provide a comprehensive answer based on the context provided."
    
    # Call OpenRouter API
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "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"
    model = "meta-llama/llama-3.3-70b-instruct:free"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# Initialize database on startup
init_db()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for chunking and embedding"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF, TXT, or DOCX files.")
        
        # Check file size (20MB limit)
        if file.size > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
        
        # Read file content
        content = await file.read()
        
        # For now, we'll assume text content is passed directly
        if file.filename.lower().endswith('.txt'):
            text_content = content.decode('utf-8')
        else:
            # For PDF/DOCX, we'll need to implement parsing
            # For now, assume text content
            text_content = content.decode('utf-8', errors='ignore')
        
        # Generate document ID
        document_id = f"doc_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": document_id,
            "text_length": len(text_content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question and get LLM answer with context"""
    try:
        # Analyze the question
        question_analysis = analyze_question(request.question, request.selected_text)
        
        # For demo purposes, create dummy context chunks
        context_chunks = [
            "This is a sample context chunk from the document that might be relevant to your question.",
            "Another relevant piece of information from the document that could help answer your question."
        ]
        
        # Get relevant history
        relevant_history = get_relevant_history(request.question, request.selected_text)
        
        # Call LLM with enhanced context
        answer = call_llm(
            request.question, 
            context_chunks, 
            request.selected_text, 
            relevant_history
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
            relevant_history=relevant_history
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

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "SynergeReader API is working!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=5000)
