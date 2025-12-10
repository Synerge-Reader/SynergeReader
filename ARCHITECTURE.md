# System Architecture: Citations & Knowledge Base

## Overall System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ FileUpload   │  │ TextPreview  │  │  CorrectionModal     │  │
│  │ Component    │  │ Component    │  │  Component           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND API (FastAPI)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   /upload    │  │    /ask      │  │ /submit_correction   │  │
│  │   endpoint   │  │   endpoint   │  │     endpoint         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATABASE (SQLite)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  documents   │  │ chat_history │  │  knowledge_base      │  │
│  │  (citations) │  │              │  │  (corrections)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Citations Flow

```
┌──────────────┐
│    USER      │
│ Uploads Doc  │
│ + Citation   │
│   Metadata   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  FileUpload Component                    │
│  ┌────────────────────────────────────┐  │
│  │ Citation Fields:                   │  │
│  │ • Title                            │  │
│  │ • Author                           │  │
│  │ • Publication Date                 │  │
│  │ • Source                           │  │
│  │ • DOI/URL                          │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │ FormData
               ▼
┌──────────────────────────────────────────┐
│  Backend: /upload Endpoint               │
│  ┌────────────────────────────────────┐  │
│  │ 1. Extract file content            │  │
│  │ 2. Extract citation metadata       │  │
│  │ 3. Chunk text                      │  │
│  │ 4. Generate embeddings             │  │
│  │ 5. Store in database               │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Database: documents Table               │
│  ┌────────────────────────────────────┐  │
│  │ id, filename, content              │  │
│  │ author, title, publication_date    │  │
│  │ source, doi_url                    │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  TextPreview Component                   │
│  ┌────────────────────────────────────┐  │
│  │ Displays:                          │  │
│  │ "Title" by Author (Date)           │  │
│  │ - Source [DOI/URL]                 │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

---

## Feature 2: Answer Correction & Knowledge Base Flow

```
┌──────────────┐
│    USER      │
│  Asks Q      │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  Backend: /ask Endpoint                  │
│  ┌────────────────────────────────────┐  │
│  │ 1. Get relevant document chunks    │  │
│  │ 2. Get relevant history            │  │
│  │ 3. ★ Get knowledge base entries ★  │  │
│  │ 4. Build prompt with KB context    │  │
│  │ 5. Call LLM                        │  │
│  │ 6. Stream response                 │  │
│  │ 7. Save to chat_history            │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  LLM Prompt Structure                    │
│  ┌────────────────────────────────────┐  │
│  │ <knowledge_base>                   │  │
│  │   Q: Similar question              │  │
│  │   A: Verified answer               │  │
│  │ </knowledge_base>                  │  │
│  │                                    │  │
│  │ <text_snippet>                     │  │
│  │   Document context                 │  │
│  │ </text_snippet>                    │  │
│  │                                    │  │
│  │ <question>                         │  │
│  │   User's question                  │  │
│  │ </question>                        │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  GridApp Component                       │
│  ┌────────────────────────────────────┐  │
│  │ Displays Answer                    │  │
│  │ [Provide Feedback Button]          │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │ User clicks button
               ▼
┌──────────────────────────────────────────┐
│  CorrectionModal                         │
│  ┌────────────────────────────────────┐  │
│  │ Shows: Q & A                       │  │
│  │                                    │  │
│  │ Options:                           │  │
│  │ [✓ Mark as Correct]                │  │
│  │        OR                          │  │
│  │ [Provide Correction]               │  │
│  │ ┌────────────────────────────────┐ │  │
│  │ │ Corrected answer text area     │ │  │
│  │ └────────────────────────────────┘ │  │
│  │ [Submit Correction]                │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Backend: /submit_correction             │
│  ┌────────────────────────────────────┐  │
│  │ 1. Get original Q&A from history   │  │
│  │ 2. Update chat_history             │  │
│  │ 3. Insert into knowledge_base      │  │
│  │    - question                      │  │
│  │    - original_answer               │  │
│  │    - corrected_answer              │  │
│  │    - created_at                    │  │
│  │    - chat_history_id               │  │
│  └────────────────────────────────────┘  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Database: knowledge_base Table          │
│  ┌────────────────────────────────────┐  │
│  │ Stores verified/corrected answers  │  │
│  │ Used in future /ask calls          │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

---

## Knowledge Base Learning Loop

```
     ┌─────────────────────────────────────┐
     │                                     │
     │         CONTINUOUS LEARNING         │
     │                                     │
     └─────────────────────────────────────┘
              ▲                    │
              │                    │
              │                    ▼
     ┌────────────────┐   ┌────────────────┐
     │  Knowledge     │   │  User asks     │
     │  Base grows    │   │  question      │
     └────────────────┘   └────────────────┘
              ▲                    │
              │                    │
              │                    ▼
     ┌────────────────┐   ┌────────────────┐
     │  Correction    │   │  System checks │
     │  saved to KB   │   │  KB for match  │
     └────────────────┘   └────────────────┘
              ▲                    │
              │                    │
              │                    ▼
     ┌────────────────┐   ┌────────────────┐
     │  User provides │   │  LLM generates │
     │  correction    │   │  answer w/ KB  │
     └────────────────┘   └────────────────┘
              ▲                    │
              │                    │
              │                    ▼
     ┌────────────────┐   ┌────────────────┐
     │  User reviews  │◄──│  Answer shown  │
     │  answer        │   │  to user       │
     └────────────────┘   └────────────────┘
```

---

## Database Schema Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                    documents                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ id (PK)                                               │  │
│  │ filename                                              │  │
│  │ upload_timestamp                                      │  │
│  │ content                                               │  │
│  │ author          ◄── Citation Fields                  │  │
│  │ title           ◄── Citation Fields                  │  │
│  │ publication_date◄── Citation Fields                  │  │
│  │ source          ◄── Citation Fields                  │  │
│  │ doi_url         ◄── Citation Fields                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 1:N
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 document_chunks                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ id (PK)                                               │  │
│  │ document_id (FK) ──► documents.id                     │  │
│  │ chunk_text                                            │  │
│  │ chunk_index                                           │  │
│  │ embedding_json                                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   chat_history                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ id (PK)                                               │  │
│  │ user_id (FK) ──► users.id                             │  │
│  │ ts                                                    │  │
│  │ selected_text                                         │  │
│  │ question                                              │  │
│  │ answer                                                │  │
│  │ rating                                                │  │
│  │ comment                                               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ 1:1 (optional)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  knowledge_base                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ id (PK)                                               │  │
│  │ question                                              │  │
│  │ original_answer                                       │  │
│  │ corrected_answer                                      │  │
│  │ created_at                                            │  │
│  │ chat_history_id (FK) ──► chat_history.id              │  │
│  │ context_text                                          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      users                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ id (PK)                                               │  │
│  │ username                                              │  │
│  │ password                                              │  │
│  │ token                                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## API Endpoints Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    UPLOAD ENDPOINTS                         │
├─────────────────────────────────────────────────────────────┤
│ POST /upload                                                │
│   Input: files + citation metadata (Form)                  │
│   Output: Upload results with citation info                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY ENDPOINTS                         │
├─────────────────────────────────────────────────────────────┤
│ POST /ask                                                   │
│   Input: { selected_text, question, model, auth_token }    │
│   Process:                                                  │
│     1. Get document chunks                                  │
│     2. Get chat history                                     │
│     3. ★ Get knowledge base entries ★                       │
│     4. Build prompt with KB                                 │
│     5. Call LLM                                             │
│   Output: Streaming answer + entry_id                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  CORRECTION ENDPOINTS                       │
├─────────────────────────────────────────────────────────────┤
│ POST /submit_correction                                     │
│   Input: { chat_id, corrected_answer, comment }            │
│   Process:                                                  │
│     1. Update chat_history                                  │
│     2. Insert into knowledge_base                           │
│   Output: Success message                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                KNOWLEDGE BASE ENDPOINTS                     │
├─────────────────────────────────────────────────────────────┤
│ GET /knowledge_base                                         │
│   Output: List of all KB entries                           │
│                                                             │
│ POST /knowledge_base                                        │
│   Input: { items: [{ question, answer, source }] }         │
│   Output: Success message                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Functions

```python
# Backend utility functions

get_relevant_chunks(question, top_k=3)
  → Returns relevant document chunks using embeddings

get_relevant_history(question, selected_text, token, limit=3)
  → Returns relevant chat history entries

★ get_relevant_knowledge_base(question, limit=3) ★
  → Returns relevant KB entries based on question similarity
  → Uses keyword matching for relevance scoring
  → Enables learning from corrections

chunk_text(text, max_chunk_size=500)
  → Splits text into manageable chunks

embed_chunks(chunks, model)
  → Generates embeddings for semantic search

cosine_similarity(vec1, vec2)
  → Calculates similarity between embeddings
```

---

## Component Hierarchy

```
GridApp (Main Container)
├── Top (Header)
├── FileUpload
│   └── Citation Input Fields
├── TextPreview
│   └── Citation Display
├── AskModal
│   └── Question Input
├── Answer Display
│   └── Feedback Button
│       └── CorrectionModal
│           ├── Mark as Correct Button
│           └── Correction Textarea
└── History View
```

---

## Data Flow Example

### Example: User corrects an answer

```
1. User asks: "What is the capital of France?"
   
2. System:
   - Checks knowledge_base (no match found)
   - Checks document_chunks
   - Generates answer: "Lyon" (incorrect!)
   
3. User clicks "Provide Feedback"
   
4. CorrectionModal opens:
   Q: "What is the capital of France?"
   A: "Lyon"
   
5. User types: "Paris"
   
6. User clicks "Submit Correction"
   
7. Backend /submit_correction:
   - Updates chat_history.answer = "Paris"
   - Inserts into knowledge_base:
     {
       question: "What is the capital of France?",
       original_answer: "Lyon",
       corrected_answer: "Paris",
       created_at: "2025-12-02T21:00:00",
       chat_history_id: 42
     }
   
8. Next time someone asks similar question:
   - System finds KB entry
   - Includes in prompt:
     <knowledge_base>
     Q: What is the capital of France?
     A: Paris
     </knowledge_base>
   - LLM generates correct answer: "Paris"
```

---

This architecture enables:
- ✅ Traceable citations for academic integrity
- ✅ User-driven quality control
- ✅ Continuous learning from corrections
- ✅ Improved accuracy over time
- ✅ Knowledge base that grows with usage
