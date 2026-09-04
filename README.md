# SynergeReader

A browser-based document reader with AI-powered question answering capabilities. Upload PDF, DOCX, or TXT documents, select text, and ask questions to get intelligent answers using PostgreSQL pgvector search and LLM integration.

## Features

### ✅ **Week 1-4 Complete Implementation**

- **Document Upload & Processing**: Support for PDF, DOCX, and TXT files (max 20MB in the current GridApp UI; the backend parser itself accepts up to 50MB)
- **Local Text Parsing**: Client-side parsing using pdf.js and mammoth.js
- **Text Selection**: Interactive text selection with automatic question modal
- **Vector Database**: PostgreSQL pgvector integration for document chunking and embedding
- **Question Analysis**: Intelligent question analysis and intent recognition
- **Vector Similarity Search**: Find relevant document chunks using embeddings
- **History Retrieval**: Smart retrieval of relevant past Q&A pairs
- **LLM Integration**: Local Ollama answer generation with no external generation fallback
- **Chat History**: Persistent storage of all Q&A interactions
- **Modern UI**: Clean, responsive interface with real-time feedback

## Architecture

```
Frontend (React) ←→ FastAPI Backend ←→ PostgreSQL + pgvector
```

### Backend Components
- **FastAPI**: Modern async web framework
- **PostgreSQL + pgvector**: Vector database for document embeddings
- **Ollama Embeddings**: Local embedding generation through the configured embedding profile
- **PostgreSQL**: Chat history storage
- **Ollama Generation**: Local streaming answer generation

### Frontend Components
- **React**: Modern UI framework
- **pdf.js**: PDF parsing
- **mammoth.js**: DOCX parsing
- **Text Selection**: Interactive document reading

## Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/synerge-reader.git
   cd synerge-reader
   ```

2. **Start the application**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

### Option 2: Local Development

#### Backend Setup
```bash
cd synerge-reader-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requiredInstall.txt
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

#### Frontend Setup
```bash
cd synerge-reader-frontend
npm install
npm start
```

## Usage

1. **Upload Document**: Drag & drop or browse for PDF, DOCX, or TXT files
2. **Select Text**: Click and drag to select text from the document
3. **Ask Questions**: The question modal will automatically open
4. **Get Answers**: Receive AI-powered answers with relevant context
5. **View History**: Browse past questions and answers

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload and process document |
| POST | `/ask` | Ask a question with context |
| GET | `/history` | Retrieve chat history |
| GET | `/test` | Health check endpoint |

## Technical Implementation

### Document Processing Pipeline
1. **Upload**: File validation and size checking
2. **Parsing**: The backend's `/upload` parser preserves PDF page boundaries, but only when it is given the original PDF bytes. The current GridApp upload path parses PDF/DOCX/TXT client-side (pdf.js/mammoth.js) and sends the backend flattened plain text instead of the original file, so page provenance is **not** yet preserved through that live UI path — see "RAG Pipeline Integration" below. This is deferred to E1a.
3. **Chunking**: Split each document into page-aware chunks (`document_chunker.py`); each chunk records its own page range when page numbers are available
4. **Embedding**: Generate embeddings through the resolved Ollama embedding profile
5. **Storage**: Store the document and its chunks — including each chunk's page locator — in one PostgreSQL pgvector transaction

### Question Answering Pipeline
1. **Analysis**: Analyze question intent and extract key terms
2. **Vector Search**: Find similar document chunks using embeddings (not every `/ask` context mode calls this — see "RAG Pipeline Integration" below)
3. **History Retrieval**: Find relevant past Q&A pairs
4. **Prompt Building**: Construct comprehensive prompt with context
5. **LLM Call**: Generate the answer using local Ollama generation
6. **Storage**: Save Q&A to PostgreSQL history

### RAG Pipeline Integration (Backend)
- **Page-aware chunking**: document uploads are split into page-aware chunks, and a document's row plus all of its chunk rows (each with `page_start`, `page_end`, and `locator_json`) are written in one database transaction.
- **Retrieval and locators**: page-aware chunk locator metadata is persisted in the database and reconstructed by the backend's internal retrieval helper. The current `/ask` stream and frontend do not yet expose or render that locator metadata as user-visible citations, and not every `/ask` context-building mode calls vector retrieval (e.g. an active document or an explicit text selection is used directly).
- **Embedding profile**: `main.py` wires every `EMBEDDING_*` variable through `resolve_embedding_profile(os.environ)` at startup. Leaving every `EMBEDDING_*` variable unset selects the documented mxbai/1024 default; using a different model (e.g. the current schema-compatible Nomic/768 example) requires the complete explicit override described in `.env.example`. An invalid, missing, malformed, wrong-count, wrong-dimension, non-finite, or all-zero embedding response fails closed rather than being persisted as a NULL or fabricated vector.
- **Schema compatibility**: the database schema is currently provisioned for 768-dimensional vectors, so the resolved embedding profile's dimension must match it or the backend refuses to start. Whether a deployment's `EMBEDDING_*` values actually reach the backend container through Compose is a separate, not-yet-verified deployment question (see `docker-compose.yml`); this has not been verified for the Compose 1.29.2 `.env` propagation path used here.
- **Generation profile (foundation only)**: `rag_model_profiles.py` defines `GenerationProfile`/`resolve_generation_profile()` and the `GENERATION_*` variables, but nothing in `main.py` calls them yet — they are not wired into `/ask`. The generation model actually used today is the one selected in the request and sent directly to local Ollama; there is no external generation fallback.

## Configuration

### Environment Variables
- `OLLAMA_BASE_URL`: Optional explicit Ollama service URL
- `OLLAMA_HOST` and `OLLAMA_PORT`: Ollama connection host and port when no base URL is set
- `OLLAMA_FALLBACK_HOSTS`: Comma-separated fallback hosts for local Ollama discovery
- `EMBEDDING_*`: optional strict embedding profile override, wired through `resolve_embedding_profile(os.environ)` in `main.py` — see `.env.example` for the full set and the all-or-nothing rules that govern them
- `GENERATION_*`: profile definitions exist in `rag_model_profiles.py` (`resolve_generation_profile()`), but this is foundation-only — nothing in `main.py` reads these yet, so setting them today has no effect on `/ask`
- **Chunk size**: not environment-driven. `document_chunker.py` currently uses a code default of 500 characters per chunk with no overlap.
- **File size limits**: not environment-driven, and enforced at two different layers with two different values — the current GridApp upload UI rejects files over 20MB client-side, while the backend's own parser (`document_parser.py`) accepts up to 50MB.

### Model Configuration
- **Embedding Model**: controlled by the resolved embedding profile; the documented default is mxbai/1024, and Nomic/768 is the current schema-compatible explicit-override example (see `.env.example`) — neither has been empirically validated against retrieval quality yet
- **LLM Model**: Ollama generation through the configured service, selected per request, with no application-level external generation fallback
- **Vector Space**: Cosine distance in pgvector

## Development

### Project Structure
```
synerge-reader/
├── synerge-reader-backend/
│   ├── main.py              # FastAPI application
│   ├── requiredInstall.txt  # Python dependencies
│   ├── Dockerfile          # Backend container
│   └── dbSetup.py          # PostgreSQL schema setup
├── synerge-reader-frontend/
│   ├── src/
│   │   ├── App.jsx         # Main application
│   │   └── components/     # React components
│   ├── package.json        # Node dependencies
│   └── Dockerfile         # Frontend container
├── docker-compose.yml     # Multi-container setup
└── README.md             # This file
```

### Key Components

#### Backend (`main.py`)
- **Document Processing**: Chunking, embedding, vector storage
- **Question Analysis**: Intent recognition and key term extraction
- **Vector Search**: Similarity-based document retrieval
- **LLM Integration**: Enhanced prompt building and local Ollama streaming generation
- **History Management**: PostgreSQL CRUD operations

#### Frontend Components
- **FileUpload.js**: Multi-format file upload with parsing
- **TextPreview.js**: Interactive text display and selection
- **AskModal.js**: Question input with context display
- **App.jsx**: Main application state and API integration

## Performance Considerations

- **Chunking Strategy**: `document_chunker.py` splits each document into non-overlapping, page-aware chunks targeting 500 characters and split on word boundaries
- **Embedding Profile Reuse**: Backend embedding calls share the configured Ollama provider and profile
- **Vector Search**: Efficient similarity search with pgvector
- **Response Streaming**: Local Ollama answers are already streamed to the client via `StreamingResponse`, not a future enhancement

## Security

- **File Validation**: Type and size checking
- **CORS Configuration**: Proper cross-origin setup
- **API Key Management**: Environment variable storage
- **Input Sanitization**: Text processing safety

## Future Enhancements

- **Client-Visible Citations**: Return the already-persisted page locators through `/ask` and render them in the UI
- **Page-Provenant Frontend Upload**: Send original PDF bytes (or page boundaries) from GridApp instead of flattened client-parsed text, so `/upload`'s existing page-aware parsing applies to the live UI path (E1a)
- **Generation Profile Wiring**: Wire `resolve_generation_profile()`/`GENERATION_*` into `/ask` instead of the current per-request model selection
- **Multi-Document Support**: Cross-document question answering
- **Advanced Parsing**: Better PDF/DOCX formatting preservation
- **User Authentication**: Multi-user support
- **Export Features**: Save Q&A sessions
- **Advanced Analytics**: Usage statistics and insights

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Check if FastAPI server is running on port 5000
   - Verify CORS configuration

2. **Document Upload Fails**
   - Ensure file size is under 20MB
   - Check file format (PDF, DOCX, TXT only)

3. **Local LLM Errors**
   - Verify that the Ollama service is reachable using the configured connection settings
   - Check that the requested generation model is installed in Ollama

4. **Vector Search Issues**
   - Ensure PostgreSQL is running and the vector extension is enabled
   - Check that the configured Ollama embedding model is available

### Logs
- Backend logs: `docker-compose logs backend`
- Frontend logs: `docker-compose logs frontend`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [pgvector](https://github.com/pgvector/pgvector) for vector database
- [Ollama](https://ollama.com/) for local embedding and generation services
- [React](https://reactjs.org/) for the frontend framework
"# SynergeReader" 
