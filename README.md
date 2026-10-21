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
| GET | `/version` | Public, read-only release identity (`/api/version` behind nginx) — see "Release identity and answer provenance" |

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
- **Embedding profile**: `main.py` wires every `EMBEDDING_*` variable through `resolve_embedding_profile(os.environ)` at startup. Leaving every `EMBEDDING_*` variable unset selects the documented mxbai/1024 default; using a different model (e.g. the Nomic/768 example) requires the complete explicit override described in `.env.example`. A complete override may match the existing column dimension without thereby establishing representation compatibility with the vectors already stored. An invalid, missing, malformed, wrong-count, wrong-dimension, non-finite, or all-zero embedding response fails closed rather than being persisted as a NULL or fabricated vector.
- **Schema compatibility**: the last read-only production inventory reported both existing vector columns as 768-dimensional, so against that deployment the resolved profile's dimension must match or the backend refuses to start. A fresh database instead creates its vector columns from the resolved `expected_dimension`. Separately, `docker-compose.yml` defines six backend pass-through entries intended to carry the complete embedding-profile configuration, but actual Compose 1.29.2 resolution and propagation — including intentionally empty prefix values — remains unverified and is a release-gate check.
- **Answer model registry**: `generation_profiles.py` is the backend-owned registry of answer models (`llama_control` → `llama3.1:8b`, the unqualified control and default; `gemma_candidate` → `gemma4:31b`, which requires qualification and stays unselectable until combined-capacity and quality evidence are recorded — only its pinned digest and `think: false` behaviour are evidenced). Each profile pins the expected digest of its installed artifact; a qualification-required profile, explicit or default, answers only after its installed digest matches. `/ask` resolves the model once per answer and sends native Ollama `options` (temperature 0.2, `num_ctx` 8192, `num_predict` 1200, `keep_alive` 5m) to local Ollama with a per-profile read timeout (Llama 60 s, Gemma 150 s); there is no external generation fallback. A request's legacy `model` string is ignored, and an explicit `model_profile_id` is honoured only for an active admin. Claim verification and the background helpers always use a fixed, bounded Llama call. The older `GenerationProfile`/`GENERATION_*` definitions in `rag_model_profiles.py` remain unused by `/ask`.

## Configuration

### Environment Variables
- `OLLAMA_BASE_URL`: Optional explicit Ollama service URL
- `OLLAMA_HOST` and `OLLAMA_PORT`: Ollama connection host and port when no base URL is set
- `OLLAMA_FALLBACK_HOSTS`: Comma-separated fallback hosts for local Ollama discovery
- `EMBEDDING_*`: optional strict embedding profile override, wired through `resolve_embedding_profile(os.environ)` in `main.py` — see `.env.example` for the full set and the all-or-nothing rules that govern them
- `GENERATION_*`: profile definitions exist in `rag_model_profiles.py` (`resolve_generation_profile()`), but nothing in `main.py` reads these, so setting them has no effect on `/ask`; answer models and their options come from `generation_profiles.py`
- `OLLAMA_KEEP_ALIVE`: keep-alive for embedding requests only; generation keep-alive comes from the model registry
- `SYNERGE_BUILD_SHA`: the full 40-character commit this deployment was built from, injected by the release operator — see "Release identity and answer provenance" below
- `SYNERGE_EXPERIMENTAL_MODELS` and `SYNERGE_COMPAT_OLLAMA_URL`: both off/unset by default. Together they make the three experimental reasoning profiles (Qwen3.8 27B, Nemotron 3.5 Lightning 30B, Mistral Medium 3.5 128B) selectable by an administrator; they run on a separate private Ollama runtime while Gemma, Llama and the embedder stay on `OLLAMA_BASE_URL` — see `synerge-reader-compat-runtime/README.md`
- **Chunk size**: not environment-driven. `document_chunker.py` currently uses a code default of 500 characters per chunk with no overlap.
- **File size limits**: not environment-driven, and enforced at two different layers with two different values — the current GridApp upload UI rejects files over 20MB client-side, while the backend's own parser (`document_parser.py`) accepts up to 50MB.

### Model Configuration
- **Embedding Model**: controlled by the resolved embedding profile (see `.env.example`). A matching dimension is not representation compatibility — a profile whose dimension fits an existing column can still encode text differently from the vectors already stored, so each candidate carries its own precondition:
  - **mxbai/1024** — the repository default. It does not match the 768 dimensions the last read-only production inventory reported for both existing vector columns, so against that deployment startup fails closed until a separately reviewed schema migration and a controlled re-embedding are approved and completed. A fresh database instead creates its vector columns from the resolved `expected_dimension`.
  - **Unprefixed Nomic/768** (both prefix keys present and intentionally empty) — a conditional legacy-compatibility candidate only, pending verification of Compose propagation, of the exact model digest/runtime/endpoint behavior, and of the provenance of the surviving stored vectors.
  - **Prefixed Nomic/768** (quoted, load-bearing trailing spaces retained) — changes the representation relative to the reported legacy unprefixed corpus, and requires a controlled re-embedding before activation.
- **Profile selection is not authorized here**: no profile has been empirically validated against retrieval quality, and this documentation authorizes no deployment, migration, re-embedding, model switch, or restart.
- **LLM Model**: Ollama generation through the configured service, selected per request, with no application-level external generation fallback
- **Vector Space**: Cosine distance in pgvector

### Release identity and answer provenance

**Versions.** `synerge-reader-backend/release_metadata.py` is the one authoritative source of:

- the application version, semantic versioning, currently `0.1.0`;
- the answer-prompt version, `rag-answer-v1` (the templates in `citation_generation.py`);
- the claim-verifier prompt version, `claim-support-v1` (`_CLAIM_VERIFIER_PROMPT` in `main.py`). The two prompts are maintained separately, so they are versioned separately. `tests/test_release_metadata.py` pins a fingerprint of each template, so editing a template without bumping its version fails the tests. Text the frontend puts into the question (such as task-mode prefixes) is request input and is not covered by these versions.

**Build commit injection.** The deployed commit is supplied by the release operator, never discovered by the application:

- The value must be the **full 40-character SHA of the exact reviewed commit or merge commit being released**.
- The backend reads it once at startup. It never runs Git and never inspects its source directory: a deployed tree may be a copy, may be dirty, or may not be a repository at all.
- A valid value is normalized to lowercase. An abbreviated, prefixed, non-hex, empty, or missing value is reported as unknown. It is never replaced by a guessed or local commit. Local development works with it unset.
- `docker-compose.yml` forwards it to the backend container only, as a bare `- SYNERGE_BUILD_SHA` entry: when it is supplied it reaches the backend, and when it is not supplied it stays absent. No commit is written into any tracked file.

There are two ways to supply it:

- **GitHub workflow (`.github/workflows/push_to_server.yml`, manually triggered).** The `Build & Start Containers` step sets `SYNERGE_BUILD_SHA: ${{ github.sha }}`. That is the commit `actions/checkout` checked out for the run, and therefore the commit the step syncs and deploys. Both `docker-compose build` and `docker-compose up` run with it in their environment, Compose forwards it to the backend, and `/api/version` reports it. Trigger the workflow only on the reviewed release commit.
- **Manual operator deployment.** Before running Compose, set `SYNERGE_BUILD_SHA` to the full 40-character SHA of the exact reviewed release commit in the shell that runs `docker-compose up` (for example `export SYNERGE_BUILD_SHA=<full reviewed SHA>`). Whether docker-compose 1.29.2 on the server also resolves this bare entry from the root `.env` is unverified, as with the `EMBEDDING_*` entries; setting it in the shell avoids depending on that.

Either way, the deployment is verified only by the post-deployment gate below.

**`GET /api/version`** (backend route `/version`) is public and read-only. It returns exactly `app_version`, `build_sha`, `build_sha_short` (7 characters), `build_known` and `display_version`, with `Cache-Control: no-store`. It never includes model inventory, profiles, endpoints, paths, environment values, or database details. The app sidebar shows `v0.1.0 · <short SHA>`, or `v0.1.0 · build unknown`, or `Version unavailable` when the request fails. The display only shows what the server reports. It does not prove a deployment by itself.

**Post-deployment gate.** After deploying, fetch `/api/version` from the public site. Confirm that `build_known` is `true` and that `build_sha` is exactly the full 40-character SHA of the reviewed release commit (the short form alone is not enough). If it says `build unknown` or names any other commit, the deployment is not verified. Answers saved by a build with an unknown commit store `build_sha: null`, never a guessed SHA.

**Answer provenance.** Each successfully saved answer row in `chat_history` carries two nullable columns, added additively at startup by `ensure_chat_history_provenance_columns` (with no backfill and no table rewrite):

- `trace_id` (`UUID`): a server-generated id for that answer attempt. It also appears in the backend's `[Ask]` log line, so failed attempts can be correlated in logs, but only a saved answer stores it.
- `answer_provenance` (`JSONB`, `schema_version: 1`): built in `answer_provenance.py` only from server values, never from the request. It records:
  - the release (`app_version`, `build_sha` or `null`);
  - the snapshotted generation profile: profile id, exact model tag, `default` or `admin_override`, the registry's expected digest, and separately the observed digest (`null` when no installed-model lookup was made, as for the default Llama);
  - the requested Ollama options, `think` and `keep_alive`;
  - both prompt versions and the answer mode;
  - the verifier call, only for graded `document_qa` answers;
  - the evidence mode and the *configured* embedding profile. Actual embedding use is not recorded, because it cannot be derived reliably;
  - server timing in monotonic milliseconds from request receipt;
  - Ollama's final-record counters and nanosecond durations, `null` when not reported.

  It never stores prompt text, evidence text, thinking output, credentials, endpoints, or paths. The answer and its provenance are written in the same `INSERT`, so either both are saved or neither is. Failed, cut-off, empty, and whitespace-only generations save neither.

Rows saved before this change keep `NULL` in both columns. `NULL` means "not recorded". It does not mean Llama, the current release, or the current prompt. Provenance is returned only by the admin-only `GET /admin/chat_history`. The answer stream, `POST /history`, and every other route are unchanged and do not return it.

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
   - Check that the registry's answer model (`llama3.1:8b` by default) is installed in Ollama

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
