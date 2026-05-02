# SynergeReader

A browser-based document reader with AI-powered question answering capabilities. Upload PDF, DOCX, or TXT documents, select text, and ask questions to get intelligent answers using PostgreSQL pgvector search and LLM integration.

## Features

### ✅ **Week 1-4 Complete Implementation**

- **Document Upload & Processing**: Support for PDF, DOCX, and TXT files (max 20MB)
- **Local Text Parsing**: Client-side parsing using pdf.js and mammoth.js
- **Text Selection**: Interactive text selection with automatic question modal
- **Vector Database**: PostgreSQL pgvector integration for document chunking and embedding
- **Question Analysis**: Intelligent question analysis and intent recognition
- **Vector Similarity Search**: Find relevant document chunks using embeddings
- **History Retrieval**: Smart retrieval of relevant past Q&A pairs
- **LLM Integration**: OpenRouter API with enhanced prompt building
- **Chat History**: Persistent storage of all Q&A interactions
- **Modern UI**: Clean, responsive interface with real-time feedback

## Architecture

```
Frontend (React) ←→ FastAPI Backend ←→ PostgreSQL + pgvector
```

### Backend Components
- **FastAPI**: Modern async web framework
- **PostgreSQL + pgvector**: Vector database for document embeddings
- **Sentence Transformers**: Text embedding generation
- **PostgreSQL**: Chat history storage
- **OpenRouter API**: LLM integration

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
2. **Parsing**: Extract text using appropriate parser (pdf.js/mammoth.js)
3. **Chunking**: Split text into overlapping chunks (1000 chars, 200 overlap)
4. **Embedding**: Generate embeddings using sentence-transformers
5. **Storage**: Store chunks and embeddings in PostgreSQL pgvector

### Question Answering Pipeline
1. **Analysis**: Analyze question intent and extract key terms
2. **Vector Search**: Find similar document chunks using embeddings
3. **History Retrieval**: Find relevant past Q&A pairs
4. **Prompt Building**: Construct comprehensive prompt with context
5. **LLM Call**: Generate answer using OpenRouter API
6. **Storage**: Save Q&A to PostgreSQL history

## Configuration

### Environment Variables
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap size (default: 200)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 20MB)

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **LLM Model**: `meta-llama/llama-3.3-70b-instruct:free` (OpenRouter)
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
- **LLM Integration**: Enhanced prompt building and API calls
- **History Management**: PostgreSQL CRUD operations

#### Frontend Components
- **FileUpload.js**: Multi-format file upload with parsing
- **TextPreview.js**: Interactive text display and selection
- **AskModal.js**: Question input with context display
- **App.jsx**: Main application state and API integration

## Performance Considerations

- **Chunking Strategy**: Overlapping chunks preserve context across boundaries
- **Embedding Caching**: Sentence transformers model loaded once
- **Vector Search**: Efficient similarity search with pgvector
- **Response Streaming**: Future enhancement for real-time answers

## Security

- **File Validation**: Type and size checking
- **CORS Configuration**: Proper cross-origin setup
- **API Key Management**: Environment variable storage
- **Input Sanitization**: Text processing safety

## Future Enhancements

- **Streaming Responses**: Real-time answer generation
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

3. **LLM API Errors**
   - Verify OpenRouter API key is valid
   - Check API rate limits

4. **Vector Search Issues**
   - Ensure PostgreSQL is running and the vector extension is enabled
   - Check embedding model download

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
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [OpenRouter](https://openrouter.ai/) for LLM access
- [React](https://reactjs.org/) for the frontend framework
"# SynergeReader" 
