# ReadLess FastAPI Server ✅ SUCCESSFULLY CONVERTED

A professional, industry-grade FastAPI backend for conversational RAG (Retrieval-Augmented Generation) with PDF uploads and chat history.

## ✅ Conversion Status

**SUCCESS!** The Streamlit application has been successfully converted to a production-ready FastAPI backend while preserving all original functionality:

- ✅ All RAG logic preserved
- ✅ Chat history functionality maintained
- ✅ PDF processing workflow intact
- ✅ Session management implemented
- ✅ Professional API endpoints created
- ✅ Auto-generated documentation available
- ✅ CORS support enabled
- ✅ Error handling enhanced
- ✅ Industry-standard architecture implemented

## 🚀 Quick Start

### 1. Start the Server
```bash
# Option 1: Using PowerShell script (Recommended for Windows)
.\start_server.ps1

# Option 2: Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Using Python directly
python main.py
```

### 2. Access the API
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## 📊 Server Status
The server is currently **RUNNING** and ready to accept requests!

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **RAG Implementation**: Conversational RAG with PDF document processing
- **Chat History**: Persistent conversation history per session
- **PDF Processing**: Upload and process multiple PDF files
- **Session Management**: Secure session-based interactions
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **CORS Support**: Cross-origin resource sharing for web frontends
- **Type Safety**: Full Pydantic model validation
- **Error Handling**: Comprehensive error handling and logging

## API Endpoints

### Core Endpoints

- `GET /` - Health check
- `POST /session/create` - Create a new session
- `POST /upload` - Upload and process PDF files
- `POST /ask` - Ask questions to the RAG system
- `GET /session/{session_id}/history` - Get chat history
- `DELETE /session/{session_id}` - Delete session
- `GET /sessions` - List all active sessions

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

## Installation

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file with:
   ```
   HF_TOKEN=your_huggingface_token
   ```

## Usage

### Start the Server

```bash
# Using the startup script
python run_server.py

# Or directly with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### API Usage Examples

#### 1. Create Session
```bash
curl -X POST "http://localhost:8000/session/create" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "groq_api_key": "your_groq_api_key"
  }'
```

#### 2. Upload PDFs
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "session_id=user123" \
  -F "groq_api_key=your_groq_api_key" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

#### 3. Ask Questions
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "question": "What is the main topic of the documents?",
    "groq_api_key": "your_groq_api_key"
  }'
```

#### 4. Get Chat History
```bash
curl -X GET "http://localhost:8000/session/user123/history"
```

## Configuration

The application can be configured through environment variables or the `config.py` file:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `HF_TOKEN`: HuggingFace token for embeddings
- `CHUNK_SIZE`: Text chunk size for processing (default: 5000)
- `CHUNK_OVERLAP`: Text chunk overlap (default: 500)

## Architecture

### Key Components

1. **FastAPI Application** (`app.py`): Main API server
2. **Configuration** (`config.py`): Application settings
3. **Utilities** (`utils.py`): Helper functions and session management
4. **Models**: Pydantic models for request/response validation

### Session Management

- Sessions store chat history and document retrievers
- Automatic session cleanup for expired sessions
- Session-based isolation of conversations

### Document Processing

- PDF upload and processing
- Text splitting and chunking
- Vector embeddings using HuggingFace models
- Chroma vector store for document retrieval

### RAG Pipeline

- History-aware retrieval
- Context-aware question answering
- Conversation memory management
- LLM integration with Groq

## Security Considerations

- API key validation
- File type validation
- File size limits
- Session isolation
- Input sanitization

## Monitoring and Logging

- Comprehensive logging for API calls
- Error tracking and reporting
- Session activity monitoring
- Performance metrics

## Production Deployment

For production deployment, consider:

1. **Environment Configuration**:
   - Set `reload=False` in production
   - Configure proper CORS origins
   - Use environment-specific settings

2. **Security**:
   - Implement authentication/authorization
   - Rate limiting
   - Input validation
   - HTTPS/TLS

3. **Scalability**:
   - Use Redis for session storage
   - Database for persistent storage
   - Load balancing
   - Containerization (Docker)

4. **Monitoring**:
   - Health checks
   - Metrics collection
   - Log aggregation
   - Error alerting

## Migrating from Streamlit

This FastAPI backend preserves all the original Streamlit logic while providing:

- RESTful API endpoints
- Better scalability
- Frontend flexibility
- Professional architecture
- Enhanced error handling
- Session management
- API documentation

The original Streamlit implementation is preserved as `streamlit_app.py`.
