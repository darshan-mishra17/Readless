# ReadLess - PDF Chat Application

A full-stack PDF chat application that allows users to upload PDF documents and ask questions about their content using AI.

## Architecture

### Backend (FastAPI)
- **Location**: `./Server/`
- **Technology**: FastAPI, Python, LangChain, Groq LLM
- **Model**: GPT-OSS-120B (best available Groq model)
- **Features**: 
  - RAG (Retrieval Augmented Generation) pipeline
  - Session-based conversations
  - PDF processing and vector storage
  - RESTful API with OpenAPI documentation

### Frontend (Next.js)
- **Location**: `./Frontend/`
- **Technology**: Next.js 15, React 19, TypeScript, Tailwind CSS
- **Features**:
  - Drag & drop PDF upload
  - Real-time chat interface
  - Dark/light mode toggle
  - Backend status indicator
  - Responsive design with Radix UI components

## Quick Start

### Prerequisites
- Python 3.8+ with virtual environment support
- Node.js 18+ and pnpm
- Groq API key

### Backend Setup

1. **Create and activate virtual environment:**
   ```bash
   cd Server
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   - Copy `.env.example` to `.env` (if available)
   - Set your Groq API key in `.env`:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     GROQ_MODEL=openai/gpt-oss-120b
     ```

4. **Start the backend server:**
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

   The API will be available at:
   - Main API: http://127.0.0.1:8000
   - Documentation: http://127.0.0.1:8000/docs
   - Health check: http://127.0.0.1:8000/health

### Frontend Setup

1. **Install dependencies:**
   ```bash
   cd Frontend
   pnpm install
   ```

2. **Start the development server:**
   ```bash
   pnpm dev
   ```

   The application will be available at: http://localhost:3000

## API Endpoints

### Session Management
- `POST /session/create-simple` - Create a new chat session

### Document Processing
- `POST /upload-simple` - Upload and process a PDF document
  - Form data: `file` (PDF file), `session_id` (string)

### Chat
- `POST /ask-simple` - Ask a question about the uploaded document
  - JSON body: `{"question": "string", "session_id": "string"}`

### Health
- `GET /health` - Check server status

## Features

### Backend Features
- **RAG Pipeline**: Uses LangChain for document processing and retrieval
- **Vector Storage**: Chroma for efficient document search
- **Session Management**: Maintains conversation history per session
- **Error Handling**: Comprehensive error responses
- **CORS Support**: Configured for frontend integration
- **Lazy Loading**: Embeddings loaded on first use to prevent startup delays

### Frontend Features
- **File Upload**: Drag & drop interface with progress tracking
- **Real-time Status**: Backend connection indicator in navbar
- **Chat Interface**: Conversation history with timestamps
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: User-friendly error messages
- **Loading States**: Visual feedback for all operations

## Environment Variables

### Backend (.env)
```env
GROQ_API_KEY=gsk_your_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### Frontend
No environment variables required. API endpoint is configured in `lib/api.ts`.

## Development

### Backend Development
- **Configuration**: Modify `config.py` for settings
- **API Routes**: Main routes in `main.py`
- **RAG Logic**: Vector store and retrieval in main application
- **Testing**: Use `test_gpt_oss_120b.py` for API testing

### Frontend Development
- **API Integration**: See `lib/api.ts` for API client
- **Components**: UI components in `components/ui/`
- **Styling**: Tailwind CSS with custom design system
- **Type Safety**: Full TypeScript implementation

## Testing

### Backend Testing
```bash
cd Server
python test_gpt_oss_120b.py
```

### Frontend Testing
The frontend includes real-time status checking and error handling for backend connectivity.

## Troubleshooting

### Common Issues

1. **Backend won't start**: 
   - Check if virtual environment is activated
   - Verify all dependencies are installed
   - Ensure Groq API key is set correctly

2. **Frontend shows "Backend Offline"**:
   - Verify backend is running on http://127.0.0.1:8000
   - Check backend health endpoint
   - Ensure no firewall is blocking the connection

3. **Upload fails**:
   - Check file size (max 10MB)
   - Ensure session is created successfully
   - Verify backend logs for processing errors

4. **Chat responses fail**:
   - Confirm document was uploaded successfully
   - Check Groq API key and quota
   - Review backend logs for errors

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **LangChain**: LLM application framework
- **Groq**: High-performance LLM inference
- **Chroma**: Vector database for embeddings
- **HuggingFace**: Embedding models
- **PyPDF**: PDF text extraction

### Frontend
- **Next.js 15**: React framework with app router
- **React 19**: Latest React with concurrent features
- **TypeScript**: Type-safe development
- **Tailwind CSS 4**: Utility-first styling
- **Radix UI**: Accessible component primitives
- **Lucide React**: Beautiful icons

## Project Structure

```
ReadLess/
├── Server/                 # FastAPI backend
│   ├── main.py            # Main application
│   ├── config.py          # Configuration
│   ├── utils.py           # Utility functions
│   ├── requirements.txt   # Python dependencies
│   ├── .env              # Environment variables
│   └── test_gpt_oss_120b.py  # API tests
│
├── Frontend/              # Next.js frontend
│   ├── app/              # App router pages
│   ├── components/       # React components
│   ├── lib/              # Utilities and API client
│   ├── hooks/            # Custom hooks
│   ├── styles/           # Global styles
│   └── package.json      # Node dependencies
│
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
