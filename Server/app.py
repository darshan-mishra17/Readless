"""
FastAPI RAG Q&A Conversation With PDF Including Chat History
Industry-grade backend implementation preserving all original logic
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, SecretStr
from typing import List, Optional, Dict, Any
import os
import tempfile
import uuid
from datetime import datetime
import asyncio
import logging
import numpy as np

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Voice chat imports
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get default GROQ API key from environment
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not DEFAULT_GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables")

# Get GROQ model from environment
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # fallback to a working model
print(f"Using GROQ model: {DEFAULT_GROQ_MODEL}")

# Voice chat globals
whisper_model = None
tts_engine = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="ReadLess RAG API",
    description="Conversational RAG with PDF uploads, chat history, and voice chat",
    version="1.0.0"
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to initialize voice models
@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize voice models on startup"""
    logger.info("🚀 Starting ReadLess RAG API with voice chat...")
    await initialize_voice_models()

# Load environment variables
load_dotenv()

# Get default GROQ API key from environment
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not DEFAULT_GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables")

# Get GROQ model from environment
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # fallback to a working model
print(f"Using GROQ model: {DEFAULT_GROQ_MODEL}")

# Voice chat globals
whisper_model = None
tts_engine = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="ReadLess RAG API",
    description="Conversational RAG with PDF uploads and chat history",
    version="1.0.0"
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize embeddings
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

# Initialize embeddings with memory optimization for Render
embeddings = None  # Initialize later on-demand to save memory

# Global storage for sessions (In production, use Redis or database)
session_store: Dict[str, ChatMessageHistory] = {}
session_chains: Dict[str, Any] = {}
session_retrievers: Dict[str, Any] = {}

# Pydantic models
class SessionRequest(BaseModel):
    session_id: str = "default_session"
    groq_api_key: Optional[str] = None

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str
    status: str

class AnswerResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    chat_history: List[Dict[str, Any]]
    status: str

class UploadResponse(BaseModel):
    session_id: str
    files_processed: List[str]
    message: str
    status: str

class VoiceChatResponse(BaseModel):
    transcribed_text: str
    ai_response: str
    audio_duration: Optional[float] = None
    processing_time: float

class VoiceChatRequest(BaseModel):
    text: str
    session_id: str
    voice_speed: Optional[float] = 1.0

# Helper functions (preserving original logic)
def get_embeddings():
    """Get embeddings instance (lazy loading for memory optimization)"""
    global embeddings
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Force CPU to save memory
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Warning: Could not initialize HuggingFace embeddings: {e}")
            print("You may need to set the HF_TOKEN environment variable")
            embeddings = None
    return embeddings

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session history - preserving original logic"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def create_rag_chain(groq_api_key: str, retriever, session_id: str):
    """Create RAG chain - preserving original logic"""
    # Set GROQ_API_KEY environment variable for this request
    import os
    original_key = os.environ.get('GROQ_API_KEY')
    os.environ['GROQ_API_KEY'] = groq_api_key
    
    try:
        # Create LLM instance using only environment variable (no direct api_key parameter)
        llm = ChatGroq(model=DEFAULT_GROQ_MODEL, temperature=0)
        
        # Contextualize question prompt - same as original
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Answer question prompt - more restrictive to prevent hallucination with markdown formatting
        qa_system_prompt = (
            "You are a strict document assistant. Your ONLY job is to answer questions based EXCLUSIVELY on the provided context from the uploaded PDF document. "
            "CRITICAL RULES:\n"
            "1. ONLY use information from the context provided below\n"
            "2. If the context doesn't contain relevant information to answer the question, respond with: 'I cannot find information about this in the uploaded document.'\n"
            "3. NEVER generate information that is not explicitly stated in the context\n"
            "4. Quote or reference specific parts of the document when possible\n"
            "5. Keep responses concise and directly related to the document content\n"
            "6. Do NOT use external knowledge - ONLY the document context\n"
            "7. Format your response in clear, readable markdown with proper headings, bullet points, and emphasis when appropriate\n"
            "8. Use **bold** for important terms, *italics* for emphasis, and `code blocks` for technical terms\n"
            "9. Structure your response with headers (##) and bullet points (-) when listing information\n\n"
            "DOCUMENT CONTEXT:\n{context}\n\n"
            "Based ONLY on the above context, provide a well-formatted markdown response to the following question:"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Create conversational chain with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversational_rag_chain
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create RAG chain: {str(e)}")
    
    finally:
        # Always restore the original environment variable
        if original_key:
            os.environ['GROQ_API_KEY'] = original_key
        else:
            os.environ.pop('GROQ_API_KEY', None)

# Voice chat helper functions
async def initialize_voice_models():
    """Initialize voice models if available - memory optimized"""
    global whisper_model, tts_engine
    
    try:
        # Only initialize Whisper if explicitly requested (save memory)
        if WHISPER_AVAILABLE and whisper_model is None and os.getenv("ENABLE_WHISPER", "false").lower() == "true":
            logger.info("🎤 Loading Whisper STT model...")
            # Use the smallest model to save memory
            whisper_model = whisper.load_model("tiny", device="cpu")
            logger.info("✅ Whisper tiny model loaded on CPU")
        elif not WHISPER_AVAILABLE:
            logger.info("🎤 Whisper not available - STT will use browser Web Speech API")
        
        # Only initialize TTS if available and requested
        if TTS_AVAILABLE and tts_engine is None and os.getenv("ENABLE_TTS", "false").lower() == "true":
            logger.info("🔊 Initializing TTS engine...")
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
            tts_engine.setProperty('volume', 0.8)
            logger.info("✅ TTS engine initialized")
        elif not TTS_AVAILABLE:
            logger.info("🔊 TTS not available - will use browser Web Speech API")
            
    except Exception as e:
        logger.warning(f"⚠️ Voice models initialization failed: {e}")
        logger.info("Voice chat will fall back to browser-based speech APIs")

async def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe audio using Whisper"""
    try:
        if not WHISPER_AVAILABLE or whisper_model is None:
            raise HTTPException(status_code=503, detail="Speech recognition not available")
        
        result = whisper_model.transcribe(audio_file_path)
        # Handle both dict and other result types
        if isinstance(result, dict) and "text" in result:
            return result["text"].strip()
        else:
            return str(result).strip()
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

async def convert_text_to_speech(text: str, output_path: str, speed: float = 1.0) -> str:
    """Convert text to speech"""
    try:
        if not TTS_AVAILABLE or tts_engine is None:
            logger.warning("TTS not available, creating silent audio")
            # Create a silent audio file as fallback
            duration = len(text) * 0.1
            sample_rate = 22050
            silence = np.zeros(int(duration * sample_rate))
            
            import scipy.io.wavfile as wavfile
            wavfile.write(output_path, sample_rate, silence.astype(np.int16))
            return output_path
        
        tts_engine.setProperty('rate', int(150 * speed))
        tts_engine.save_to_file(text, output_path)
        tts_engine.runAndWait()
        return output_path
        
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

# API Endpoints

@fastapi_app.get("/")
async def root():
    """Root endpoint with memory usage info"""
    import psutil
    import os
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        return {
            "message": "ReadLess RAG API is running",
            "status": "healthy",
            "memory_usage_mb": round(memory_mb, 2),
            "memory_limit_mb": 512,
            "memory_usage_percent": round((memory_mb / 512) * 100, 2),
            "timestamp": datetime.now().isoformat(),
            "optimization": "memory_optimized_for_render"
        }
    except:
        return {
            "message": "ReadLess RAG API is running",
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "optimization": "memory_optimized_for_render"
        }

@fastapi_app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "message": "ReadLess RAG API is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.post("/session/create", response_model=SessionResponse)
async def create_session():
    """Create a new session with default settings"""
    try:
        # Generate a unique session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Validate that we have the required configuration
        if not DEFAULT_GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="Server configuration error: GROQ API key not available")
        
        if not DEFAULT_GROQ_MODEL:
            raise HTTPException(status_code=500, detail="Server configuration error: GROQ model not configured")
        
        # Initialize session if it doesn't exist
        get_session_history(session_id)
        
        return SessionResponse(
            session_id=session_id,
            message="Session created successfully",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create session: {str(e)}")

@fastapi_app.post("/session/create-custom", response_model=SessionResponse)
async def create_custom_session(request: SessionRequest):
    """Create a new session with custom API key"""
    try:
        # Validate that we have the required configuration
        api_key = request.groq_api_key or DEFAULT_GROQ_API_KEY
        if not api_key:
            raise HTTPException(status_code=400, detail="GROQ API key is required")
        
        # Initialize session if it doesn't exist
        get_session_history(request.session_id)
        
        return SessionResponse(
            session_id=request.session_id,
            message="Session created successfully",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Groq API key: {str(e)}")

@fastapi_app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    session_id: str,
    groq_api_key: Optional[str] = None,
    files: List[UploadFile] = File(...)
):
    """Upload and process PDF files - preserving original logic"""
    try:
        # Validate file types
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename or 'unknown'} is not a PDF")
        
        documents = []
        processed_files = []
        
        # Process each uploaded file - same logic as original
        for uploaded_file in files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                content = await uploaded_file.read()
                temp_file.write(content)
                temp_pdf_path = temp_file.name
            
            try:
                # Load PDF using same logic
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                documents.extend(docs)
                processed_files.append(uploaded_file.filename)
            finally:
                # Clean up temporary file
                os.unlink(temp_pdf_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be processed")
        
        # Split and create embeddings - memory optimized
        embeddings_instance = get_embeddings()
        if embeddings_instance is None:
            raise HTTPException(status_code=500, detail="Embeddings not initialized. Please check HF_TOKEN.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks to reduce memory usage
            chunk_overlap=100  # Reduced overlap
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_instance)
        
        # Configure retriever to return more relevant chunks
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}  # Retrieve top 6 most similar chunks
        )
        
        # Store retriever for session
        session_retrievers[session_id] = retriever
        
        # Use provided API key or default
        api_key = groq_api_key or DEFAULT_GROQ_API_KEY
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ API key not available")
        
        # Create RAG chain
        rag_chain = create_rag_chain(api_key, retriever, session_id)
        session_chains[session_id] = rag_chain
        
        return UploadResponse(
            session_id=session_id,
            files_processed=processed_files,
            message=f"Successfully processed {len(processed_files)} PDF files",
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@fastapi_app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG system - preserving original logic"""
    try:
        # Check if session has processed documents
        if request.session_id not in session_chains:
            raise HTTPException(
                status_code=400, 
                detail="No documents uploaded for this session. Please upload PDFs first."
            )
        
        conversational_rag_chain = session_chains[request.session_id]
        
        # Get session history
        session_history = get_session_history(request.session_id)
        
        # Invoke the chain - same logic as original
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # Format chat history for response
        chat_history = []
        for message in session_history.messages:
            chat_history.append({
                "type": message.type,
                "content": message.content
            })
        
        return AnswerResponse(
            session_id=request.session_id,
            question=request.question,
            answer=response['answer'],
            chat_history=chat_history,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@fastapi_app.get("/session/{session_id}/history")
async def get_session_history_endpoint(session_id: str):
    """Get chat history for a session"""
    try:
        session_history = get_session_history(session_id)
        
        chat_history = []
        for message in session_history.messages:
            chat_history.append({
                "type": message.type,
                "content": message.content,
                "timestamp": datetime.now().isoformat()  # In production, store actual timestamps
            })
        
        return {
            "session_id": session_id,
            "chat_history": chat_history,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@fastapi_app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data"""
    try:
        # Clean up session data
        if session_id in session_store:
            del session_store[session_id]
        if session_id in session_chains:
            del session_chains[session_id]
        if session_id in session_retrievers:
            del session_retrievers[session_id]
        
        return {
            "session_id": session_id,
            "message": "Session deleted successfully",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@fastapi_app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": list(session_store.keys()),
        "total_sessions": len(session_store),
        "status": "success"
    }

# Voice Chat Endpoints

@fastapi_app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text - fallback to browser API if Whisper unavailable"""
    # For Render deployment, recommend using browser Web Speech API
    raise HTTPException(
        status_code=503, 
        detail="Server-side speech recognition disabled for memory optimization. Please use browser Web Speech API."
    )

@fastapi_app.post("/api/text-to-speech")
async def text_to_speech(request: VoiceChatRequest):
    """Convert text to speech - fallback to browser API"""
    # For Render deployment, recommend using browser Web Speech API
    raise HTTPException(
        status_code=503,
        detail="Server-side text-to-speech disabled for memory optimization. Please use browser Web Speech API."
    )

@fastapi_app.post("/api/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    voice_speed: float = Form(1.0)
):
    """Complete voice chat: recommend using browser APIs for STT/TTS"""
    raise HTTPException(
        status_code=503,
        detail="Server-side voice chat disabled for memory optimization. Please use frontend Web Speech API for STT/TTS with /ask endpoint for text processing."
    )

@fastapi_app.get("/api/voice-chat/health")
async def voice_chat_health():
    """Health check for voice chat functionality"""
    return {
        "status": "browser_based",
        "message": "Voice chat uses browser Web Speech API for optimal memory usage",
        "server_speech": False,
        "browser_speech": True,
        "recommendation": "Use frontend Web Speech API with /ask endpoint"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, reload=True)

# Export for external use
app = fastapi_app










