"""
FastAPI RAG Q&A Conversation With PDF Including Chat History
Industry-grade backend implementation preserving all original logic
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, SecretStr
from typing import List, Optional, Dict, Any
import os
import tempfile
import uuid
from datetime import datetime

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

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="ReadLess RAG API",
    description="Conversational RAG with PDF uploads and chat history",
    version="1.0.0"
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
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

# Initialize embeddings with error handling
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"Warning: Could not initialize HuggingFace embeddings: {e}")
    print("You may need to set the HF_TOKEN environment variable")
    embeddings = None

# Global storage for sessions (In production, use Redis or database)
session_store: Dict[str, ChatMessageHistory] = {}
session_chains: Dict[str, Any] = {}
session_retrievers: Dict[str, Any] = {}

# Pydantic models
class SessionRequest(BaseModel):
    session_id: str = "default_session"
    groq_api_key: str

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: str

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

# Helper functions (preserving original logic)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session history - preserving original logic"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def create_rag_chain(groq_api_key: str, retriever, session_id: str):
    """Create RAG chain - preserving original logic"""
    llm = ChatGroq(api_key=SecretStr(groq_api_key), model="Gemma2-9b-It")
    
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
    
    # Answer question prompt - same as original
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def get_session_history_func(session: str) -> BaseChatMessageHistory:
        return get_session_history(session_id)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history_func,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_rag_chain

# API Endpoints

@fastapi_app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ReadLess RAG API is running",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.post("/session/create", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new session"""
    try:
        # Validate Groq API key by creating LLM instance
        ChatGroq(api_key=SecretStr(request.groq_api_key), model="Gemma2-9b-It")
        
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
    groq_api_key: str,
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
        
        # Split and create embeddings - same logic as original
        if embeddings is None:
            raise HTTPException(status_code=500, detail="Embeddings not initialized. Please check HF_TOKEN.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, 
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        # Store retriever for session
        session_retrievers[session_id] = retriever
        
        # Create RAG chain
        rag_chain = create_rag_chain(groq_api_key, retriever, session_id)
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port, reload=True)

# Export for external use
app = fastapi_app










