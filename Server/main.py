"""
FastAPI RAG Q&A Conversation With PDF Including Chat History
Industry-grade backend implementation preserving all original logic
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from typing import List, Dict, Any
import os
import tempfile
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

# Get configuration
from config import settings

# Initialize FastAPI app
app = FastAPI(
    title="ReadLess RAG API",
    description="Conversational RAG with PDF uploads and chat history",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings with error handling - lazy loading
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ['HF_TOKEN'] = hf_token

# Don't initialize embeddings immediately - do it lazily
embeddings = None

def get_embeddings():
    """Lazy loading of embeddings"""
    global embeddings
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not initialize HuggingFace embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Embeddings initialization failed: {e}")
    return embeddings

# Global storage for sessions
session_store: Dict[str, ChatMessageHistory] = {}
session_chains: Dict[str, Any] = {}
session_retrievers: Dict[str, Any] = {}

# Pydantic models
class SessionRequest(BaseModel):
    session_id: str = "default_session"
    groq_api_key: str

class SimpleSessionRequest(BaseModel):
    session_id: str = "default_session"

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    groq_api_key: str

class SimpleQuestionRequest(BaseModel):
    session_id: str
    question: str

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

# Simplified models for using environment API key
class SimpleSessionRequest(BaseModel):
    session_id: str = "default_session"

class SimpleQuestionRequest(BaseModel):
    session_id: str
    question: str

# Helper functions (preserving original logic)
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session history - preserving original logic"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def create_rag_chain(groq_api_key: str, retriever, session_id: str):
    """Create RAG chain with strict PDF-only responses and chunk logging"""
    llm = ChatGroq(api_key=SecretStr(groq_api_key), model=settings.groq_model)
    
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
    
    # STRICT PDF-ONLY Answer prompt - prevent hallucination with clean formatting
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "IMPORTANT: Answer STRICTLY from the uploaded PDF content provided in the context below. "
        "DO NOT use any external knowledge or information not present in the provided context. "
        "If the answer is not found in the provided context, reply EXACTLY: "
        "'The document does not contain this information.' "
        "Use the following pieces of retrieved context to answer the question. "
        "Provide clear, well-formatted responses without citation markers or reference numbers. "
        "Keep your answer concise and cite only what is explicitly mentioned in the context. "
        "Format your response in a clean, readable manner with proper paragraphs and bullet points when appropriate."
        "\n\n"
        "CONTEXT FROM PDF:\n{context}"
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ReadLess RAG API is running",
        "status": "healthy",
        "model": settings.groq_model,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for frontend"""
    return {
        "status": "healthy",
        "message": "ReadLess RAG API is running",
        "model": settings.groq_model,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/session/create", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new session with custom API key"""
    try:
        # Validate Groq API key by creating LLM instance
        ChatGroq(api_key=SecretStr(request.groq_api_key), model=settings.groq_model)
        
        # Initialize session if it doesn't exist
        get_session_history(request.session_id)
        
        return SessionResponse(
            session_id=request.session_id,
            message="Session created successfully",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Groq API key: {str(e)}")

@app.post("/session/create-simple", response_model=SessionResponse)
async def create_session_simple(request: SimpleSessionRequest):
    """Create a new session using environment API key"""
    try:
        if not settings.groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured in environment variables")
        
        # Validate Groq API key by creating LLM instance
        ChatGroq(api_key=SecretStr(settings.groq_api_key), model=settings.groq_model)
        
        # Initialize session if it doesn't exist
        get_session_history(request.session_id)
        
        return SessionResponse(
            session_id=request.session_id,
            message=f"Session created successfully with model {settings.groq_model}",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating session: {str(e)}")

@app.post("/upload-simple", response_model=UploadResponse)
async def upload_pdfs_simple(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload and process PDF files using environment API key"""
    try:
        if not settings.groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured in environment variables")
        
        # Validate file types
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename or 'unknown'} is not a PDF")
        
        # Get embeddings (lazy loading)
        embeddings = get_embeddings()
        
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
                
                # Validate that we extracted text from the PDF
                if not docs:
                    print(f"WARNING: No pages extracted from {uploaded_file.filename}")
                    continue
                
                # Check if any of the documents have actual text content
                has_content = False
                for doc in docs:
                    if doc.page_content and doc.page_content.strip():
                        has_content = True
                        break
                
                if not has_content:
                    print(f"WARNING: No text content found in {uploaded_file.filename}")
                    continue
                
                documents.extend(docs)
                processed_files.append(uploaded_file.filename)
                print(f"Successfully processed {uploaded_file.filename} - extracted {len(docs)} pages")
            finally:
                # Clean up temporary file
                os.unlink(temp_pdf_path)
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents could be processed")
        
        # Split and create embeddings - same logic as original
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        # Filter out empty splits and validate content
        valid_splits = [split for split in splits if split.page_content.strip()]
        
        if not valid_splits:
            raise HTTPException(
                status_code=400, 
                detail="No valid text content found in the uploaded PDF(s). The documents may be image-only or contain no readable text."
            )
        
        vectorstore = Chroma.from_documents(documents=valid_splits, embedding=embeddings)
        
        # Configure retriever with more lenient settings for better recall
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 8,  # Return top 8 most relevant chunks
            }
        )
        
        print(f"✅ Created vectorstore with {len(valid_splits)} chunks and configured retriever")
        
        # Store retriever for session
        session_retrievers[session_id] = retriever
        
        # Create RAG chain using environment API key
        rag_chain = create_rag_chain(settings.groq_api_key, retriever, session_id)
        session_chains[session_id] = rag_chain
        
        return UploadResponse(
            session_id=session_id,
            files_processed=processed_files,
            message=f"Successfully processed {len(processed_files)} PDF files with {settings.groq_model}",
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    session_id: str = Form(...),
    groq_api_key: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload and process PDF files - preserving original logic"""
    try:
        # Validate file types
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename or 'unknown'} is not a PDF")
        
        # Get embeddings (lazy loading)
        embeddings = get_embeddings()
        
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
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

@app.post("/ask-simple", response_model=AnswerResponse)
async def ask_question_simple(request: SimpleQuestionRequest):
    """Ask a question using environment API key with strict PDF-only responses"""
    try:
        if not settings.groq_api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured in environment variables")
        
        # Check if session has processed documents
        if request.session_id not in session_chains:
            raise HTTPException(
                status_code=400, 
                detail="No documents uploaded for this session. Please upload PDFs first."
            )
        
        conversational_rag_chain = session_chains[request.session_id]
        retriever = session_retrievers[request.session_id]
        
        # STEP 1: Test retrieval before invoking the full chain
        print(f"🔍 RETRIEVING CHUNKS for question: '{request.question}'")
        try:
            # Get relevant chunks from the PDF
            retrieved_docs = retriever.get_relevant_documents(request.question)
            
            print(f"📄 Retrieved {len(retrieved_docs)} chunks from PDF:")
            for i, doc in enumerate(retrieved_docs):
                print(f"   Chunk {i+1}: {doc.page_content[:150]}...")
            
            # More lenient check - only fail if absolutely no chunks or all empty
            if not retrieved_docs:
                print("⚠️  No chunks retrieved from PDF")
                return AnswerResponse(
                    session_id=request.session_id,
                    question=request.question,
                    answer="The document does not contain this information.",
                    chat_history=[],
                    status="success"
                )
                
        except Exception as retrieval_error:
            print(f"❌ Retrieval error: {retrieval_error}")
            # Don't immediately fail - let the chain try anyway
            print("🔄 Continuing with chain invocation despite retrieval error...")
        
        # STEP 2: Get session history
        session_history = get_session_history(request.session_id)
        
        # STEP 3: Invoke the chain with verified chunks
        print("🤖 Invoking LLM with retrieved context...")
        response = conversational_rag_chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        print(f"✅ LLM Response: {response['answer'][:100]}...")
        
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

@app.post("/ask", response_model=AnswerResponse)
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

@app.get("/session/{session_id}/history")
async def get_session_history_endpoint(session_id: str):
    """Get chat history for a session"""
    try:
        session_history = get_session_history(session_id)
        
        chat_history = []
        for message in session_history.messages:
            chat_history.append({
                "type": message.type,
                "content": message.content,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "session_id": session_id,
            "chat_history": chat_history,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.delete("/session/{session_id}")
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

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": list(session_store.keys()),
        "total_sessions": len(session_store),
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
