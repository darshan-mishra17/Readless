# Configuration file for the FastAPI application

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "ReadLess RAG API"
    api_description: str = "Conversational RAG with PDF uploads and chat history"
    api_version: str = "1.0.0"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # CORS Configuration
    allowed_origins: list = ["*"]  # Configure for production
    allowed_methods: list = ["*"]
    allowed_headers: list = ["*"]
    
    # Groq Configuration
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"  # Best available model from Groq
    
    # HuggingFace Configuration
    hf_token: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # RAG Configuration
    chunk_size: int = 5000
    chunk_overlap: int = 500
    max_tokens: int = 3
    
    # File Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: str = ".pdf"  # Single file type as string
    
    # Session Configuration
    session_timeout: int = 3600  # 1 hour in seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()
