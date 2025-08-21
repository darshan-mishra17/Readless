#!/usr/bin/env python3
"""
Startup script for ReadLess FastAPI server
"""

import uvicorn
from main import app
from config import settings

if __name__ == "__main__":
    print(f"Starting {settings.api_title} v{settings.api_version}")
    print(f"Server will be available at: http://{settings.host}:{settings.port}")
    print("API Documentation will be available at: http://localhost:8000/docs")
    print("Alternative API Documentation at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info"
    )
