# Utility functions for the FastAPI application

import os
import uuid
import asyncio
from typing import List, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionManager:
    """Manages session lifecycle and cleanup"""
    
    def __init__(self, timeout_seconds: int = 3600):
        self.timeout_seconds = timeout_seconds
        self.session_timestamps = {}
    
    def touch_session(self, session_id: str):
        """Update session last access time"""
        self.session_timestamps[session_id] = datetime.now()
    
    def is_session_expired(self, session_id: str) -> bool:
        """Check if session has expired"""
        if session_id not in self.session_timestamps:
            return True
        
        last_access = self.session_timestamps[session_id]
        expiry_time = last_access + timedelta(seconds=self.timeout_seconds)
        return datetime.now() > expiry_time
    
    def cleanup_expired_sessions(self, session_store, session_chains, session_retrievers):
        """Clean up expired sessions"""
        expired_sessions = []
        
        for session_id in list(self.session_timestamps.keys()):
            if self.is_session_expired(session_id):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            # Clean up session data
            if session_id in session_store:
                del session_store[session_id]
            if session_id in session_chains:
                del session_chains[session_id]
            if session_id in session_retrievers:
                del session_retrievers[session_id]
            if session_id in self.session_timestamps:
                del self.session_timestamps[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return expired_sessions

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate if file type is allowed"""
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in allowed_types

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    i = 0
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f}{size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace unsafe characters
    unsafe_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    return filename

async def cleanup_temp_file(file_path: str, delay: int = 60):
    """Clean up temporary file after delay"""
    await asyncio.sleep(delay)
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {file_path}: {e}")

def log_api_call(endpoint: str, session_id: str, user_ip: Optional[str] = None):
    """Log API call for monitoring"""
    timestamp = datetime.now().isoformat()
    logger.info(f"API Call - Endpoint: {endpoint}, Session: {session_id}, IP: {user_ip}, Time: {timestamp}")
