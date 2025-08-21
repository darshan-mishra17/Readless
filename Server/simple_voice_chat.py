"""
Simplified Voice Chat API with Speech-to-Text and Text-to-Speech
Compatible with current environment setup
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import os
import asyncio
from typing import Optional
import logging
from datetime import datetime
import json

# Audio processing imports
import whisper
import torch

# TTS imports
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Use requests for Groq API instead of langchain to avoid compatibility issues
import requests

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize simplified voice chat app
simple_voice_app = FastAPI(
    title="Simple Voice Chat API",
    description="Real-time voice interaction with AI using STT and TTS",
    version="1.0.0"
)

# Add CORS middleware
simple_voice_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
whisper_model = None
tts_engine = None
GROQ_API_KEY = None

class VoiceChatRequest(BaseModel):
    text: str
    voice_speed: Optional[float] = 1.0

class ChatResponse(BaseModel):
    transcribed_text: str
    ai_response: str
    processing_time: float

@simple_voice_app.on_event("startup")
async def initialize_models():
    """Initialize STT and TTS models on startup"""
    global whisper_model, tts_engine, GROQ_API_KEY
    
    try:
        logger.info("🚀 Initializing simple voice chat models...")
        
        # Get Groq API key
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            logger.warning("⚠️ GROQ_API_KEY not found - Groq LLM will not work")
        else:
            logger.info("✅ Groq API key loaded")
        
        # Initialize Whisper STT model
        logger.info("📝 Loading Whisper STT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("base", device=device)
        logger.info(f"✅ Whisper model loaded on {device}")
        
        # Initialize TTS engine
        if pyttsx3:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)  # Speed of speech
            tts_engine.setProperty('volume', 0.9)  # Volume level
            logger.info("✅ TTS engine initialized")
        else:
            logger.warning("⚠️ pyttsx3 not available, TTS will not work")
        
        logger.info("🎉 Simple voice chat models initialized!")
        
    except Exception as e:
        logger.error(f"❌ Error initializing models: {e}")
        raise

async def transcribe_audio_simple(audio_file_path: str) -> str:
    """
    Transcribe audio file using Whisper STT
    """
    try:
        logger.info(f"🎤 Transcribing audio: {os.path.basename(audio_file_path)}")
        
        if not whisper_model:
            raise Exception("Whisper model not initialized")
        
        # Load and transcribe audio
        result = whisper_model.transcribe(audio_file_path)
        transcribed_text = result["text"].strip()
        
        logger.info(f"📝 Transcription: '{transcribed_text[:100]}...'")
        return transcribed_text
        
    except Exception as e:
        logger.error(f"❌ STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

async def generate_groq_response(text: str) -> str:
    """
    Generate AI response using Groq API directly
    """
    try:
        logger.info(f"🤖 Generating Groq response for: '{text[:50]}...'")
        
        if not GROQ_API_KEY:
            return "I'm sorry, but I cannot process your request right now. Please check the API configuration."
        
        # Groq API endpoint
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-70b-versatile",  # Updated to supported model
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant engaged in a voice conversation. Keep your responses natural, concise, and conversational as they will be spoken aloud. Avoid overly long responses. Be friendly and engaging."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            ai_text = result["choices"][0]["message"]["content"].strip()
            logger.info(f"💭 Groq response: '{ai_text[:100]}...'")
            return ai_text
        else:
            logger.error(f"Groq API error: {response.status_code} - {response.text}")
            return "I'm having trouble processing your request right now. Please try again."
        
    except Exception as e:
        logger.error(f"❌ Groq Error: {e}")
        return "I encountered an error while processing your request. Please try again."

async def convert_text_to_speech_simple(text: str, output_path: str, speed: float = 1.0) -> str:
    """
    Convert text to speech using TTS engine
    """
    try:
        logger.info(f"🔊 Converting text to speech: '{text[:50]}...'")
        
        if not tts_engine:
            # Create a silent audio file as fallback
            logger.warning("TTS engine not available, creating silent audio")
            import wave
            import numpy as np
            
            # Create 2 seconds of silence
            sample_rate = 22050
            duration = 2.0
            frames = int(duration * sample_rate)
            audio_data = np.zeros(frames, dtype=np.int16)
            
            with wave.open(output_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes per sample
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            return output_path
        
        # Use pyttsx3 for TTS
        tts_engine.setProperty('rate', int(150 * speed))
        tts_engine.save_to_file(text, output_path)
        tts_engine.runAndWait()
        
        logger.info(f"🎵 TTS audio saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@simple_voice_app.post("/api/voice-chat", response_model=ChatResponse)
async def simple_voice_chat_endpoint(
    audio: UploadFile = File(...),
    voice_speed: float = Form(1.0),
    session_id: str = Form("default")
):
    """
    Complete voice chat pipeline: STT → Groq LLM → TTS
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"🎙️ Starting voice chat for session: {session_id}")
        
        # Validate audio file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        try:
            # Step 1: Speech-to-Text
            transcribed_text = await transcribe_audio_simple(temp_audio_path)
            
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                raise HTTPException(status_code=400, detail="No speech detected in audio")
            
            # Step 2: Generate Groq response
            ai_response = await generate_groq_response(transcribed_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Voice chat completed in {processing_time:.2f}s")
            
            return ChatResponse(
                transcribed_text=transcribed_text,
                ai_response=ai_response,
                processing_time=processing_time
            )
            
        finally:
            # Cleanup temporary files
            try:
                os.unlink(temp_audio_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Voice chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")

@simple_voice_app.post("/api/text-to-speech")
async def simple_text_to_speech_endpoint(request: VoiceChatRequest):
    """
    Convert text to speech and return audio file
    """
    try:
        logger.info(f"🔊 TTS request: '{request.text[:50]}...'")
        
        # Create temporary file for TTS output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            output_path = temp_file.name
        
        # Generate TTS audio
        await convert_text_to_speech_simple(request.text, output_path, request.voice_speed)
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )
        
    except Exception as e:
        logger.error(f"❌ TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@simple_voice_app.post("/api/speech-to-text")
async def simple_speech_to_text_endpoint(audio: UploadFile = File(...)):
    """
    Convert speech to text using Whisper STT
    """
    try:
        logger.info(f"🎤 STT request for file: {audio.filename}")
        
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_audio_path = temp_file.name
        
        try:
            # Transcribe audio
            transcribed_text = await transcribe_audio_simple(temp_audio_path)
            return {"transcribed_text": transcribed_text}
            
        finally:
            # Cleanup
            os.unlink(temp_audio_path)
            
    except Exception as e:
        logger.error(f"❌ STT endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

@simple_voice_app.get("/api/voice-chat/health")
async def health_check():
    """Health check endpoint for voice chat service"""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "groq_api_key": GROQ_API_KEY is not None,
        "tts_loaded": tts_engine is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(simple_voice_app, host="0.0.0.0", port=8001)
