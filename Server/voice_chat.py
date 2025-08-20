"""
Voice Chat API with Speech-to-Text and Text-to-Speech
Integrates with existing Groq LLM for real-time voice conversations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import tempfile
import os
import io
import wave
import asyncio
from typing import Optional
import logging
from datetime import datetime

# Audio processing imports
import whisper
import torch
import torchaudio
import numpy as np

# TTS imports
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize voice chat app
voice_app = FastAPI(
    title="Voice Chat API",
    description="Real-time voice interaction with AI using STT and TTS",
    version="1.0.0"
)

# Add CORS middleware
voice_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
whisper_model = None
groq_llm = None
tts_engine = None

class VoiceChatRequest(BaseModel):
    text: str
    voice_speed: Optional[float] = 1.0
    voice_pitch: Optional[float] = 0.0

class ChatResponse(BaseModel):
    transcribed_text: str
    ai_response: str
    audio_duration: float
    processing_time: float

@voice_app.on_event("startup")
async def initialize_models():
    """Initialize STT and TTS models on startup"""
    global whisper_model, groq_llm, tts_engine
    
    try:
        logger.info("🚀 Initializing voice chat models...")
        
        # Initialize Whisper STT model
        logger.info("📝 Loading Whisper STT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model("base", device=device)
        logger.info(f"✅ Whisper model loaded on {device}")
        
        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        groq_llm = ChatGroq(
            temperature=0.7,
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768"
        )
        logger.info("✅ Groq LLM initialized")
        
        # Initialize TTS engine
        if pyttsx3:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)  # Speed of speech
            tts_engine.setProperty('volume', 0.9)  # Volume level
            logger.info("✅ TTS engine initialized")
        else:
            logger.warning("⚠️ pyttsx3 not available, using fallback TTS")
        
        logger.info("🎉 All voice chat models initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error initializing models: {e}")
        raise

async def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe audio file using Whisper STT
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Transcribed text
    """
    try:
        logger.info(f"🎤 Transcribing audio: {os.path.basename(audio_file_path)}")
        
        # Load and transcribe audio
        result = whisper_model.transcribe(audio_file_path)
        transcribed_text = result["text"].strip()
        
        logger.info(f"📝 Transcription completed: '{transcribed_text[:100]}...'")
        return transcribed_text
        
    except Exception as e:
        logger.error(f"❌ STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

async def generate_llm_response(text: str) -> str:
    """
    Generate AI response using Groq LLM
    
    Args:
        text: User input text
        
    Returns:
        AI response text
    """
    try:
        logger.info(f"🤖 Generating LLM response for: '{text[:50]}...'")
        
        # Create a conversational prompt
        prompt = f"""You are a helpful AI assistant engaged in a voice conversation. 
        Keep your responses natural, concise, and conversational as they will be spoken aloud.
        Avoid overly long responses. Be friendly and engaging.
        
        User: {text}
        
        Assistant:"""
        
        response = await groq_llm.ainvoke(prompt)
        ai_text = response.content.strip()
        
        logger.info(f"💭 LLM response generated: '{ai_text[:100]}...'")
        return ai_text
        
    except Exception as e:
        logger.error(f"❌ LLM Error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

async def convert_text_to_speech(text: str, output_path: str, speed: float = 1.0) -> str:
    """
    Convert text to speech using TTS engine
    
    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file
        speed: Speech speed multiplier
        
    Returns:
        Path to the generated audio file
    """
    try:
        logger.info(f"🔊 Converting text to speech: '{text[:50]}...'")
        
        if tts_engine:
            # Use pyttsx3 for TTS
            tts_engine.setProperty('rate', int(150 * speed))
            tts_engine.save_to_file(text, output_path)
            tts_engine.runAndWait()
        else:
            # Fallback: Create a simple beep or use system TTS
            logger.warning("Using fallback TTS (no audio generation)")
            # Create a silent audio file as fallback
            duration = len(text) * 0.1  # Estimate duration
            sample_rate = 22050
            silence = np.zeros(int(duration * sample_rate))
            
            # Save as WAV file
            import scipy.io.wavfile as wavfile
            wavfile.write(output_path, sample_rate, silence.astype(np.int16))
        
        logger.info(f"🎵 TTS audio saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"❌ TTS Error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@voice_app.post("/api/voice-chat", response_model=ChatResponse)
async def voice_chat_endpoint(
    audio: UploadFile = File(...),
    voice_speed: float = Form(1.0),
    session_id: str = Form("default")
):
    """
    Complete voice chat pipeline: STT → LLM → TTS
    
    Args:
        audio: Audio file from user
        voice_speed: TTS speed multiplier
        session_id: Session identifier for conversation context
        
    Returns:
        ChatResponse with transcription, AI response, and timing info
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"🎙️ Starting voice chat pipeline for session: {session_id}")
        
        # Validate audio file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        logger.info(f"📁 Audio saved to temp file: {temp_audio_path}")
        
        try:
            # Step 1: Speech-to-Text
            transcribed_text = await transcribe_audio(temp_audio_path)
            
            if not transcribed_text or len(transcribed_text.strip()) < 2:
                raise HTTPException(status_code=400, detail="No speech detected in audio")
            
            # Step 2: Generate LLM response
            ai_response = await generate_llm_response(transcribed_text)
            
            # Step 3: Text-to-Speech
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_tts:
                tts_output_path = temp_tts.name
            
            await convert_text_to_speech(ai_response, tts_output_path, voice_speed)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Estimate audio duration (rough approximation)
            audio_duration = len(ai_response) * 0.08  # ~80ms per character
            
            logger.info(f"✅ Voice chat pipeline completed in {processing_time:.2f}s")
            
            return ChatResponse(
                transcribed_text=transcribed_text,
                ai_response=ai_response,
                audio_duration=audio_duration,
                processing_time=processing_time
            )
            
        finally:
            # Cleanup temporary files
            try:
                os.unlink(temp_audio_path)
                if 'tts_output_path' in locals():
                    # Keep TTS file for download endpoint
                    pass
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Voice chat pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice chat failed: {str(e)}")

@voice_app.post("/api/text-to-speech")
async def text_to_speech_endpoint(request: VoiceChatRequest):
    """
    Convert text to speech and return audio file
    
    Args:
        request: VoiceChatRequest with text and voice settings
        
    Returns:
        Audio file response
    """
    try:
        logger.info(f"🔊 TTS request: '{request.text[:50]}...'")
        
        # Create temporary file for TTS output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            output_path = temp_file.name
        
        # Generate TTS audio
        await convert_text_to_speech(request.text, output_path, request.voice_speed)
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
        )
        
    except Exception as e:
        logger.error(f"❌ TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@voice_app.post("/api/speech-to-text")
async def speech_to_text_endpoint(audio: UploadFile = File(...)):
    """
    Convert speech to text using Whisper STT
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        Transcribed text
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
            transcribed_text = await transcribe_audio(temp_audio_path)
            
            return {"transcribed_text": transcribed_text}
            
        finally:
            # Cleanup
            os.unlink(temp_audio_path)
            
    except Exception as e:
        logger.error(f"❌ STT endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")

@voice_app.get("/api/voice-chat/health")
async def health_check():
    """Health check endpoint for voice chat service"""
    return {
        "status": "healthy",
        "whisper_loaded": whisper_model is not None,
        "groq_loaded": groq_llm is not None,
        "tts_loaded": tts_engine is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(voice_app, host="0.0.0.0", port=8001)
