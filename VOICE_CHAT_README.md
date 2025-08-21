# 🎙️ Voice Chat with AI - Complete Integration Guide

This is a complete real-time voice interaction system that adds **"Talk with AI"** functionality to your existing PDF chatbot project.

## 🚀 Features

- **Speech-to-Text (STT)**: OpenAI Whisper for accurate transcription
- **AI Conversations**: Groq LLM for intelligent responses  
- **Text-to-Speech (TTS)**: High-quality voice synthesis
- **Real-time Processing**: Low-latency voice interactions
- **Web Integration**: Full frontend and backend implementation
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📋 System Requirements

- Python 3.8+
- Node.js 16+ (for frontend)
- Microphone access
- Internet connection (for Groq API)

## 🛠️ Installation & Setup

### Backend Setup

1. **Install Python Dependencies**:
```bash
pip install openai-whisper torch torchaudio pyttsx3 scipy fastapi uvicorn python-multipart requests python-dotenv
```

2. **Configure Environment Variables**:
Create a `.env` file in the `Server` directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

3. **Test Installation**:
```bash
cd Server
python test_voice_chat.py
```

### Frontend Setup

1. **Install Required UI Components** (if not already installed):
```bash
cd Frontend
npm install @radix-ui/react-slider lucide-react
```

2. **Add Voice Chat Route**:
The voice chat page is already created at `/voice-chat`

## 🎯 Usage

### Starting the Voice Chat Server

```bash
cd Server
python simple_voice_chat.py
```

The server will start on `http://localhost:8001`

### Available API Endpoints

- `POST /api/voice-chat` - Complete voice chat pipeline
- `POST /api/speech-to-text` - STT only
- `POST /api/text-to-speech` - TTS only
- `GET /api/voice-chat/health` - Health check

### Frontend Integration

The voice chat is available at `http://localhost:3000/voice-chat`

## 🔧 How It Works

### Complete Voice Chat Flow

1. **Record Audio**: User speaks into microphone
2. **Speech-to-Text**: Audio is transcribed using Whisper
3. **AI Processing**: Text is sent to Groq LLM for response
4. **Text-to-Speech**: AI response is converted to audio
5. **Playback**: Audio response is played to user

### Backend Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Frontend      │───▶│   FastAPI    │───▶│   Whisper   │
│   (React)       │    │   Server     │    │    STT      │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
                       ┌─────────────┐    ┌─────────────┐
                       │    Groq     │───▶│   pyttsx3   │
                       │    LLM      │    │    TTS      │
                       └─────────────┘    └─────────────┘
```

### Frontend Components

- **VoiceChat.tsx**: Main voice interaction component
- **audio-utils.ts**: Audio processing utilities
- **page.tsx**: Voice chat page with tabs and settings

## 🎛️ Configuration Options

### Voice Settings

- **Voice Speed**: 0.5x to 2.0x (configurable via UI)
- **Volume**: 0% to 100% (configurable via UI)
- **Audio Quality**: 16kHz, 16-bit, Mono
- **Noise Reduction**: Enabled by default

### AI Model Settings

- **STT Model**: Whisper base (can be upgraded to large)
- **LLM Model**: Llama-3.1-70B-Versatile (Groq)
- **TTS Engine**: System TTS with pyttsx3
- **Response Length**: Max 1000 tokens

## 🔍 Testing

### Backend Testing

```bash
cd Server
python test_voice_chat.py
```

This will test:
- Model initialization
- STT functionality  
- LLM integration
- TTS generation
- API endpoints

### Frontend Testing

1. Start the voice chat server
2. Navigate to `/voice-chat` in your browser
3. Allow microphone permissions
4. Click the microphone button and speak
5. Listen to the AI response

## 🛡️ Error Handling

### Common Issues & Solutions

1. **Microphone Access Denied**:
   - Enable microphone permissions in browser
   - Check system microphone settings

2. **Groq API Errors**:
   - Verify API key in `.env` file
   - Check internet connection
   - Ensure sufficient API credits

3. **TTS Not Working**:
   - Windows: Ensure Windows Speech Platform is installed
   - macOS/Linux: Install system TTS dependencies

4. **Audio Quality Issues**:
   - Use a good quality microphone
   - Reduce background noise
   - Speak clearly and at normal pace

## 📊 Performance Metrics

### Typical Performance

- **STT Latency**: ~1-2 seconds
- **LLM Response**: ~1-2 seconds  
- **TTS Generation**: ~0.5-1 seconds
- **Total Pipeline**: ~3-5 seconds
- **Audio Quality**: 16kHz, 16-bit

### Optimization Tips

1. **GPU Acceleration**: Install CUDA for faster Whisper processing
2. **Model Selection**: Use Whisper "small" for faster STT
3. **Response Length**: Keep AI responses concise for faster TTS
4. **Caching**: Implement response caching for common queries

## 🔗 Integration with Existing PDF Chat

The voice chat system is designed to complement your existing PDF chatbot:

1. **Shared Backend**: Can use the same FastAPI server
2. **Session Management**: Supports session-based conversations
3. **Context Switching**: Easy to switch between voice and text chat
4. **Unified UI**: Integrated into the same frontend application

## 📱 Browser Compatibility

- ✅ Chrome 60+
- ✅ Firefox 55+
- ✅ Safari 11+
- ✅ Edge 79+

## 🔒 Security Considerations

- Audio data is processed temporarily and not stored
- API keys are server-side only
- CORS is configured for development (update for production)
- User consent required for microphone access

## 🎨 Customization

### Styling

The components use Tailwind CSS and can be customized:
- Modify colors in `VoiceChat.tsx`
- Adjust layout in `page.tsx`
- Update themes via CSS variables

### Functionality

- **Add Voice Commands**: Extend the LLM prompt for commands
- **Multi-language**: Configure Whisper for other languages
- **Voice Selection**: Add multiple TTS voices
- **Conversation Memory**: Implement chat history

## 📚 API Documentation

### Voice Chat Endpoint

```http
POST /api/voice-chat
Content-Type: multipart/form-data

audio: File (audio file)
voice_speed: float (0.5-2.0)
session_id: string
```

Response:
```json
{
  "transcribed_text": "Hello, how are you?",
  "ai_response": "I'm doing great, thank you!",
  "processing_time": 2.34
}
```

### Health Check

```http
GET /api/voice-chat/health
```

Response:
```json
{
  "status": "healthy",
  "whisper_loaded": true,
  "groq_api_key": true,
  "tts_loaded": true,
  "timestamp": "2025-01-20T10:30:00"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is part of your ReadLess application. See the main project license.

## 🆘 Support

For issues and questions:
1. Check the error logs in terminal
2. Verify all dependencies are installed
3. Test with the provided test scripts
4. Check browser console for frontend errors

---

**🎉 Congratulations!** You now have a complete voice chat system integrated with your PDF chatbot. Users can seamlessly switch between text and voice interactions for a truly modern AI experience!
