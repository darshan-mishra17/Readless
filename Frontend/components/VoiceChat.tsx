/**
 * Voice Chat Component - Real-time voice interaction with AI
 * Handles audio recording, STT, LLM communication, and TTS playback
 */

'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Play, 
  Pause, 
  Settings,
  MessageCircle,
  Loader2,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

// Types
interface VoiceChatResponse {
  transcribed_text: string;
  ai_response: string;
  audio_duration: number;
  processing_time: number;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  text: string;
  timestamp: Date;
  audio?: Blob;
}

interface VoiceChatProps {
  groqApiKey?: string;
  onError?: (error: string) => void;
  onSuccess?: (response: VoiceChatResponse) => void;
}

export default function VoiceChat({ groqApiKey, onError, onSuccess }: VoiceChatProps) {
  // State management
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [voiceSpeed, setVoiceSpeed] = useState(1.0);
  const [volume, setVolume] = useState(0.8);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  
  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // API Configuration
  const API_BASE_URL = 'http://localhost:8001/api';

  /**
   * Initialize audio context and check microphone permissions
   */
  const initializeAudio = useCallback(async () => {
    try {
      setConnectionStatus('connecting');
      
      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });
      
      streamRef.current = stream;
      setConnectionStatus('connected');
      setError(null);
      
      return stream;
    } catch (err) {
      const errorMessage = 'Microphone access denied. Please enable microphone permissions.';
      setError(errorMessage);
      setConnectionStatus('disconnected');
      onError?.(errorMessage);
      throw err;
    }
  }, [onError]);

  /**
   * Start audio recording
   */
  const startRecording = useCallback(async () => {
    try {
      const stream = streamRef.current || await initializeAudio();
      
      // Clear previous audio chunks
      audioChunksRef.current = [];
      
      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      
      // Handle data available
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      // Handle recording stop
      mediaRecorder.onstop = async () => {
        await processRecording();
      };
      
      // Start recording
      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
      setError(null);
      
    } catch (err) {
      console.error('Failed to start recording:', err);
      setError('Failed to start recording. Please check microphone permissions.');
    }
  }, [initializeAudio]);

  /**
   * Stop audio recording
   */
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  /**
   * Process recorded audio through the voice chat pipeline
   */
  const processRecording = useCallback(async () => {
    if (audioChunksRef.current.length === 0) {
      setError('No audio recorded');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Create audio blob
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      
      // Convert to WAV format for better compatibility
      const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
      
      // Create form data
      const formData = new FormData();
      formData.append('audio', audioFile);
      formData.append('voice_speed', voiceSpeed.toString());
      formData.append('session_id', 'voice-chat-session');

      // Send to voice chat API
      const response = await fetch(`${API_BASE_URL}/voice-chat`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Voice chat API error');
      }

      const result: VoiceChatResponse = await response.json();
      
      // Add user message
      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        type: 'user',
        text: result.transcribed_text,
        timestamp: new Date(),
        audio: audioBlob
      };

      // Get AI audio response
      const ttsResponse = await fetch(`${API_BASE_URL}/text-to-speech`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: result.ai_response,
          voice_speed: voiceSpeed,
          voice_pitch: 0.0
        }),
      });

      let aiAudioBlob: Blob | undefined;
      if (ttsResponse.ok) {
        aiAudioBlob = await ttsResponse.blob();
      }

      // Add AI message
      const aiMessage: ChatMessage = {
        id: `ai-${Date.now()}`,
        type: 'ai',
        text: result.ai_response,
        timestamp: new Date(),
        audio: aiAudioBlob
      };

      // Update messages
      setMessages(prev => [...prev, userMessage, aiMessage]);

      // Auto-play AI response
      if (aiAudioBlob) {
        await playAudio(aiAudioBlob);
      }

      // Callback
      onSuccess?.(result);

    } catch (err) {
      console.error('Voice chat error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Voice chat processing failed';
      setError(errorMessage);
      onError?.(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  }, [voiceSpeed, onSuccess, onError]);

  /**
   * Play audio blob
   */
  const playAudio = useCallback(async (audioBlob: Blob): Promise<void> => {
    return new Promise((resolve, reject) => {
      try {
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audioElementRef.current = audio;
        
        audio.volume = volume;
        audio.onended = () => {
          setIsPlaying(false);
          URL.revokeObjectURL(audioUrl);
          resolve();
        };
        
        audio.onerror = () => {
          setIsPlaying(false);
          URL.revokeObjectURL(audioUrl);
          reject(new Error('Audio playback failed'));
        };
        
        setIsPlaying(true);
        audio.play();
        
      } catch (err) {
        reject(err);
      }
    });
  }, [volume]);

  /**
   * Stop audio playback
   */
  const stopAudio = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  }, []);

  /**
   * Check API health on mount
   */
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/voice-chat/health`);
        if (response.ok) {
          setConnectionStatus('connected');
        } else {
          setConnectionStatus('disconnected');
        }
      } catch {
        setConnectionStatus('disconnected');
      }
    };

    checkApiHealth();
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioElementRef.current) {
        audioElementRef.current.pause();
      }
    };
  }, []);

  /**
   * Render connection status
   */
  const renderConnectionStatus = () => {
    const statusConfig = {
      connected: { color: 'bg-green-500', text: 'Connected', icon: CheckCircle },
      connecting: { color: 'bg-yellow-500', text: 'Connecting', icon: Loader2 },
      disconnected: { color: 'bg-red-500', text: 'Disconnected', icon: AlertCircle }
    };

    const config = statusConfig[connectionStatus];
    const Icon = config.icon;

    return (
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${config.color}`} />
        <Icon className={`w-4 h-4 ${connectionStatus === 'connecting' ? 'animate-spin' : ''}`} />
        <span className="text-sm text-gray-600">{config.text}</span>
      </div>
    );
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <MessageCircle className="w-5 h-5" />
            Voice Chat with AI
          </CardTitle>
          {renderConnectionStatus()}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Voice Controls */}
        <div className="flex items-center justify-center gap-4">
          <Button
            size="lg"
            variant={isRecording ? "destructive" : "default"}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isProcessing || connectionStatus === 'disconnected'}
            className="h-16 w-16 rounded-full"
          >
            {isRecording ? (
              <MicOff className="w-6 h-6" />
            ) : (
              <Mic className="w-6 h-6" />
            )}
          </Button>

          {isPlaying && (
            <Button
              size="lg"
              variant="outline"
              onClick={stopAudio}
              className="h-16 w-16 rounded-full"
            >
              <Pause className="w-6 h-6" />
            </Button>
          )}

          {isProcessing && (
            <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Processing...</span>
            </div>
          )}
        </div>

        {/* Voice Settings */}
        <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            <span className="font-medium">Voice Settings</span>
          </div>
          
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Voice Speed: {voiceSpeed}x</label>
              <Slider
                value={[voiceSpeed]}
                onValueChange={([value]) => setVoiceSpeed(value)}
                min={0.5}
                max={2.0}
                step={0.1}
                className="mt-2"
              />
            </div>
            
            <div>
              <label className="text-sm font-medium">Volume: {Math.round(volume * 100)}%</label>
              <Slider
                value={[volume]}
                onValueChange={([value]) => setVolume(value)}
                min={0}
                max={1}
                step={0.1}
                className="mt-2"
              />
            </div>
          </div>
        </div>

        {/* Chat History */}
        <div className="space-y-3 max-h-96 overflow-y-auto">
          <h3 className="font-medium">Conversation</h3>
          
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <Mic className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>Click the microphone to start talking</p>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.type === 'ai' ? 'justify-start' : 'justify-end'}`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.type === 'ai'
                      ? 'bg-blue-100 text-blue-900'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant={message.type === 'ai' ? 'default' : 'secondary'}>
                      {message.type === 'ai' ? 'AI' : 'You'}
                    </Badge>
                    <span className="text-xs text-gray-500">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                    {message.audio && (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => playAudio(message.audio!)}
                        disabled={isPlaying}
                      >
                        {isPlaying ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                      </Button>
                    )}
                  </div>
                  <p className="text-sm">{message.text}</p>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Instructions */}
        <div className="text-xs text-gray-500 space-y-1">
          <p>• Click and hold the microphone button to record your voice</p>
          <p>• Release to process and get AI response</p>
          <p>• AI responses will play automatically</p>
          <p>• Adjust voice speed and volume in settings</p>
        </div>
      </CardContent>
    </Card>
  );
}
