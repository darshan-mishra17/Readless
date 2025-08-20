'use client';

import React from 'react';
import VoiceChat from '@/components/VoiceChat';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MessageCircle, Mic, FileText, Settings } from 'lucide-react';

export default function VoiceChatPage() {
  const handleVoiceChatError = (error: string) => {
    console.error('Voice chat error:', error);
    // You can add toast notifications here
  };

  const handleVoiceChatSuccess = (response: any) => {
    console.log('Voice chat success:', response);
    // You can add analytics or logging here
  };

  return (
    <div className="container mx-auto py-8 space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">Talk with AI</h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Experience natural voice conversations with AI. Speak naturally and get intelligent responses.
        </p>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="voice-chat" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="voice-chat" className="flex items-center gap-2">
            <Mic className="w-4 h-4" />
            Voice Chat
          </TabsTrigger>
          <TabsTrigger value="text-chat" className="flex items-center gap-2">
            <MessageCircle className="w-4 h-4" />
            Text Chat
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="voice-chat" className="space-y-6">
          <VoiceChat
            onError={handleVoiceChatError}
            onSuccess={handleVoiceChatSuccess}
          />
        </TabsContent>

        <TabsContent value="text-chat" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageCircle className="w-5 h-5" />
                Text Chat with AI
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center text-muted-foreground py-8">
                <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Text chat integration coming soon!</p>
                <p className="text-sm mt-2">
                  This will integrate with your existing PDF chat functionality.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Voice Chat Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium mb-2">Audio Quality</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <strong>Sample Rate:</strong> 16kHz<br />
                      <strong>Channels:</strong> Mono<br />
                      <strong>Bit Depth:</strong> 16-bit
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <strong>Echo Cancellation:</strong> Enabled<br />
                      <strong>Noise Suppression:</strong> Enabled<br />
                      <strong>Auto Gain Control:</strong> Enabled
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-medium mb-2">AI Model Settings</h3>
                  <div className="p-3 bg-gray-50 rounded-lg text-sm">
                    <strong>STT Model:</strong> OpenAI Whisper (base)<br />
                    <strong>LLM Model:</strong> Mixtral-8x7B (Groq)<br />
                    <strong>TTS Engine:</strong> System TTS with pyttsx3
                  </div>
                </div>

                <div>
                  <h3 className="font-medium mb-2">Performance</h3>
                  <div className="p-3 bg-gray-50 rounded-lg text-sm">
                    <strong>Average Latency:</strong> ~2-3 seconds<br />
                    <strong>Max Audio Length:</strong> 30 seconds<br />
                    <strong>Supported Formats:</strong> WebM, WAV, MP4
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Features Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6 text-center">
            <Mic className="w-8 h-8 mx-auto mb-3 text-primary" />
            <h3 className="font-semibold mb-2">Speech Recognition</h3>
            <p className="text-sm text-muted-foreground">
              Advanced speech-to-text using OpenAI Whisper for accurate transcription
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <MessageCircle className="w-8 h-8 mx-auto mb-3 text-primary" />
            <h3 className="font-semibold mb-2">AI Conversations</h3>
            <p className="text-sm text-muted-foreground">
              Natural conversations powered by Groq's high-performance LLM models
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6 text-center">
            <Settings className="w-8 h-8 mx-auto mb-3 text-primary" />
            <h3 className="font-semibold mb-2">Voice Synthesis</h3>
            <p className="text-sm text-muted-foreground">
              High-quality text-to-speech with customizable speed and voice settings
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Technical Info */}
      <Card>
        <CardHeader>
          <CardTitle>How It Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
            <div className="p-4 border rounded-lg">
              <div className="w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-2 font-semibold">
                1
              </div>
              <h4 className="font-medium mb-1">Record</h4>
              <p className="text-xs text-muted-foreground">
                Capture your voice with optimized audio settings
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <div className="w-8 h-8 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-2 font-semibold">
                2
              </div>
              <h4 className="font-medium mb-1">Transcribe</h4>
              <p className="text-xs text-muted-foreground">
                Convert speech to text using Whisper AI
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <div className="w-8 h-8 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center mx-auto mb-2 font-semibold">
                3
              </div>
              <h4 className="font-medium mb-1">Process</h4>
              <p className="text-xs text-muted-foreground">
                Generate AI response using Groq LLM
              </p>
            </div>
            <div className="p-4 border rounded-lg">
              <div className="w-8 h-8 bg-orange-100 text-orange-600 rounded-full flex items-center justify-center mx-auto mb-2 font-semibold">
                4
              </div>
              <h4 className="font-medium mb-1">Speak</h4>
              <p className="text-xs text-muted-foreground">
                Convert response to speech and play
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
