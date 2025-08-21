"use client"

import type React from "react"
import Link from "next/link"
import ReactMarkdown from "react-markdown"

// Type declarations for Web Speech API
declare global {
  interface Window {
    SpeechRecognition: any
    webkitSpeechRecognition: any
  }
}

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"
import { Upload, Send, Moon, Sun, FileText, X, Mic, MicOff, Volume2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { createSession, uploadPDF, askQuestion, ApiError } from "@/lib/api"
import { NavbarServerStatus } from "@/components/navbar-server-status"

interface Message {
  id: string
  content: string
  isUser: boolean
  timestamp: Date
}

export default function PDFChatbot() {
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isSessionCreated, setIsSessionCreated] = useState(false)
  
  // Voice chat states
  const [isListening, setIsListening] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [speechRecognition, setSpeechRecognition] = useState<any>(null)
  const [speechSynthesis, setSpeechSynthesis] = useState<SpeechSynthesis | null>(null)
  const [voiceSupported, setVoiceSupported] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode)
  }, [isDarkMode])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Initialize voice functionality
  useEffect(() => {
    const initVoice = () => {
      try {
        // Check if Web Speech API is supported
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
        const SpeechSynthesis = window.speechSynthesis

        if (SpeechRecognition && SpeechSynthesis) {
          const recognition = new SpeechRecognition()
          recognition.continuous = false
          recognition.interimResults = false
          recognition.lang = 'en-US'

          setSpeechRecognition(recognition)
          setSpeechSynthesis(SpeechSynthesis)
          setVoiceSupported(true)
        } else {
          setVoiceSupported(false)
        }
      } catch (error) {
        console.warn("Voice functionality not supported:", error)
        setVoiceSupported(false)
      }
    }

    initVoice()
  }, [])

  // Initialize session on component mount
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await createSession()
        setSessionId(response.session_id)
        setIsSessionCreated(true)
      } catch (error) {
        toast({
          title: "Connection Error",
          description: "Unable to connect to server. Please check if the backend is running.",
          variant: "destructive",
        })
      }
    }

    if (!isSessionCreated) {
      initSession()
    }
  }, [isSessionCreated, toast])

  const handleFileUpload = async (file: File) => {
    if (file.type !== "application/pdf") {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF file only.",
        variant: "destructive",
      })
      return
    }

    if (!sessionId) {
      toast({
        title: "Session not ready",
        description: "Please wait for session to be created.",
        variant: "destructive",
      })
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    try {
      // Simulate upload progress UI
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 100)

      const response = await uploadPDF(file, sessionId)
      
      clearInterval(progressInterval)
      setUploadProgress(100)
      setUploadedFile(file)
      
      toast({
        title: "Upload successful",
        description: `${file.name} has been processed (${response.pages} pages)`,
      })
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof ApiError ? error.message : "Upload failed. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const sendMessage = async () => {
    if (!inputMessage.trim() || !uploadedFile || !sessionId) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      isUser: true,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const currentMessage = inputMessage
    setInputMessage("")
    setIsTyping(true)

    try {
      const response = await askQuestion(currentMessage, sessionId)
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        isUser: false,
        timestamp: new Date(),
      }
      
      setMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Sorry, I encountered an error: ${error instanceof ApiError ? error.message : 'Please try again.'}`,
        isUser: false,
        timestamp: new Date(),
      }
      
      setMessages((prev) => [...prev, errorMessage])
      
      toast({
        title: "Message failed",
        description: error instanceof ApiError ? error.message : "Unable to send message. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsTyping(false)
    }
  }

  const clearPDF = async () => {
    setUploadedFile(null)
    setMessages([])
    setUploadProgress(0)
    
    // Create a new session for fresh start
    try {
      const response = await createSession()
      setSessionId(response.session_id)
      toast({
        title: "Reset successful",
        description: "Ready for a new PDF and conversation!",
      })
    } catch (error) {
      toast({
        title: "Reset failed",
        description: "Unable to create new session. Please refresh the page.",
        variant: "destructive",
      })
    }
  }

  // Voice chat functions
  const startListening = () => {
    if (!speechRecognition || !voiceSupported || !uploadedFile) {
      toast({
        title: "Voice not available",
        description: !uploadedFile ? "Please upload a PDF first" : "Voice recognition not supported in this browser",
        variant: "destructive",
      })
      return
    }

    setIsListening(true)
    
    speechRecognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript
      setInputMessage(transcript)
      
      // Automatically send the transcribed message
      if (transcript.trim()) {
        sendVoiceMessage(transcript)
      }
    }

    speechRecognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error)
      setIsListening(false)
      toast({
        title: "Voice recognition error",
        description: "Please try again",
        variant: "destructive",
      })
    }

    speechRecognition.onend = () => {
      setIsListening(false)
    }

    try {
      speechRecognition.start()
    } catch (error) {
      console.error('Error starting speech recognition:', error)
      setIsListening(false)
      toast({
        title: "Voice recognition error",
        description: "Could not start voice recognition",
        variant: "destructive",
      })
    }
  }

  const stopListening = () => {
    if (speechRecognition) {
      speechRecognition.stop()
    }
    setIsListening(false)
  }

  const sendVoiceMessage = async (transcript: string) => {
    if (!transcript.trim() || !uploadedFile || !sessionId) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: transcript,
      isUser: true,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsTyping(true)

    try {
      const response = await askQuestion(transcript, sessionId)
      
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        isUser: false,
        timestamp: new Date(),
      }
      
      setMessages((prev) => [...prev, aiMessage])
      
      // Speak the AI response
      speakText(response.answer)
      
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Sorry, I encountered an error: ${error instanceof ApiError ? error.message : 'Please try again.'}`,
        isUser: false,
        timestamp: new Date(),
      }
      
      setMessages((prev) => [...prev, errorMessage])
      
      toast({
        title: "Message failed",
        description: error instanceof ApiError ? error.message : "Unable to send message. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsTyping(false)
      setInputMessage("")
    }
  }

  const speakText = (text: string) => {
    if (!speechSynthesis || !voiceSupported) return

    // Stop any current speech
    speechSynthesis.cancel()

    // Clean markdown syntax for TTS
    const cleanText = cleanMarkdownForTTS(text)

    const utterance = new SpeechSynthesisUtterance(cleanText)
    utterance.rate = 0.9
    utterance.pitch = 1
    utterance.volume = 0.8

    utterance.onstart = () => setIsSpeaking(true)
    utterance.onend = () => setIsSpeaking(false)
    utterance.onerror = () => setIsSpeaking(false)

    setIsSpeaking(true)
    speechSynthesis.speak(utterance)
  }

  const cleanMarkdownForTTS = (text: string): string => {
    return text
      // Remove headers (##, ###, etc.)
      .replace(/^#{1,6}\s+/gm, '')
      // Remove bold/italic markers
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/__([^_]+)__/g, '$1')
      .replace(/_([^_]+)_/g, '$1')
      // Remove code blocks
      .replace(/```[^`]*```/g, '')
      .replace(/`([^`]+)`/g, '$1')
      // Remove links
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      // Remove bullet points
      .replace(/^[\s]*[-*+]\s+/gm, '')
      // Remove numbered lists
      .replace(/^[\s]*\d+\.\s+/gm, '')
      // Remove blockquotes
      .replace(/^>\s+/gm, '')
      // Clean up extra whitespace
      .replace(/\n\s*\n/g, '\n')
      .replace(/\s+/g, ' ')
      .trim()
  }

  const stopSpeaking = () => {
    if (speechSynthesis) {
      speechSynthesis.cancel()
    }
    setIsSpeaking(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#E0EAF3] to-[#cfdef3] dark:from-gray-900 dark:to-gray-800">
      {/* Navbar */}
      <nav className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-b border-white/20 dark:border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <FileText className="h-8 w-8 text-gray-700 dark:text-white" />
              <span className="text-xl font-bold text-gray-700 dark:text-white">Readless</span>
            </div>
            <div className="flex items-center space-x-6">
              <NavbarServerStatus />
              <Link href="/" className="text-gray-800 dark:text-white font-medium transition-colors">
                Home
              </Link>
              <Link
                href="/about"
                className="text-gray-600 dark:text-white/80 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                About
              </Link>
              <Link
                href="/contact"
                className="text-gray-600 dark:text-white/80 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                Contact
              </Link>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsDarkMode(!isDarkMode)}
                className="text-gray-700 dark:text-white hover:bg-white/10 dark:hover:bg-white/10"
              >
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 py-8">
        {!uploadedFile ? (
          // Landing Page / Upload Section
          <div className="text-center space-y-8">
            <div className="space-y-4">
              <h1 className="text-5xl font-bold text-gray-800 dark:text-white mb-4">Chat with your PDF</h1>
              <p className="text-xl text-gray-700 dark:text-white/80 mb-8">
                Upload any PDF and ask questions instantly.
              </p>
            </div>

            <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-8 rounded-2xl shadow-2xl">
              <div
                className={cn(
                  "border-2 border-dashed rounded-xl p-12 transition-all duration-300",
                  isDragOver
                    ? "border-blue-400 bg-blue-50/10"
                    : "border-gray-400/50 dark:border-white/30 hover:border-gray-500 dark:hover:border-white/50",
                )}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
              >
                <div className="text-center space-y-4">
                  <Upload className="h-16 w-16 text-gray-500 dark:text-white/60 mx-auto" />
                  <div>
                    <p className="text-lg text-gray-700 dark:text-white/80 mb-2">Drag and drop your PDF here, or</p>
                    <Button
                      onClick={() => fileInputRef.current?.click()}
                      className="bg-white/20 hover:bg-white/30 text-gray-800 dark:text-white border-gray-400/30 dark:border-white/30"
                      disabled={isUploading}
                    >
                      Choose File
                    </Button>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-white/60">Supports PDF files up to 10MB</p>
                </div>
              </div>

              {isUploading && (
                <div className="mt-6 space-y-2">
                  <div className="flex justify-between text-sm text-gray-700 dark:text-white/80">
                    <span>Uploading...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} className="h-2" />
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) handleFileUpload(file)
                }}
                className="hidden"
              />
            </Card>
          </div>
        ) : (
          // Chat Interface
          <div className="space-y-6">
            {/* File Info */}
            <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-4 rounded-2xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <FileText className="h-8 w-8 text-gray-700 dark:text-white" />
                  <div>
                    <p className="text-gray-800 dark:text-white font-medium">{uploadedFile.name}</p>
                    <p className="text-gray-600 dark:text-white/60 text-sm">
                      {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={clearPDF}
                  className="text-gray-600 dark:text-white/60 hover:text-gray-800 dark:hover:text-white hover:bg-white/10"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
            </Card>

            {/* Chat Messages */}
            <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 rounded-2xl overflow-hidden">
              <div className="h-96 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 ? (
                  <div className="text-center text-gray-600 dark:text-white/60 py-8">
                    <p>Start asking questions about your PDF!</p>
                  </div>
                ) : (
                  messages.map((message) => (
                    <div key={message.id} className={cn("flex", message.isUser ? "justify-end" : "justify-start")}>
                      <div
                        className={cn(
                          "max-w-xs lg:max-w-md px-4 py-2 rounded-2xl shadow-lg",
                          message.isUser
                            ? "bg-blue-500 text-white"
                            : "bg-white/90 dark:bg-gray-800 text-gray-900 dark:text-white",
                        )}
                      >
                        {message.isUser ? (
                          <p className="text-sm">{message.content}</p>
                        ) : (
                          <div className="text-sm prose prose-sm dark:prose-invert max-w-none">
                            <ReactMarkdown
                              components={{
                                h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                                h2: ({ children }) => <h2 className="text-md font-semibold mb-1">{children}</h2>,
                                h3: ({ children }) => <h3 className="text-sm font-medium mb-1">{children}</h3>,
                                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                                ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                                ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                                li: ({ children }) => <li className="text-sm">{children}</li>,
                                strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                                em: ({ children }) => <em className="italic">{children}</em>,
                                code: ({ children }) => <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-xs font-mono">{children}</code>,
                                blockquote: ({ children }) => <blockquote className="border-l-4 border-gray-300 dark:border-gray-600 pl-3 italic">{children}</blockquote>,
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        )}
                        <p className="text-xs opacity-70 mt-1">{message.timestamp.toLocaleTimeString()}</p>
                      </div>
                    </div>
                  ))
                )}

                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-white/90 dark:bg-gray-800 px-4 py-2 rounded-2xl shadow-lg">
                      <p className="text-sm text-gray-600 dark:text-gray-300">AI is thinking...</p>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Bar */}
              <div className="border-t border-white/20 dark:border-white/10 p-4">
                <div className="flex space-x-2">
                  <Input
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder="Ask a question about your PDF..."
                    className="flex-1 bg-white/20 border-gray-400/30 dark:border-white/30 text-gray-800 dark:text-white placeholder:text-gray-600 dark:placeholder:text-white/60 rounded-xl"
                    onKeyPress={(e) => {
                      if (e.key === "Enter") {
                        sendMessage()
                      }
                    }}
                  />
                  
                  {/* Voice Chat Buttons */}
                  {voiceSupported && (
                    <>
                      <Button
                        onClick={isListening ? stopListening : startListening}
                        disabled={!uploadedFile || isTyping}
                        variant={isListening ? "destructive" : "outline"}
                        className={cn(
                          "rounded-xl px-3",
                          isListening 
                            ? "bg-red-500 hover:bg-red-600 text-white animate-pulse" 
                            : "bg-white/20 border-gray-400/30 dark:border-white/30 text-gray-800 dark:text-white hover:bg-white/30"
                        )}
                        title={isListening ? "Stop listening" : "Start voice chat"}
                      >
                        {isListening ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                      </Button>
                      
                      <Button
                        onClick={stopSpeaking}
                        disabled={!isSpeaking}
                        variant="outline"
                        className={cn(
                          "rounded-xl px-3",
                          isSpeaking
                            ? "bg-orange-500 hover:bg-orange-600 text-white animate-pulse"
                            : "bg-white/20 border-gray-400/30 dark:border-white/30 text-gray-800 dark:text-white hover:bg-white/30"
                        )}
                        title="Stop speaking"
                      >
                        <Volume2 className="h-4 w-4" />
                      </Button>
                    </>
                  )}
                  
                  <Button
                    onClick={sendMessage}
                    disabled={!inputMessage.trim() || isTyping}
                    className="bg-blue-500 hover:bg-blue-600 text-white rounded-xl px-4"
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
                
                {/* Voice Status Indicator */}
                {voiceSupported && isListening && (
                  <div className="mt-2 text-center">
                    <p className="text-sm text-blue-600 dark:text-blue-400 animate-pulse">
                      🎤 Listening... Speak your question
                    </p>
                  </div>
                )}
                
                {!voiceSupported && (
                  <div className="mt-2 text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Voice chat not supported in this browser
                    </p>
                  </div>
                )}
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
