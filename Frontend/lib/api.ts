// API configuration and functions for connecting to backend services

export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 
    (process.env.NODE_ENV === 'production' 
      ? 'https://readless-i6cb.onrender.com' 
      : 'http://localhost:8000'),
  // Keeping legacy names for compatibility
  PDF_BASE_URL: process.env.NEXT_PUBLIC_API_URL || 
    (process.env.NODE_ENV === 'production' 
      ? 'https://readless-i6cb.onrender.com' 
      : 'http://localhost:8000'),
  VOICE_BASE_URL: process.env.NEXT_PUBLIC_API_URL || 
    (process.env.NODE_ENV === 'production' 
      ? 'https://readless-i6cb.onrender.com' 
      : 'http://localhost:8000'),
}

export interface SessionResponse {
  session_id: string
  message: string
}

export interface UploadResponse {
  message: string
  filename: string
  file_id: string
  pages: number
}

export interface AnswerResponse {
  question: string
  answer: string
  session_id: string
  timestamp: string
}

export interface ChatHistory {
  messages: Array<{
    content: string
    is_user: boolean
    timestamp: string
  }>
}

export interface VoiceChatResponse {
  response: string
  audio_file?: string
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

// Check if backend is available
async function checkBackendHealth(baseUrl: string, healthEndpoint: string = '/'): Promise<boolean> {
  try {
    const response = await fetch(`${baseUrl}${healthEndpoint}`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5 second timeout
    })
    return response.ok
  } catch {
    return false
  }
}

// PDF API functions
export async function createSession(): Promise<SessionResponse> {
  try {
    // Check if PDF backend is available
    const isAvailable = await checkBackendHealth(API_CONFIG.PDF_BASE_URL)
    if (!isAvailable) {
      throw new ApiError(503, 'PDF backend server is not running. Please start the server on port 8000.')
    }

    const response = await fetch(`${API_CONFIG.PDF_BASE_URL}/session/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      throw new ApiError(response.status, `Failed to create session: ${response.statusText}`)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

export async function uploadPDF(file: File, sessionId: string): Promise<UploadResponse> {
  try {
    const formData = new FormData()
    formData.append('files', file)  // Note: backend expects 'files' not 'file'

    const response = await fetch(`${API_CONFIG.PDF_BASE_URL}/upload?session_id=${encodeURIComponent(sessionId)}`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new ApiError(response.status, `Failed to upload PDF: ${errorText}`)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

export async function askQuestion(question: string, sessionId: string): Promise<AnswerResponse> {
  try {
    const response = await fetch(`${API_CONFIG.PDF_BASE_URL}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        session_id: sessionId,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new ApiError(response.status, `Failed to ask question: ${errorText}`)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

export async function getChatHistory(sessionId: string): Promise<ChatHistory> {
  try {
    const response = await fetch(`${API_CONFIG.PDF_BASE_URL}/session/${sessionId}/history`)

    if (!response.ok) {
      throw new ApiError(response.status, `Failed to get chat history: ${response.statusText}`)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

// Voice API functions
export async function voiceChat(message: string): Promise<VoiceChatResponse> {
  try {
    const response = await fetch(`${API_CONFIG.VOICE_BASE_URL}/api/voice-chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new ApiError(response.status, `Voice chat failed: ${errorText}`)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

export async function textToSpeech(text: string): Promise<Blob> {
  try {
    const response = await fetch(`${API_CONFIG.VOICE_BASE_URL}/api/text-to-speech`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new ApiError(response.status, `Text-to-speech failed: ${errorText}`)
    }

    return await response.blob()
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

export async function speechToText(audioBlob: Blob): Promise<{ text: string }> {
  try {
    const formData = new FormData()
    formData.append('audio', audioBlob, 'audio.wav')

    const response = await fetch(`${API_CONFIG.VOICE_BASE_URL}/api/speech-to-text`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new ApiError(response.status, `Speech-to-text failed: ${errorText}`)
    }

    return await response.json()
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(0, `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}
