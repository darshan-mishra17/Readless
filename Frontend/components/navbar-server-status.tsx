"use client"

import { useState, useEffect } from "react"
import { cn } from "@/lib/utils"
import { API_CONFIG } from "@/lib/api"

interface ServerStatus {
  pdf: 'checking' | 'online' | 'offline'
  voice: 'checking' | 'online' | 'offline'
}

export function NavbarServerStatus() {
  const [status, setStatus] = useState<ServerStatus>({
    pdf: 'checking',
    voice: 'checking'
  })

  const checkServerStatus = async () => {
    // Check PDF RAG Server
    try {
      const pdfResponse = await fetch(`${API_CONFIG.PDF_BASE_URL}/health`, { 
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      })
      setStatus(prev => ({ ...prev, pdf: pdfResponse.ok ? 'online' : 'offline' }))
    } catch {
      setStatus(prev => ({ ...prev, pdf: 'offline' }))
    }

    // Check Voice Chat Server
    try {
      const voiceResponse = await fetch(`${API_CONFIG.VOICE_BASE_URL}/api/voice-chat/health`, { 
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      })
      setStatus(prev => ({ ...prev, voice: voiceResponse.ok ? 'online' : 'offline' }))
    } catch {
      setStatus(prev => ({ ...prev, voice: 'offline' }))
    }
  }

  useEffect(() => {
    checkServerStatus()
    // Check every 30 seconds
    const interval = setInterval(checkServerStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (serverStatus: 'checking' | 'online' | 'offline') => {
    switch (serverStatus) {
      case 'checking':
        return 'bg-yellow-500 animate-pulse'
      case 'online':
        return 'bg-green-500'
      case 'offline':
        return 'bg-red-500'
    }
  }

  const getTooltipText = () => {
    const pdfStatus = status.pdf === 'online' ? 'PDF Server: Online' : status.pdf === 'offline' ? 'PDF Server: Offline' : 'PDF Server: Checking...'
    const voiceStatus = status.voice === 'online' ? 'Voice Server: Online' : status.voice === 'offline' ? 'Voice Server: Offline' : 'Voice Server: Checking...'
    return `${pdfStatus}\n${voiceStatus}`
  }

  return (
    <div className="flex items-center space-x-2" title={getTooltipText()}>
      <div className="flex items-center space-x-1">
        {/* PDF Server Status */}
        <div className={cn(
          "w-2 h-2 rounded-full",
          getStatusColor(status.pdf)
        )} />
        
        {/* Voice Server Status */}
        <div className={cn(
          "w-2 h-2 rounded-full",
          getStatusColor(status.voice)
        )} />
      </div>
      
      <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">
        {status.pdf === 'online' && status.voice === 'online' ? 'All Systems Online' :
         status.pdf === 'offline' || status.voice === 'offline' ? 'Service Issues' :
         'Checking...'}
      </span>
    </div>
  )
}
