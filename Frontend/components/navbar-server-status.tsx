"use client"

import { useState, useEffect } from "react"
import { cn } from "@/lib/utils"
import { API_CONFIG } from "@/lib/api"

interface ServerStatus {
  main: 'checking' | 'online' | 'offline'
  voice: 'checking' | 'online' | 'offline'
}

export function NavbarServerStatus() {
  const [status, setStatus] = useState<ServerStatus>({
    main: 'checking',
    voice: 'checking'
  })

  const checkServerStatus = async () => {
    // Check Main Server (PDF + Voice)
    try {
      const mainResponse = await fetch(`${API_CONFIG.BASE_URL}/health`, { 
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      })
      setStatus(prev => ({ ...prev, main: mainResponse.ok ? 'online' : 'offline' }))
    } catch {
      setStatus(prev => ({ ...prev, main: 'offline' }))
    }

    // Check Voice Chat functionality
    try {
      const voiceResponse = await fetch(`${API_CONFIG.BASE_URL}/api/voice-chat/health`, { 
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
    const mainStatus = status.main === 'online' ? 'Main Server: Online' : status.main === 'offline' ? 'Main Server: Offline' : 'Main Server: Checking...'
    const voiceStatus = status.voice === 'online' ? 'Voice Chat: Online' : status.voice === 'offline' ? 'Voice Chat: Limited' : 'Voice Chat: Checking...'
    return `${mainStatus}\n${voiceStatus}`
  }

  return (
    <div className="flex items-center space-x-2" title={getTooltipText()}>
      <div className="flex items-center space-x-1">
        {/* Main Server Status */}
        <div className={cn(
          "w-2 h-2 rounded-full",
          getStatusColor(status.main)
        )} />
        
        {/* Voice Chat Status */}
        <div className={cn(
          "w-2 h-2 rounded-full",
          getStatusColor(status.voice)
        )} />
      </div>
      
      <span className="text-xs text-gray-600 dark:text-gray-400 font-medium">
        {status.main === 'online' && status.voice === 'online' ? 'All Systems Online' :
         status.main === 'offline' ? 'Server Offline' :
         status.voice === 'offline' ? 'Voice Limited' :
         'Checking...'}
      </span>
    </div>
  )
}
