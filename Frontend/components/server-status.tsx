"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, XCircle, Loader2, Server, Terminal } from "lucide-react"
import { API_CONFIG } from "@/lib/api"

interface ServerStatus {
  name: string
  url: string
  port: number
  status: 'checking' | 'online' | 'offline'
  description: string
  healthEndpoint: string
}

export function ServerStatus() {
  const [servers, setServers] = useState<ServerStatus[]>([
    {
      name: 'PDF RAG Server',
      url: API_CONFIG.PDF_BASE_URL,
      port: 8000,
      status: 'offline',
      description: 'Handles PDF upload and question answering',
      healthEndpoint: '/health'
    },
    {
      name: 'Voice Chat Server',
      url: API_CONFIG.VOICE_BASE_URL,
      port: 8001,
      status: 'offline',
      description: 'Provides voice chat and TTS/STT functionality',
      healthEndpoint: '/api/voice-chat/health'
    }
  ])

  const checkServerStatus = async (server: ServerStatus) => {
    setServers(prev => prev.map(s => 
      s.port === server.port ? { ...s, status: 'checking' } : s
    ))

    try {
      const response = await fetch(`${server.url}${server.healthEndpoint}`, { 
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      })
      
      const status = response.ok ? 'online' : 'offline'
      setServers(prev => prev.map(s => 
        s.port === server.port ? { ...s, status } : s
      ))
    } catch {
      setServers(prev => prev.map(s => 
        s.port === server.port ? { ...s, status: 'offline' } : s
      ))
    }
  }

  const checkAllServers = () => {
    servers.forEach(checkServerStatus)
  }

  const getStatusIcon = (status: ServerStatus['status']) => {
    switch (status) {
      case 'checking':
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
      case 'online':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'offline':
        return <XCircle className="h-4 w-4 text-red-500" />
    }
  }

  const getStatusBadge = (status: ServerStatus['status']) => {
    switch (status) {
      case 'checking':
        return <Badge variant="outline">Checking...</Badge>
      case 'online':
        return <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">Online</Badge>
      case 'offline':
        return <Badge variant="destructive">Offline</Badge>
    }
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Server className="h-5 w-5" />
          Backend Server Status
        </CardTitle>
        <CardDescription>
          Check and manage your backend servers before using the application
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-3">
          {servers.map((server) => (
            <div
              key={server.port}
              className="flex items-center justify-between p-3 border rounded-lg"
            >
              <div className="flex items-center gap-3">
                {getStatusIcon(server.status)}
                <div>
                  <div className="font-medium">{server.name}</div>
                  <div className="text-sm text-muted-foreground">
                    {server.description}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Port: {server.port}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {getStatusBadge(server.status)}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => checkServerStatus(server)}
                  disabled={server.status === 'checking'}
                >
                  Check
                </Button>
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          <Button onClick={checkAllServers} className="flex-1">
            Check All Servers
          </Button>
        </div>

        <div className="mt-6 p-4 bg-muted rounded-lg">
          <h4 className="font-medium flex items-center gap-2 mb-2">
            <Terminal className="h-4 w-4" />
            How to Start Servers
          </h4>
          <div className="space-y-2 text-sm text-muted-foreground">
            <div>
              <strong>PDF RAG Server (Port 8000):</strong>
              <code className="block mt-1 p-2 bg-background rounded text-xs">
                cd Server && python app.py
              </code>
            </div>
            <div>
              <strong>Voice Chat Server (Port 8001):</strong>
              <code className="block mt-1 p-2 bg-background rounded text-xs">
                cd Server && python simple_voice_chat.py
              </code>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
