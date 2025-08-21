"use client"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  FileText,
  Moon,
  Sun,
  GraduationCap,
  Scale,
  Heart,
  Building,
  Headphones,
  Zap,
  Database,
  Code,
  Cpu,
  Globe,
} from "lucide-react"
import Link from "next/link"

export default function AboutPage() {
  const [isDarkMode, setIsDarkMode] = useState(true)

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode)
  }, [isDarkMode])

  const useCases = [
    {
      icon: GraduationCap,
      title: "Students & Researchers",
      description: "Summarize papers, extract insights.",
    },
    {
      icon: Scale,
      title: "Legal & Corporate Teams",
      description: "Review contracts, compliance docs.",
    },
    {
      icon: Heart,
      title: "Healthcare & Pharma",
      description: "Extract findings from medical research.",
    },
    {
      icon: Building,
      title: "Government & Policy",
      description: "Make policies and tenders more accessible.",
    },
    {
      icon: Headphones,
      title: "Customer Support",
      description: "Answer questions from manuals instantly.",
    },
  ]

  const techStack = [
    {
      icon: Zap,
      name: "LangChain",
      description: "PDF parsing and AI workflows",
    },
    {
      icon: Cpu,
      name: "Groq API",
      description: "Lightning-fast inference",
    },
    {
      icon: Code,
      name: "Python / FastAPI",
      description: "Backend service",
    },
    {
      icon: Database,
      name: "MongoDB",
      description: "Data storage",
    },
    {
      icon: Globe,
      name: "React/Tailwind",
      description: "Chat interface",
    },
  ]

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
              <Link
                href="/"
                className="text-gray-600 dark:text-white/80 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                Home
              </Link>
              <Link href="/about" className="text-gray-800 dark:text-white font-medium transition-colors">
                About
              </Link>
              <a
                href="#"
                className="text-gray-600 dark:text-white/80 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                Contact
              </a>
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

      <div className="max-w-6xl mx-auto px-4 py-12 space-y-16">
        {/* Hero Section */}
        <div className="text-center space-y-6">
          <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-12 rounded-3xl shadow-2xl">
            <div className="space-y-6">
              <h1 className="text-6xl font-bold text-gray-800 dark:text-white">
                About <span className="text-blue-600 dark:text-blue-400">Readless</span>
              </h1>
              <p className="text-2xl text-gray-700 dark:text-white/80 max-w-3xl mx-auto">
                Transforming PDFs into interactive conversations with AI.
              </p>
              <div className="flex justify-center items-center space-x-4 pt-4">
                <FileText className="h-12 w-12 text-blue-500" />
                <div className="text-4xl text-gray-400">+</div>
                <Cpu className="h-12 w-12 text-purple-500" />
                <div className="text-4xl text-gray-400">=</div>
                <Zap className="h-12 w-12 text-yellow-500" />
              </div>
            </div>
          </Card>
        </div>

        {/* About the App */}
        <div className="space-y-8">
          <h2 className="text-4xl font-bold text-center text-gray-800 dark:text-white">What is Readless?</h2>
          <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-8 rounded-2xl">
            <div className="text-center space-y-6">
              <p className="text-xl text-gray-700 dark:text-white/80 leading-relaxed max-w-4xl mx-auto">
                Readless helps you upload PDFs and chat with them in natural language. Instead of spending hours reading
                through lengthy documents, simply ask questions and get instant, accurate answers.
              </p>
              <div className="grid md:grid-cols-3 gap-6 pt-6">
                <div className="space-y-3">
                  <div className="h-16 w-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto">
                    <Zap className="h-8 w-8 text-blue-500" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Time-Saving</h3>
                  <p className="text-gray-600 dark:text-white/70">Get answers in seconds, not hours</p>
                </div>
                <div className="space-y-3">
                  <div className="h-16 w-16 bg-green-500/20 rounded-full flex items-center justify-center mx-auto">
                    <Cpu className="h-8 w-8 text-green-500" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Productivity</h3>
                  <p className="text-gray-600 dark:text-white/70">Focus on insights, not searching</p>
                </div>
                <div className="space-y-3">
                  <div className="h-16 w-16 bg-purple-500/20 rounded-full flex items-center justify-center mx-auto">
                    <Globe className="h-8 w-8 text-purple-500" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Accessibility</h3>
                  <p className="text-gray-600 dark:text-white/70">Make any document conversational</p>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Use Cases */}
        <div className="space-y-8">
          <h2 className="text-4xl font-bold text-center text-gray-800 dark:text-white">Where can you use it?</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {useCases.map((useCase, index) => (
              <Card
                key={index}
                className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-6 rounded-2xl hover:bg-white/20 dark:hover:bg-black/20 transition-all duration-300 hover:scale-105"
              >
                <div className="space-y-4">
                  <div className="h-12 w-12 bg-blue-500/20 rounded-xl flex items-center justify-center">
                    <useCase.icon className="h-6 w-6 text-blue-500" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white">{useCase.title}</h3>
                  <p className="text-gray-600 dark:text-white/70">{useCase.description}</p>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Tools & Tech */}
        <div className="space-y-8">
          <h2 className="text-4xl font-bold text-center text-gray-800 dark:text-white">How we built it</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {techStack.map((tech, index) => (
              <Card
                key={index}
                className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-6 rounded-2xl hover:bg-white/20 dark:hover:bg-black/20 transition-all duration-300"
              >
                <div className="flex items-center space-x-4">
                  <div className="h-12 w-12 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl flex items-center justify-center">
                    <tech.icon className="h-6 w-6 text-blue-500" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800 dark:text-white">{tech.name}</h3>
                    <p className="text-sm text-gray-600 dark:text-white/70">{tech.description}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>

        {/* Mission Statement */}
        <div className="text-center">
          <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-12 rounded-3xl">
            <h2 className="text-3xl font-bold text-gray-800 dark:text-white mb-6">Our Mission</h2>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400 leading-relaxed">
              "Our mission is to make knowledge inside documents instantly accessible to everyone."
            </p>
          </Card>
        </div>

        {/* Footer */}
        <footer className="text-center pt-12 border-t border-white/20 dark:border-white/10">
          <div className="space-y-4">
            <p className="text-gray-600 dark:text-white/60">© 2024 Readless. All rights reserved.</p>
            <div className="flex justify-center space-x-6">
              <Link
                href="/"
                className="text-gray-600 dark:text-white/60 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                Home
              </Link>
              <Link
                href="/about"
                className="text-gray-600 dark:text-white/60 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                About
              </Link>
              <a
                href="#"
                className="text-gray-600 dark:text-white/60 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                Contact
              </a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}
