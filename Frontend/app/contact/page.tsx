"use client"

import type React from "react"
import Link from "next/link"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { useToast } from "@/hooks/use-toast"
import { Moon, Sun, FileText, Mail, Linkedin, Send } from "lucide-react"

export default function ContactPage() {
  const [isDarkMode, setIsDarkMode] = useState(true)
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  })
  const { toast } = useToast()

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDarkMode)
  }, [isDarkMode])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Simulate form submission
    toast({
      title: "Message sent!",
      description: "Thank you for reaching out. We'll get back to you soon.",
    })
    setFormData({ name: "", email: "", message: "" })
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#E0EAF3] to-[#cfdef3] dark:from-gray-900 dark:to-gray-800">
      {/* Navbar */}
      <nav className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-b border-white/20 dark:border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <FileText className="h-8 w-8 text-gray-700 dark:text-white" />
              <span className="text-xl font-bold text-gray-700 dark:text-white">ReadLess</span>
            </div>
            <div className="flex items-center space-x-6">
              <Link
                href="/"
                className="text-gray-600 dark:text-white/80 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                Home
              </Link>
              <Link
                href="/about"
                className="text-gray-600 dark:text-white/80 hover:text-gray-800 dark:hover:text-white transition-colors"
              >
                About
              </Link>
              <Link href="/contact" className="text-gray-800 dark:text-white font-medium transition-colors">
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

      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center space-y-4 mb-12">
          <h1 className="text-5xl font-bold text-gray-800 dark:text-white">Get in Touch</h1>
          <p className="text-xl text-gray-700 dark:text-white/80">We'd love to hear from you!</p>
        </div>

        {/* Contact Options Grid */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Contact Cards */}
          <div className="space-y-6">
            {/* Email Card */}
            <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-6 rounded-2xl shadow-2xl hover:bg-white/20 dark:hover:bg-black/20 transition-all duration-300 group">
              <div className="flex items-center space-x-4">
                <div className="p-3 rounded-full bg-blue-500/20 group-hover:bg-blue-500/30 transition-colors">
                  <Mail className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-1">Email</h3>
                  <a
                    href="mailto:mishradarshan22@gmail.com"
                    className="text-blue-600 dark:text-blue-400 hover:underline transition-colors"
                  >
                    mishradarshan22@gmail.com
                  </a>
                </div>
              </div>
            </Card>

            {/* LinkedIn Card */}
            <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-6 rounded-2xl shadow-2xl hover:bg-white/20 dark:hover:bg-black/20 transition-all duration-300 group">
              <div className="flex items-center space-x-4">
                <div className="p-3 rounded-full bg-blue-500/20 group-hover:bg-blue-500/30 transition-colors">
                  <Linkedin className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-1">LinkedIn</h3>
                  <a
                    href="https://www.linkedin.com/in/darshan-mishra-8834b51b6"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 dark:text-blue-400 hover:underline transition-colors"
                  >
                    Darshan Mishra
                  </a>
                </div>
              </div>
            </Card>

            {/* Quick Message Section */}
            <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-6 rounded-2xl shadow-2xl">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-3">Reach out for:</h3>
              <ul className="space-y-2 text-gray-700 dark:text-white/80">
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Feedback on the app</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Collaboration opportunities</span>
                </li>
                <li className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>General inquiries</span>
                </li>
              </ul>
            </Card>
          </div>

          {/* Contact Form */}
          <Card className="backdrop-blur-md bg-white/10 dark:bg-black/10 border-white/20 dark:border-white/10 p-6 rounded-2xl shadow-2xl">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white mb-4">Send a Message</h3>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <Input
                  name="name"
                  placeholder="Your Name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className="bg-white/20 border-gray-400/30 dark:border-white/30 text-gray-800 dark:text-white placeholder:text-gray-600 dark:placeholder:text-white/60 rounded-xl"
                  required
                />
              </div>
              <div>
                <Input
                  name="email"
                  type="email"
                  placeholder="Your Email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className="bg-white/20 border-gray-400/30 dark:border-white/30 text-gray-800 dark:text-white placeholder:text-gray-600 dark:placeholder:text-white/60 rounded-xl"
                  required
                />
              </div>
              <div>
                <Textarea
                  name="message"
                  placeholder="Your Message"
                  value={formData.message}
                  onChange={handleInputChange}
                  rows={4}
                  className="bg-white/20 border-gray-400/30 dark:border-white/30 text-gray-800 dark:text-white placeholder:text-gray-600 dark:placeholder:text-white/60 rounded-xl resize-none"
                  required
                />
              </div>
              <Button
                type="submit"
                className="w-full bg-blue-500 hover:bg-blue-600 text-white rounded-xl py-2 flex items-center justify-center space-x-2 transition-colors"
              >
                <Send className="h-4 w-4" />
                <span>Send Message</span>
              </Button>
            </form>
          </Card>
        </div>

        {/* Footer */}
        <footer className="text-center pt-8 border-t border-white/20 dark:border-white/10">
          <p className="text-gray-600 dark:text-white/60 mb-4">© 2025 ReadLess, All Rights Reserved.</p>
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
            <Link
              href="/contact"
              className="text-gray-600 dark:text-white/60 hover:text-gray-800 dark:hover:text-white transition-colors"
            >
              Contact
            </Link>
          </div>
        </footer>
      </div>
    </div>
  )
}
