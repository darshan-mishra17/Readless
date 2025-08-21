#!/usr/bin/env powershell
# PowerShell script to start the ReadLess FastAPI server

Write-Host "Starting ReadLess FastAPI Server..." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Change to the server directory
Set-Location "c:\Users\HP\OneDrive\Desktop\ReadLess\Server"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Start the server
Write-Host "Starting FastAPI server..." -ForegroundColor Yellow
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Alternative Documentation: http://localhost:8000/redoc" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Run uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
