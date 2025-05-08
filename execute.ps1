# Katomaran Face Recognition Platform - Execution Order

# 1. Environment Setup
Write-Host "1. Setting up environment..." -ForegroundColor Yellow

# Check Python
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python 3.10.17 required" -ForegroundColor Red
    exit 1
}

# Check Node.js
node --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Node.js required" -ForegroundColor Red
    exit 1
}

# 2. Install Dependencies
Write-Host "`n2. Installing dependencies..." -ForegroundColor Yellow

# First uninstall any existing TensorFlow and Keras to avoid conflicts
Write-Host "Removing existing TensorFlow and Keras installations..." -ForegroundColor Cyan
python -m pip uninstall -y tensorflow tensorflow-intel keras

# Install specific versions of TensorFlow and Keras
Write-Host "Installing TensorFlow 2.10.0 and Keras 2.10.0..." -ForegroundColor Cyan
python -m pip install tensorflow==2.10.0 --no-deps
python -m pip install keras==2.10.0 --no-deps

# Install other dependencies
Write-Host "Installing other Python packages..." -ForegroundColor Cyan
python -m pip install deepface==0.0.79 mediapipe opencv-python protobuf==3.20.0
python -m pip install "fastapi[all]" langchain pydantic uvicorn faiss-cpu

# Node.js packages
Write-Host "Installing Node.js packages..." -ForegroundColor Cyan
Set-Location -Path "node_ws"
npm install
Set-Location -Path ".."

Set-Location -Path "frontend"
npm install
Set-Location -Path ".."

# 3. Create Directories
Write-Host "`n3. Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "registered_faces"
New-Item -ItemType Directory -Force -Path "test_images"
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "database"

# 4. Start Services
Write-Host "`n4. Starting services..." -ForegroundColor Yellow

# Option menu
$option = Read-Host "Choose an option:
1. Test face detection
2. Start servers only
3. Start complete system
Enter (1-3)"

switch ($option) {
    "1" {
        python test_face_recognition.py
    }
    "2" {
        Start-Process powershell -ArgumentList "python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8001"
        Start-Process powershell -ArgumentList "cd node_ws; node server.js"
    }
    "3" {
        Start-Process powershell -ArgumentList "python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8001"
        Start-Process powershell -ArgumentList "cd node_ws; node server.js"
        Set-Location -Path "frontend"
        npm start
    }
}

Write-Host "`nServices started! Press Ctrl+C to stop." -ForegroundColor Green 