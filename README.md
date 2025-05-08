# 🎯 Katomaran Face Recognition Platform

A powerful **3-tab browser-based platform** for real-time **face registration**, **live recognition**, and **AI-powered activity querying** — built for speed, precision, and ease of use.

---

## 🧠 Live Interactive Tabs

🎥 **Face Registration** | 📡 **Live Recognition** | 💬 **AI Activity Chat**

> Intuitive UI with tabs that guide the user through:
- 👤 Registering new faces
- 👁️ Real-time face recognition
- 🧠 Querying face activity history using natural language

---

## 🚀 Features

- 🔴 **Webcam-Based Face Registration**  
- 🟢 **Live Face Detection & Recognition**  
- 🧠 **AI-Powered Chat to Query Activity Logs**  
- ⚡ **Real-Time Sync via WebSockets**  
- 💻 **Sleek React UI with TailwindCSS & Framer Motion**  

---

## 🔧 Prerequisites

- **🧠 Python**: `3.10.17`  
  - `DeepFace` + `MediaPipe` for face detection & recognition  
  - `TensorFlow 2.10.0` & `Keras 2.10.0` (only if using DeepFace with Facenet)  

- **🟢 Node.js**: `v18.0.0` or later  

- **🪟 Windows with WSL**: Required for Python runtime  

---

## 📦 Tech Stack

### 🐍 Python Packages
| Package | Purpose |
|--------|---------|
| `deepface==0.0.79` | Face recognition (Facenet model) |
| `mediapipe` | Lightweight face detection |
| `opencv-python` | Webcam interface & frame handling |
| `tensorflow==2.10.0`, `keras==2.10.0` | Only when required |
| `fastapi` | Backend REST API |
| `langchain`, `faiss-cpu` | RAG pipeline for AI chat |
| `sqlite3` | Event history storage |

### 💻 JavaScript Packages
| Package | Purpose |
|--------|---------|
| `react`, `tailwindcss`, `framer-motion` | UI/UX |
| `express`, `socket.io` | Backend server & WebSocket bridge |

---

## 📂 Core Python Logic – `face_and_identification.py`

This file handles both **face registration** and **recognition**:

```python
capture_image(name)     # Opens webcam and captures an image
recognize_face_from_frame(frame)  # Compares captured frame with registered faces


👇 Sample Usage (Command Line)

$ python face_and_identification.py

Then choose:

    1: Register a face — saves image to registered_faces/

    2: Recognize a face — scans webcam & finds matches

✅ Uses DeepFace with Facenet model (no dlib or cmake)
✅ Compatible with execute.ps1 startup flow
🛠️ Installation & Setup

    Clone the Repo

git clone https://github.com/yourusername/katomaran-face-platform.git
cd katomaran-face-platform

Run Setup Script

    # Run in PowerShell (as Admin)
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    .\execute.ps1

    This script:

    ✅ Installs all Python & Node.js dependencies

    ✅ Sets up folders like registered_faces/

    ✅ Starts chosen services (Test Mode / Servers / Full Platform)

🎮 Usage Modes
Mode	Description
🔍 Test Face Detection	Webcam + MediaPipe validation
🔧 Start Servers Only	FastAPI (8001) + Node (3002)
🚀 Complete System	Full platform + React UI (3000)
💻 Accessing the Platform

Once running:

    🌐 Frontend: http://localhost:3000

    📘 API Docs: http
