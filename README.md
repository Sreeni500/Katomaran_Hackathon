# Katomaran Face Recognition Platform 🎯

A powerful 3-tab browser-based platform for face registration, live recognition, and AI-powered activity querying.

## 🚀 Features

- **Face Registration**: Register new faces through webcam
- **Live Face Recognition**: Real-time face detection and recognition
- **AI Chat Interface**: Query face activity history using natural language
- **Modern UI**: Responsive React.js frontend with TailwindCSS
- **Real-time Communication**: WebSocket integration between React, Node.js, and Python

## 🔧 Prerequisites

- **Python**: Version 3.10.17
  - Uses DeepFace and MediaPipe for face processing
  - TensorFlow 2.10.0 and Keras 2.10.0 for deep learning
- **Node.js**: Version 18.0.0 or higher
- **Windows**: WSL enabled for Python operations

## 📦 Tech Stack

### Python Packages
- `deepface==0.0.79`: Face recognition
- `mediapipe`: Face detection
- `opencv-python`: Video processing
- `tensorflow==2.10.0`: Deep learning framework
- `keras==2.10.0`: Neural network library
- `fastapi`: Backend API
- `langchain`: RAG pipeline
- `faiss-cpu`: Vector similarity search
- `sqlite3`: Database management

### JavaScript/Node.js Packages
- `react`: Frontend framework
- `socket.io`: Real-time communication
- `express`: WebSocket server
- `tailwindcss`: UI styling
- `framer-motion`: Animations

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/katomaran-face-platform.git
   cd katomaran-face-platform
   ```

2. **Run the Setup Script**
   ```powershell
   # Open PowerShell as Administrator
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\execute.ps1
   ```

   The script will:
   - Check Python and Node.js installations
   - Install required Python packages
   - Install Node.js dependencies
   - Create necessary directories
   - Start the services based on your choice

## 🎮 Usage Options

The platform can be started in three modes:

1. **Test Face Detection**
   - Tests webcam and face detection functionality
   - Useful for verifying system setup

2. **Start Servers Only**
   - Launches Python FastAPI backend (Port 8001)
   - Starts Node.js WebSocket server (Port 3002)
   - Ideal for development and testing

3. **Complete System**
   - Starts all servers
   - Launches React frontend (Port 3000)
   - Full production mode

## 💻 Accessing the Platform

After starting the complete system:

1. Open your browser and navigate to:
   - Frontend UI: http://localhost:3000
   - API Documentation: http://localhost:8001/docs
   - WebSocket Server: ws://localhost:3002

2. Use the tabs to:
   - Register new faces
   - Monitor live recognition
   - Query face activity through AI chat

## 📝 Important Notes

1. **Face Registration**:
   - Ensure good lighting
   - Position face centrally in frame
   - Use "Force Register" option if detection fails

2. **Live Recognition**:
   - Maintains real-time WebSocket connection
   - Shows confidence scores for matches
   - Logs all recognition events

3. **AI Querying**:
   - Uses natural language processing
   - Searches face activity history
   - Provides context-aware responses

## 🔍 Troubleshooting

1. **Face Detection Issues**:
   - Check lighting conditions
   - Ensure face is clearly visible
   - Verify webcam permissions

2. **Connection Errors**:
   - Confirm all ports are available
   - Check if services are running
   - Verify WebSocket connection

3. **Package Conflicts**:
   - Use specified versions of TensorFlow and Keras
   - Avoid installing multiple versions
   - Follow the installation order in execute.ps1

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- DeepFace library contributors
- MediaPipe team
- OpenRouter for AI capabilities 