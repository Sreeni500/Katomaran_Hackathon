import os
import base64
import json
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import logging
import sys

# Add parent directory to path to import face recognition modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_recognition.improved_detection import (
    detect_face,
    recognize_face_from_frame,
    decode_image
)

# Import RAG engine
from ..rag_engine.minimal_rag import MinimalRAGEngine

# Import logger
from ..utils.logger import setup_logger, log_function_call

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_api')

# Initialize FastAPI app
app = FastAPI(title="Katomaran Face Recognition Platform")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG chat engine
rag_engine = MinimalRAGEngine()

# Pydantic models for requests and responses
class RegisterFaceRequest(BaseModel):
    name: str
    frame_data: str  # Base64 encoded image
    force_register: Optional[bool] = True  # Default to True

class RegisterFaceResponse(BaseModel):
    success: bool
    message: str

class RecognizeFaceRequest(BaseModel):
    frame_data: str  # Base64 encoded image

class RecognizeFaceResponse(BaseModel):
    success: bool
    message: str
    faces: List[Dict[str, Any]] = []

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str

class FaceData(BaseModel):
    frame_data: str
    check_only: bool = False

class RegistrationData(BaseModel):
    name: str
    frame_data: str
    force_register: bool = False

@app.get("/")
def read_root():
    """API root endpoint"""
    return {"message": "Katomaran Face Recognition Platform API"}

@app.post("/register-face", response_model=RegisterFaceResponse)
@log_function_call(logger)
def api_register_face(request: RegisterFaceRequest):
    """Register a face with the given name"""
    try:
        # Input validation
        if not request.name:
            return RegisterFaceResponse(
                success=False,
                message="Name is required"
            )
            
        if not request.frame_data:
            return RegisterFaceResponse(
                success=False,
                message="Image data is required"
            )
            
        # Call the simplified register_face function
        result = register_face(
            request.name, 
            request.frame_data, 
            force_register=True  # Always force registration
        )
        
        # Return the result directly
        return RegisterFaceResponse(
            success=result["success"],
            message=result["message"]
        )
    except Exception as e:
        logger.error(f"Error registering face: {str(e)}")
        # Return a response instead of raising an exception
        return RegisterFaceResponse(
            success=False,
            message=f"Error registering face: {str(e)}"
        )

@app.post("/recognize-face", response_model=RecognizeFaceResponse)
@log_function_call(logger)
def api_recognize_face(request: RecognizeFaceRequest):
    """Recognize faces in a frame"""
    try:
        # Input validation
        if not request.frame_data:
            return RecognizeFaceResponse(
                success=False,
                message="Image data is required",
                faces=[]
            )
            
        # Call the simplified recognize_face function
        result = recognize_face(request.frame_data)
        
        # Return the result directly
        return RecognizeFaceResponse(
            success=result["success"],
            message=result["message"],
            faces=result["faces"]
        )
    except Exception as e:
        logger.error(f"Error recognizing face: {str(e)}")
        # Return a response instead of raising an exception
        return RecognizeFaceResponse(
            success=False,
            message=f"Error recognizing face: {str(e)}",
            faces=[]
        )

@app.post("/chat", response_model=ChatResponse)
@log_function_call(logger)
def api_chat(request: ChatRequest):
    """Process a chat request using RAG"""
    try:
        response = rag_engine.chat(request.query, request.chat_history)
        
        return ChatResponse(
            response=response
        )
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )

@app.post("/api/recognize-face")
async def recognize_face(face_data: FaceData):
    """
    Recognize faces in the provided frame
    
    Args:
        face_data: FaceData object containing frame_data and check_only flag
        
    Returns:
        dict: Recognition results with success, message and faces
    """
    try:
        # Decode the image
        frame = decode_image(face_data.frame_data)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
            
        # If check_only is True, just perform detection
        if face_data.check_only:
            face_detected, _, confidence = detect_face(frame, check_only=True)
            return {
                "success": face_detected,
                "message": "Face detected" if face_detected else "No face detected",
                "faces": [{
                    "detected": face_detected,
                    "confidence": confidence
                }] if face_detected else []
            }
            
        # Perform full recognition
        name, confidence = recognize_face_from_frame(frame)
        
        if name:
            return {
                "success": True,
                "message": f"Face recognized as {name}",
                "faces": [{
                    "label": name,
                    "confidence": confidence,
                    "registered": True
                }]
            }
        else:
            # Face was detected but not recognized
            face_detected, _, detection_confidence = detect_face(frame)
            if face_detected:
                return {
                    "success": True,
                    "message": "Face detected but not recognized",
                    "faces": [{
                        "label": "Unknown",
                        "confidence": detection_confidence,
                        "registered": False
                    }]
                }
            else:
                return {
                    "success": False,
                    "message": "No face detected",
                    "faces": []
                }
                
    except Exception as e:
        logger.error(f"Error in recognize_face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register-face")
async def register_face(reg_data: RegistrationData):
    """
    Register a new face
    
    Args:
        reg_data: RegistrationData object containing name and frame_data
        
    Returns:
        dict: Registration results
    """
    try:
        # Validate name
        if not reg_data.name or not reg_data.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")
            
        # Decode the image
        frame = decode_image(reg_data.frame_data)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
            
        # Detect face in the frame
        face_detected, face_region, confidence = detect_face(frame)
        
        if not face_detected and not reg_data.force_register:
            return {
                "success": False,
                "message": "No face detected in the image. Please try again with better lighting and positioning."
            }
            
        # Save the face (either detected region or full frame if force_register)
        image_to_save = face_region if face_detected else frame
        
        # Save to registered_faces directory
        REGISTERED_DIR = "registered_faces"
        os.makedirs(REGISTERED_DIR, exist_ok=True)
        
        image_path = os.path.join(REGISTERED_DIR, f"{reg_data.name}.jpg")
        cv2.imwrite(image_path, image_to_save)
        
        return {
            "success": True,
            "message": "Face registered successfully" if face_detected else "Image saved (forced registration)",
            "confidence": confidence if face_detected else 0
        }
        
    except Exception as e:
        logger.error(f"Error in register_face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/recognition")
async def websocket_recognition(websocket: WebSocket):
    """WebSocket endpoint for live face recognition"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                data_json = json.loads(data)
                
                if "frame_data" in data_json:
                    # Process face recognition
                    result = recognize_face(data_json["frame_data"])
                    
                    # Send back result
                    await manager.send_message(json.dumps(result), websocket)
                else:
                    await manager.send_message(
                        json.dumps({"error": "Invalid data format"}),
                        websocket
                    )
            
            except Exception as e:
                logger.error(f"Error processing WebSocket data: {str(e)}")
                await manager.send_message(
                    json.dumps({"error": str(e)}),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for chat"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                data_json = json.loads(data)
                
                if "query" in data_json:
                    # Process chat
                    chat_history = data_json.get("chat_history", None)
                    response = rag_engine.chat(data_json["query"], chat_history)
                    
                    # Send back result
                    await manager.send_message(
                        json.dumps({"response": response}),
                        websocket
                    )
                else:
                    await manager.send_message(
                        json.dumps({"error": "Invalid data format"}),
                        websocket
                    )
            
            except Exception as e:
                logger.error(f"Error processing WebSocket chat: {str(e)}")
                await manager.send_message(
                    json.dumps({"error": str(e)}),
                    websocket
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Run API server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 