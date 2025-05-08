import os
import cv2
from deepface import DeepFace
import base64
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_face')

# Directory to store registered faces
REGISTERED_DIR = "registered_faces"
os.makedirs(REGISTERED_DIR, exist_ok=True)

# Use Facenet to avoid TensorFlow compatibility issues
MODEL_NAME = "Facenet"

# Decode base64 image from frontend
def decode_image(frame_data):
    """Decode base64 image data to numpy array"""
    try:
        # Handle different possible formats of the data
        if not frame_data:
            logger.error("Empty frame data received")
            return None
            
        # Extract the base64 part from the data URL if present
        if 'base64,' in frame_data:
            encoded_data = frame_data.split('base64,')[1]
        else:
            encoded_data = frame_data
            
        # Remove any whitespace or newlines that might be present
        encoded_data = encoded_data.strip()
        
        # Decode and convert to image
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image after successful base64 decode")
            return None
            
        return frame
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        return None

# Capture an image from webcam
def capture_image(name=None):
    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'c' to capture, 'q' to quit.")

    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Webcam - Press 'c' to capture", frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            captured = frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured is not None and name:
        path = os.path.join(REGISTERED_DIR, f"{name}.jpg")
        cv2.imwrite(path, captured)
        print(f"[INFO] Saved face as '{path}'")
    return captured

# Recognize captured face using DeepFace + Facenet
def recognize_face_from_frame(frame, threshold=0.4):
    for file in os.listdir(REGISTERED_DIR):
        if file.endswith(".jpg"):
            registered_path = os.path.join(REGISTERED_DIR, file)
            name = os.path.splitext(file)[0]
            try:
                result = DeepFace.verify(
                    frame,
                    registered_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )
                if result["verified"] and result["distance"] < threshold:
                    print(f"[MATCH] Face matches with: {name}")
                    return name
            except Exception as e:
                print(f"[SKIP] Could not verify against {name}: {e}")
    print("[NO MATCH] No matching face found.")
    return None

# Simple function to register a face
def register_face(name, frame_data, force_register=True):
    """Register a face with the given name
    
    Args:
        name: Name to associate with the face
        frame_data: Base64 encoded image data
        force_register: Whether to force registration even if face detection fails
        
    Returns:
        bool: Success status
    """
    try:
        # Decode the image
        frame = decode_image(frame_data)
        if frame is None:
            return {"success": False, "message": "Failed to decode image data"}
        
        # Save the face image directly without any detection
        image_filename = f"{name}.jpg"
        image_path = os.path.join(REGISTERED_DIR, image_filename)
        
        # Save the image
        cv2.imwrite(image_path, frame)
        logger.info(f"Saved face as '{image_path}'")
        print(f"[INFO] Registered face for {name} at '{image_path}'")
        
        return {"success": True, "message": f"Face registered for {name}"}
    
    except Exception as e:
        logger.error(f"Error registering face: {str(e)}")
        print(f"[ERROR] Failed to register face: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}

# Simple function to recognize a face
def recognize_face(frame_data):
    """Recognize faces in a frame
    
    Args:
        frame_data: Base64 encoded image data
        
    Returns:
        dict: Recognition results
    """
    try:
        # Decode the image
        frame = decode_image(frame_data)
        if frame is None:
            return {"success": False, "message": "Failed to decode image data", "faces": []}
        
        # Recognize face
        name = recognize_face_from_frame(frame)
        
        if name:
            # Create response with recognized face
            faces = [{
                "label": name,
                "registered": True,
                "confidence": 0.95,
                "facial_area": {
                    "x": 0,
                    "y": 0,
                    "w": 100,
                    "h": 100
                }
            }]
            
            return {
                "success": True,
                "message": f"Face recognized as {name}",
                "faces": faces
            }
        else:
            # No face recognized
            return {
                "success": False,
                "message": "No face recognized",
                "faces": []
            }
    
    except Exception as e:
        logger.error(f"Error recognizing face: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "faces": []
        }

# --- Main Flow ---
if __name__ == "__main__":
    print("1: Register a new face")
    print("2: Recognize a face from webcam")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        name = input("Enter name to register: ")
        capture_image(name=name)
    elif choice == "2":
        print("[INFO] Starting webcam for recognition...")
        frame = capture_image()
        if frame is not None:
            recognize_face_from_frame(frame)
    else:
        print("Invalid choice.") 