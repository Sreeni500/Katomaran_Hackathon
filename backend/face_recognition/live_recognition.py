import cv2
import sqlite3
import json
import base64
import time
import os
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('live_recognition')

# Directory to store registered faces
REGISTERED_DIR = "registered_faces"
os.makedirs(REGISTERED_DIR, exist_ok=True)

# Use Facenet model to avoid TensorFlow compatibility issues
MODEL_NAME = "Facenet"

def recognize_face(frame_data, threshold=0.4):
    """
    Recognize faces in a frame and return matches from registered faces
    
    Args:
        frame_data (str): Base64 encoded frame data
        threshold (float): Similarity threshold (lower = stricter)
        
    Returns:
        dict: Recognition results with success, message and faces
    """
    try:
        logger.info("Starting face recognition")
        
        # Decode frame data
        try:
            encoded_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info(f"Successfully decoded image data, shape: {frame.shape if frame is not None else 'None'}")
        except Exception as e:
            logger.error(f"Error decoding frame data: {str(e)}")
            return {"success": False, "message": "Invalid frame data", "faces": []}
        
        # Check if image has content
        if frame is None or frame.size == 0:
            logger.error("Invalid frame data - empty frame")
            return {"success": False, "message": "Empty frame data", "faces": []}
            
        # Save debug image to track what's being processed 
        timestamp = int(time.time())
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"recognize_frame_{timestamp}.jpg")
        cv2.imwrite(debug_path, frame)
        logger.info(f"Saved debug frame as {debug_path}")
        
        # Try to detect faces with different backends
        backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']  # Reordered to try more accurate methods first
        detected_faces = []
        detection_error = None
        
        for backend in backends:
            try:
                logger.info(f"Attempting face detection with {backend} backend")
                # Using analyze with detection action to detect faces
                face_analysis = DeepFace.analyze(
                    frame, 
                    actions=['detection'], 
                    detector_backend=backend,
                    enforce_detection=False
                )
                
                if face_analysis and len(face_analysis) > 0:
                    if 'facial_area' in face_analysis[0]:
                        # Extract detected faces
                        for face in face_analysis:
                            if 'facial_area' in face:
                                area = face['facial_area']
                                x, y, w, h = area['x'], area['y'], area['w'], area['h']
                                
                                # Add padding for better face capture (20% on each side)
                                padding_x = int(w * 0.2)
                                padding_y = int(h * 0.2)
                                
                                # Make sure we don't go beyond image boundaries
                                frame_h, frame_w = frame.shape[:2]
                                x1 = max(0, x - padding_x)
                                y1 = max(0, y - padding_y)
                                x2 = min(frame_w, x + w + padding_x)
                                y2 = min(frame_h, y + h + padding_y)
                                
                                # Extract the padded face region
                                detected_face = frame[y1:y2, x1:x2]
                                
                                if detected_face.size > 0:
                                    # Add enhanced face to detected faces
                                    detected_faces.append({
                                        "face_img": detected_face,
                                        "area": area,
                                        "backend": backend,
                                        "enhanced": True
                                    })
                        
                        logger.info(f"Detected {len(detected_faces)} faces with {backend} backend")
                        break  # Stop after first successful backend
                    else:
                        logger.warning(f"Backend {backend} did not return facial_area")
                else:
                    logger.warning(f"No faces detected with {backend} backend")
                    
            except Exception as e:
                detection_error = str(e)
                logger.error(f"Face detection error with {backend} backend: {str(e)}")
                continue  # Try next backend
        
        # Fallback to MediaPipe face detection if DeepFace fails
        if not detected_faces:
            try:
                logger.info("Attempting MediaPipe face detection as fallback")
                mp_face_detection = mp.solutions.face_detection
                
                with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
                    # Convert to RGB for MediaPipe
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            h, w, _ = frame.shape
                            
                            # Convert relative coordinates to absolute
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            
                            # Add padding (20%)
                            padding_x = int(width * 0.2)
                            padding_y = int(height * 0.2)
                            
                            # Make sure we don't go beyond image boundaries
                            x1 = max(0, x - padding_x)
                            y1 = max(0, y - padding_y)
                            x2 = min(w, x + width + padding_x)
                            y2 = min(h, y + height + padding_y)
                            
                            # Extract face region
                            face_region = frame[y1:y2, x1:x2]
                            
                            if face_region.size > 0:
                                # We found a face with MediaPipe
                                area = {'x': x, 'y': y, 'w': width, 'h': height}
                                detected_faces.append({
                                    "face_img": face_region,
                                    "area": area,
                                    "backend": "mediapipe",
                                    "enhanced": True
                                })
                                logger.info("Face detected successfully with MediaPipe")
                                
                                # Save debug image of the detected face
                                debug_dir = "debug_frames"
                                debug_face_path = os.path.join(debug_dir, f"mediapipe_face_{int(time.time())}.jpg")
                                cv2.imwrite(debug_face_path, face_region)
                                logger.info(f"Saved MediaPipe detected face as {debug_face_path}")
            except Exception as e:
                logger.error(f"MediaPipe face detection error: {str(e)}")
                
        # If we still have no faces detected, return an empty result but with success=false and error message
        if not detected_faces:
            logger.warning("No faces detected in the image after trying all methods")
            return {"success": False, "message": "Failed to recognize face", "faces": []}
            
        # Get registered faces
        registered_faces = []
        for file in os.listdir(REGISTERED_DIR):
            if file.endswith(".jpg") and not file.startswith("debug_"):
                registered_path = os.path.join(REGISTERED_DIR, file)
                name = os.path.splitext(file)[0]
                registered_faces.append({"name": name, "path": registered_path})
        
        if not registered_faces:
            logger.info("No registered faces found to compare against")
            return {"success": True, "message": "No registered faces to compare", "faces": [{"label": "Unknown", "confidence": 0, "registered": False}]}
                
        # For each detected face, compare with registered faces
        recognition_results = []
        for face_data in detected_faces:
            face_img = face_data["face_img"]
            matched = False
            
            # Check if face has enough content to match
            if face_img is None or face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                logger.warning("Detected face is too small or invalid")
                continue
                
            face_result = {
                "facial_area": face_data["area"],
                "registered": False,
                "confidence": 0
            }
            
            # Try to find a match among registered faces
            closest_match = None
            closest_distance = 1.0
            
            for reg_face in registered_faces:
                try:
                    verification = DeepFace.verify(
                        face_img, 
                        reg_face["path"],
                        model_name=MODEL_NAME,
                        enforce_detection=False,
                        distance_metric='cosine'
                    )
                    
                    distance = verification["distance"]
                    logger.info(f"Distance to {reg_face['name']}: {distance}")
                    
                    if distance < threshold and distance < closest_distance:
                        closest_distance = distance
                        closest_match = reg_face["name"]
                        matched = True
                        
                except Exception as e:
                    logger.error(f"Error comparing with {reg_face['name']}: {str(e)}")
                    continue
            
            if matched:
                # Log the recognition
                log_conn = sqlite3.connect('database/logs.db')
                log_cursor = log_conn.cursor()
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                log_cursor.execute("INSERT INTO face_logs (action, name, timestamp, details) VALUES (?, ?, ?, ?)",
                                ("recognition", closest_match, timestamp, f"Distance: {closest_distance}"))
                log_conn.commit()
                log_conn.close()
                
                # Return match info
                confidence = 1.0 - closest_distance
                face_result.update({
                    "label": closest_match,
                    "confidence": round(confidence, 3),
                    "registered": True
                })
                logger.info(f"Recognized {closest_match} with confidence {confidence}")
            else:
                face_result.update({
                    "label": "Unknown",
                    "confidence": 0,
                    "registered": False
                })
                logger.info("Face not recognized as any registered person")
                
            recognition_results.append(face_result)
            
        # Return results
        return {"success": True, "message": "Recognition completed", "faces": recognition_results}
        
    except Exception as e:
        logger.error(f"Error in recognition: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "message": str(e), "faces": []}

def recognize_face_from_image(face_img, threshold=0.4):
    """
    Recognize a face by comparing with registered faces
    
    Args:
        face_img: Face image to recognize
        threshold: Recognition threshold
        
    Returns:
        tuple: (name, confidence)
    """
    # Check if we have registered faces
    if not os.path.exists(REGISTERED_DIR) or not os.listdir(REGISTERED_DIR):
        # Try database as fallback
        registered_faces = get_registered_faces_from_db()
        if not registered_faces:
            return "Unknown", 0
    
    # First try with image files directly
    best_match = "Unknown"
    best_confidence = 0
    lowest_distance = float('inf')
    
    # Check every registered face image
    for file in os.listdir(REGISTERED_DIR):
        if file.endswith((".jpg", ".jpeg", ".png")):
            registered_path = os.path.join(REGISTERED_DIR, file)
            # Extract name from filename (remove timestamp if present)
            name_parts = os.path.splitext(file)[0].split('_')
            name = name_parts[0] if len(name_parts) > 0 else "Unknown"
            
            try:
                # Use DeepFace.verify for direct comparison - this is like the reference code
                result = DeepFace.verify(
                    face_img,
                    registered_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                    distance_metric='cosine'
                )
                
                # Calculate confidence (invert distance)
                distance = result["distance"]
                confidence = round((1 - distance) * 100, 2)
                
                # Update if this is the best match so far
                if result["verified"] and distance < lowest_distance:
                    lowest_distance = distance
                    best_match = name
                    best_confidence = confidence
            
            except Exception as e:
                print(f"Error verifying against {name}: {str(e)}")
                continue
    
    # If a match was found
    if best_match != "Unknown":
        return best_match, best_confidence
    
    # As a fallback, try the database records
    registered_faces = get_registered_faces_from_db()
    
    for name, image_path in registered_faces:
        try:
            # Verify against the stored image path
            if os.path.exists(image_path):
                result = DeepFace.verify(
                    face_img,
                    image_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                    distance_metric='cosine'
                )
                
                # Calculate confidence
                distance = result["distance"]
                confidence = round((1 - distance) * 100, 2)
                
                # Update if this is the best match so far
                if result["verified"] and distance < lowest_distance:
                    lowest_distance = distance
                    best_match = name
                    best_confidence = confidence
        
        except Exception as e:
            print(f"Error verifying against database record {name}: {str(e)}")
            continue
    
    return best_match, best_confidence

def get_registered_faces_from_db():
    """
    Get list of registered faces from database
    
    Returns:
        list: List of (name, image_path) tuples
    """
    try:
        conn = sqlite3.connect('database/faces.db')
        cursor = conn.cursor()
        
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='faces'")
        if not cursor.fetchone():
            return []
        
        # Get registered faces
        cursor.execute("SELECT name, image_path FROM faces")
        return cursor.fetchall()
    
    except Exception as e:
        print(f"Error getting faces from database: {str(e)}")
        return []
    
    finally:
        if 'conn' in locals():
            conn.close()

def log_recognition(name, confidence):
    """
    Log a face recognition event
    
    Args:
        name: Name of the recognized person
        confidence: Recognition confidence
    """
    try:
        # Create logs database if it doesn't exist
        if not os.path.exists('database/logs.db'):
            log_conn = sqlite3.connect('database/logs.db')
            log_cursor = log_conn.cursor()
            log_cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                name TEXT,
                timestamp TEXT,
                details TEXT
            )
            """)
            log_conn.commit()
            log_conn.close()
        
        # Log the recognition
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_conn = sqlite3.connect('database/logs.db')
        log_cursor = log_conn.cursor()
        log_cursor.execute(
            "INSERT INTO face_logs (action, name, timestamp, details) VALUES (?, ?, ?, ?)",
            ("recognition", name, timestamp, f"Confidence: {confidence}%")
        )
        log_conn.commit()
        log_conn.close()
    
    except Exception as e:
        print(f"Error logging recognition: {str(e)}")

if __name__ == "__main__":
    # Simple test from command line using reference code approach
    print("1: Register a new face")
    print("2: Recognize a face from webcam")
    choice = input("Enter 1 or 2: ")
    
    if choice == "2":
        print("[INFO] Starting webcam for recognition...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if frame is not None:
            result = recognize_face_from_image(frame)
            print(f"Recognition result: {result}")
    else:
        print("Use option 1 in register_face.py to register new faces") 