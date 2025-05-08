import cv2
import sqlite3
import time
import json
import base64
import os
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_registration')

# Directory to store registered faces
REGISTERED_DIR = "registered_faces"
os.makedirs(REGISTERED_DIR, exist_ok=True)

# Use Facenet model to avoid TensorFlow compatibility issues
MODEL_NAME = "Facenet"

def register_face(name, frame_data=None, save_to_db=True, force_register=False):
    """
    Register a face with the given name
    
    Args:
        name (str): Name of the person
        frame_data (str, optional): Base64 encoded frame data from frontend
        save_to_db (bool): Whether to save to database or return encoding
        force_register (bool): Whether to force registration even if face detection fails
        
    Returns:
        bool: Success status
        dict: Face data if save_to_db is False
    """
    try:
        logger.info(f"Starting face registration for {name}")
        
        # Create database directory if it doesn't exist
        os.makedirs('database', exist_ok=True)
        
        # Connect to SQLite database
        conn = sqlite3.connect('database/faces.db')
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding TEXT,
            timestamp TEXT
        )
        """)
        
        # Check if the image_path column exists, add it if it doesn't
        cursor.execute("PRAGMA table_info(faces)")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        
        if 'image_path' not in column_names:
            logger.info("Adding image_path column to faces table")
            cursor.execute("ALTER TABLE faces ADD COLUMN image_path TEXT")
            conn.commit()
        
        # Get frame data
        if frame_data:
            # Decode base64 frame data from frontend
            try:
                encoded_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                logger.info(f"Successfully decoded image data, shape: {frame.shape if frame is not None else 'None'}")
            except Exception as e:
                logger.error(f"Error decoding frame data: {str(e)}")
                return False
        else:
            # Capture from webcam if no frame data provided
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture from webcam")
                return False
            cap.release()
        
        # Check if image has content
        if frame is None or frame.size == 0:
            logger.error("Invalid frame data - empty frame")
            return False
            
        # Save debug image to see what's being processed
        debug_path = os.path.join(REGISTERED_DIR, f"debug_{name}_{int(time.time())}.jpg")
        cv2.imwrite(debug_path, frame)
        logger.info(f"Saved debug image as {debug_path}")
        
        # Pre-process the image to improve detection
        # Resize if too large to optimize performance
        max_dim = 1024
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            logger.info(f"Resized image to {frame.shape[:2]}")
        
        # Try to detect face with DeepFace - try different backends if one fails
        backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']  # Reordered to try more accurate methods first
        face_detected = False
        detection_error = None
        
        for backend in backends:
            try:
                logger.info(f"Attempting face detection with {backend} backend")
                # Using analyze with detection action to check if a face is present
                face_analysis = DeepFace.analyze(
                    frame, 
                    actions=['detection'], 
                    detector_backend=backend,
                    enforce_detection=False
                )
                
                if face_analysis and len(face_analysis) > 0:
                    if 'facial_area' in face_analysis[0]:
                        face_detected = True
                        logger.info(f"Face detected successfully with {backend} backend")
                        
                        # Extract the face region for saving
                        area = face_analysis[0]['facial_area']
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
                        face_region = frame[y1:y2, x1:x2]
                        
                        if face_region.size > 0:
                            # Save the face region for better quality
                            debug_face_path = os.path.join(REGISTERED_DIR, f"debug_face_{name}_{int(time.time())}.jpg")
                            cv2.imwrite(debug_face_path, face_region)
                            logger.info(f"Saved detected face region as {debug_face_path}")
                            
                            # Use the face region instead of full frame
                            frame = face_region
                        
                        break
                    else:
                        logger.warning(f"Backend {backend} did not return facial_area")
                else:
                    logger.warning(f"No faces detected with {backend} backend")
                
            except Exception as e:
                detection_error = str(e)
                logger.error(f"Face detection error with {backend} backend: {str(e)}")
                continue  # Try next backend
        
        # Fallback to MediaPipe face detection if DeepFace fails
        if not face_detected:
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
                                face_detected = True
                                logger.info("Face detected successfully with MediaPipe")
                                
                                # Save the face region
                                debug_face_path = os.path.join(REGISTERED_DIR, f"debug_mediapipe_{name}_{int(time.time())}.jpg")
                                cv2.imwrite(debug_face_path, face_region)
                                
                                # Use the face region
                                frame = face_region
                                break
            except Exception as e:
                logger.error(f"MediaPipe face detection error: {str(e)}")
        
        # Force registration even if no face is detected (if explicitly requested by frontend)
        if not face_detected and not force_register:
            logger.error(f"No face detected in the image after trying all methods. Last error: {detection_error}")
            return {"success": False, "message": "Failed to recognize face. Please position your face clearly in the frame and ensure good lighting."}
        elif not face_detected and force_register:
            logger.warning("No face detected in the image after trying all methods, but force_register is enabled, so proceeding with registration anyway")
            # We'll register the full frame, but log the issue
        
        # At this point, we have a valid face in the frame (or at least the best we could get)
        # Save the image to the registered faces directory using exact same format as reference code
        image_filename = f"{name}.jpg"
        image_path = os.path.join(REGISTERED_DIR, image_filename)
        
        # Check if file already exists, append timestamp if it does
        if os.path.exists(image_path):
            timestamp = int(time.time())
            image_filename = f"{name}_{timestamp}.jpg"
            image_path = os.path.join(REGISTERED_DIR, image_filename)
            
        # Save the image (either face region or full frame)
        cv2.imwrite(image_path, frame)
        logger.info(f"Saved face as '{image_path}'")
        
        # Save to database
        if save_to_db:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            # Use NULL for encoding since we're storing the image path instead
            cursor.execute("INSERT INTO faces (name, encoding, image_path, timestamp) VALUES (?, ?, ?, ?)",
                        (name, None, image_path, timestamp))
            conn.commit()
            
            # Create logs DB and table if they don't exist
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
            
            # Log face registration
            log_conn = sqlite3.connect('database/logs.db')
            log_cursor = log_conn.cursor()
            log_cursor.execute("INSERT INTO face_logs (action, name, timestamp, details) VALUES (?, ?, ?, ?)",
                            ("registration", name, timestamp, f"Image saved at {image_path}, face_detected={face_detected}"))
            log_conn.commit()
            log_conn.close()
            
            logger.info(f"Face registered successfully for {name}")
            return True
        else:
            # For non-database operations, just return the path and success
            return {'name': name, 'image_path': image_path}
        
    except Exception as e:
        logger.error(f"Error registering face: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        if 'conn' in locals():
            conn.close()

def verify_face(face_img, reference_img, threshold=0.4):
    """
    Verify if two face images belong to the same person
    
    Args:
        face_img: First face image (numpy array or path)
        reference_img: Second face image (numpy array or path)
        threshold: Verification threshold (lower = stricter)
        
    Returns:
        bool: True if same person, False otherwise
    """
    try:
        result = DeepFace.verify(
            face_img, 
            reference_img,
            model_name=MODEL_NAME,
            enforce_detection=False,
            distance_metric='cosine'
        )
        
        if result["verified"] and result["distance"] < threshold:
            return True, result["distance"]
        return False, result["distance"]
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return False, 1.0

if __name__ == "__main__":
    # For testing from command line
    # Simple test using the reference code approach
    print("1: Register a new face")
    print("2: Recognize a face from webcam")
    choice = input("Enter 1 or 2: ")
    
    if choice == "1":
        name = input("Enter name to register: ")
        result = register_face(name)
        print(f"Registration result: {result}")
    elif choice == "2":
        # Test recognition
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture from webcam")
        else:
            # Check each saved face
            for file in os.listdir(REGISTERED_DIR):
                if file.endswith(".jpg"):
                    registered_path = os.path.join(REGISTERED_DIR, file)
                    name = os.path.splitext(file)[0]
                    matched, distance = verify_face(frame, registered_path)
                    if matched:
                        print(f"[MATCH] Face matches with: {name}")
                    else:
                        print(f"[NO MATCH] Face doesn't match with: {name}")
    else:
        print("Invalid choice.") 