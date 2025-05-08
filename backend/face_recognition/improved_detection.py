import os
import cv2
from deepface import DeepFace
import logging
import mediapipe as mp
import numpy as np
import time
import base64

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('improved_face_detection')

# Directory to store registered faces
REGISTERED_DIR = "registered_faces"
os.makedirs(REGISTERED_DIR, exist_ok=True)

# Use Facenet to avoid TensorFlow compatibility issues
MODEL_NAME = "Facenet"

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

def detect_face(frame, return_largest=True, check_only=False):
    """
    Detect faces in an image frame using multiple backends
    
    Args:
        frame: Image frame to detect faces in
        return_largest: Whether to return only the largest face
        check_only: If True, only perform detection without recognition
        
    Returns:
        tuple: (face_detected, face_region, confidence) where face_region is the cropped face image
    """
    if frame is None:
        logger.error("Invalid frame input")
        return False, None, 0
    
    # Pre-process the image to improve detection
    # Resize if too large to optimize performance
    max_dim = 1024
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        logger.info(f"Resized image to {frame.shape[:2]}")
    
    # Try MediaPipe first as it's faster and more reliable
    try:
        logger.info("Attempting MediaPipe face detection")
        mp_face_detection = mp.solutions.face_detection
        
        with mp_face_detection.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1  # Use the full-range model
        ) as face_detection:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                # Get the detection with highest confidence
                best_detection = max(results.detections, 
                                   key=lambda x: x.score[0])
                confidence = best_detection.score[0]
                
                bbox = best_detection.location_data.relative_bounding_box
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
                    logger.info(f"Face detected with MediaPipe, confidence: {confidence:.2f}")
                    return True, face_region, confidence
    
    except Exception as e:
        logger.error(f"MediaPipe face detection error: {str(e)}")
    
    # If MediaPipe fails, try DeepFace backends
    backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
    
    for backend in backends:
        try:
            logger.info(f"Attempting face detection with {backend} backend")
            face_analysis = DeepFace.analyze(
                frame, 
                actions=['detection'], 
                detector_backend=backend,
                enforce_detection=False
            )
            
            if face_analysis and len(face_analysis) > 0:
                if 'facial_area' in face_analysis[0]:
                    area = face_analysis[0]['facial_area']
                    x, y, w, h = area['x'], area['y'], area['w'], area['h']
                    
                    # Add padding (20%)
                    padding_x = int(w * 0.2)
                    padding_y = int(h * 0.2)
                    
                    # Make sure we don't go beyond image boundaries
                    frame_h, frame_w = frame.shape[:2]
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(frame_w, x + w + padding_x)
                    y2 = min(frame_h, y + h + padding_y)
                    
                    # Extract face region
                    face_region = frame[y1:y2, x1:x2]
                    
                    if face_region.size > 0:
                        logger.info(f"Face detected with {backend} backend")
                        return True, face_region, 0.8  # Assume 0.8 confidence for DeepFace
        
        except Exception as e:
            logger.error(f"Error with {backend} backend: {str(e)}")
            continue
    
    logger.warning("No face detected with any backend")
    return False, None, 0

def recognize_face_from_frame(frame, threshold=0.4):
    """
    Recognize a face from a frame by comparing with registered faces
    
    Args:
        frame: Image frame containing a face
        threshold: Similarity threshold (lower = stricter)
        
    Returns:
        tuple: (name, confidence) of the recognized person or (None, 0)
    """
    # First detect the face
    face_detected, face_region, detection_confidence = detect_face(frame)
    
    if not face_detected or face_region is None:
        logger.warning("No face detected in frame")
        return None, 0
    
    matched_names = []
    
    for file in os.listdir(REGISTERED_DIR):
        if file.endswith((".jpg", ".jpeg", ".png")):
            registered_path = os.path.join(REGISTERED_DIR, file)
            name = os.path.splitext(file)[0]
            
            # Handle underscore in filename (timestamp separator)
            if "_" in name:
                name = name.split("_")[0]
                
            try:
                result = DeepFace.verify(
                    face_region,  # Use the detected face region
                    registered_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                    distance_metric='cosine'
                )
                
                if result["verified"] and result["distance"] < threshold:
                    confidence = 1 - result["distance"]  # Convert distance to confidence
                    logger.info(f"Face matches with: {name}, confidence: {confidence:.2f}")
                    matched_names.append((name, confidence))
            except Exception as e:
                logger.error(f"Could not verify against {name}: {e}")
    
    if matched_names:
        # Return the best match (highest confidence)
        best_match = max(matched_names, key=lambda x: x[1])
        return best_match[0], best_match[1]
    
    logger.info("No matching face found")
    return None, 0

# Capture an image from webcam
def capture_image(name=None):
    cap = cv2.VideoCapture(0)
    logger.info("Press 'c' to capture, 'q' to quit.")
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
        logger.info(f"Saved face as '{path}'")
        print(f"[INFO] Saved face as '{path}'")
    return captured

# Register a new face
def register_face(name, frame=None, force_register=False):
    """
    Register a new face in the system
    
    Args:
        name: Name to associate with the face
        frame: Optional pre-captured frame, if None will capture from webcam
        force_register: Whether to force registration even if face detection fails
        
    Returns:
        bool: Success status
    """
    try:
        if frame is None:
            # Capture from webcam
            frame = capture_image()
            if frame is None:
                logger.error("Failed to capture image")
                return False
        
        # Try to detect face
        face_detected, face_region, detection_confidence = detect_face(frame)
        
        if face_detected:
            # Use detected face region for registration
            register_image = face_region
            logger.info("Face detected successfully")
        elif force_register:
            # Use full frame if face detection failed but force_register is True
            register_image = frame
            logger.warning("No face detected, but force_register enabled - using full image")
        else:
            logger.error("No face detected in the image")
            return False
        
        # Save the face image
        image_filename = f"{name}.jpg"
        image_path = os.path.join(REGISTERED_DIR, image_filename)
        
        # Check if file already exists, append timestamp if it does
        if os.path.exists(image_path):
            timestamp = int(time.time())
            image_filename = f"{name}_{timestamp}.jpg"
            image_path = os.path.join(REGISTERED_DIR, image_filename)
        
        # Save the image
        cv2.imwrite(image_path, register_image)
        logger.info(f"Saved face as '{image_path}'")
        print(f"[INFO] Registered face for {name} at '{image_path}'")
        
        return True
    
    except Exception as e:
        logger.error(f"Error registering face: {str(e)}")
        print(f"[ERROR] Failed to register face: {str(e)}")
        return False

# Main execution
if __name__ == "__main__":
    print("1: Register a new face")
    print("2: Recognize a face from webcam")
    print("3: Test face detection")
    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        name = input("Enter name to register: ")
        register_face(name, force_register=True)
    elif choice == "2":
        print("[INFO] Starting webcam for recognition...")
        frame = capture_image()
        if frame is not None:
            name, confidence = recognize_face_from_frame(frame)
            if name:
                print(f"[RESULT] Recognized as: {name}, confidence: {confidence:.2f}")
            else:
                print("[RESULT] Not recognized")
    elif choice == "3":
        # Test face detection
        print("[INFO] Testing face detection...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            face_detected, face_region, detection_confidence = detect_face(frame)
            if face_detected:
                print("[RESULT] Face detected successfully!")
                # Save the detected face for inspection
                debug_path = os.path.join("test_images", f"detected_face_{int(time.time())}.jpg")
                os.makedirs("test_images", exist_ok=True)
                cv2.imwrite(debug_path, face_region)
                print(f"[INFO] Saved detected face to {debug_path}")
            else:
                print("[RESULT] No face detected.")
        else:
            print("[ERROR] Failed to capture image from webcam.")
    else:
        print("Invalid choice.") 