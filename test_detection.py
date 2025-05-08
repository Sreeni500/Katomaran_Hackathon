import cv2
import os
import time
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('face_detection_test')

# Directory to store test images
TEST_DIR = "test_images"
os.makedirs(TEST_DIR, exist_ok=True)

def test_face_detection():
    print("\nFace Detection Test Starting...")
    print("===============================")
    
    # Try to capture image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("Failed to capture image from webcam!")
        return False
    
    # Save the test image
    timestamp = int(time.time())
    test_image_path = os.path.join(TEST_DIR, f"test_image_{timestamp}.jpg")
    cv2.imwrite(test_image_path, frame)
    print(f"Saved test image to {test_image_path}")
    
    # Pre-process the image to improve detection
    # Resize if too large to optimize performance
    max_dim = 1024
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        print(f"Resized image to {frame.shape[:2]}")
    
    # Test different detection backends
    backends = ['retinaface', 'mtcnn', 'opencv', 'ssd']
    results = {}
    face_detected = False
    
    print("\nTesting face detection with different backends:")
    print("----------------------------------------------")
    
    for backend in backends:
        try:
            print(f"\nTrying {backend} backend...")
            start_time = time.time()
            
            face_analysis = DeepFace.analyze(
                frame, 
                actions=['detection'], 
                detector_backend=backend,
                enforce_detection=False
            )
            
            elapsed_time = time.time() - start_time
            
            if face_analysis and len(face_analysis) > 0:
                if 'facial_area' in face_analysis[0]:
                    area = face_analysis[0]['facial_area']
                    results[backend] = {
                        "status": "SUCCESS",
                        "time": elapsed_time,
                        "area": area
                    }
                    face_detected = True
                    print(f"✅ SUCCESS: Face detected with {backend} backend")
                    print(f"   Detection time: {elapsed_time:.2f} seconds")
                    print(f"   Face area: {area}")
                    
                    # Extract the face region for saving
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
                        face_path = os.path.join(TEST_DIR, f"face_{backend}_{timestamp}.jpg")
                        cv2.imwrite(face_path, face_region)
                        print(f"   Saved detected face region as {face_path}")
                else:
                    results[backend] = {
                        "status": "PARTIAL",
                        "time": elapsed_time,
                        "error": "No facial_area in response"
                    }
                    print(f"⚠️ PARTIAL: {backend} returned a response but no facial area")
            else:
                results[backend] = {
                    "status": "FAILED",
                    "time": elapsed_time,
                    "error": "No faces detected"
                }
                print(f"❌ FAILED: No faces detected with {backend} backend")
                
        except Exception as e:
            results[backend] = {
                "status": "ERROR",
                "error": str(e)
            }
            print(f"❌ ERROR: {backend} backend failed with error: {str(e)}")
    
    # Fallback to MediaPipe face detection if DeepFace fails
    if not face_detected:
        try:
            print("\nTrying MediaPipe face detection as fallback...")
            mp_face_detection = mp.solutions.face_detection
            
            with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                start_time = time.time()
                mp_results = face_detection.process(rgb_frame)
                elapsed_time = time.time() - start_time
                
                if mp_results.detections:
                    for detection in mp_results.detections:
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
                            print(f"✅ SUCCESS: Face detected with MediaPipe")
                            print(f"   Detection time: {elapsed_time:.2f} seconds")
                            area = {'x': x, 'y': y, 'w': width, 'h': height}
                            print(f"   Face area: {area}")
                            
                            # Save the face region
                            face_path = os.path.join(TEST_DIR, f"face_mediapipe_{timestamp}.jpg")
                            cv2.imwrite(face_path, face_region)
                            print(f"   Saved detected face region as {face_path}")
                            
                            # Save to results dictionary for consistent handling
                            results["mediapipe"] = {
                                "status": "SUCCESS",
                                "time": elapsed_time,
                                "area": area
                            }
                            break
                else:
                    print(f"❌ FAILED: No faces detected with MediaPipe backend")
        except Exception as e:
            print(f"❌ ERROR: MediaPipe face detection failed with error: {str(e)}")
    
    # Print summary
    print("\nDetection Results Summary:")
    print("=========================")
    
    successful_backends = [b for b in backends if results.get(b, {}).get("status") == "SUCCESS"]
    
    if face_detected:
        if successful_backends:
            print(f"✅ Face detected successfully with: {', '.join(successful_backends)}")
            print(f"✅ Recommended backend: {successful_backends[0]}")
        else:
            print(f"✅ Face detected successfully with MediaPipe fallback method")
        return True
    else:
        print("❌ No faces were detected.")
        print("Recommendations:")
        print("1. Ensure there is good lighting on your face")
        print("2. Position your face clearly in the center of the camera")
        print("3. Make sure there are no obstructions (hair, accessories, etc.)")
        return False

if __name__ == "__main__":
    test_face_detection()
    print("\nTest completed. Press Enter to continue...")
    input() 