import os
import sys
import cv2
import time
from backend.face_recognition.improved_detection import register_face, recognize_face_from_frame, detect_face

def main():
    print("\n=== Katomaran Face Recognition Test ===")
    print("1. Test face detection")
    print("2. Register a new face")
    print("3. Recognize faces")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        test_detection()
    elif choice == '2':
        register_new_face()
    elif choice == '3':
        recognize_faces()
    elif choice == '4':
        sys.exit(0)
    else:
        print("Invalid choice. Please select 1-4.")
    
    # Call main again for continuous operation
    main()

def test_detection():
    print("\n=== Testing Face Detection ===")
    
    # Create test_images directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture frame or 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam")
            break
        
        # Display the frame
        cv2.imshow("Test Detection - Press 'c' to capture", frame)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
    
    if not ret or frame is None:
        print("No frame captured")
        return
    
    # Save the test image
    timestamp = int(time.time())
    test_image_path = os.path.join("test_images", f"test_image_{timestamp}.jpg")
    cv2.imwrite(test_image_path, frame)
    print(f"Test image saved to {test_image_path}")
    
    # Test face detection
    print("\nDetecting faces...")
    face_detected, face_region, confidence = detect_face(frame)
    
    if face_detected:
        print(f"✅ Face detected successfully! (confidence: {confidence:.2f})")
        
        # Save the detected face
        face_path = os.path.join("test_images", f"detected_face_{timestamp}.jpg")
        cv2.imwrite(face_path, face_region)
        print(f"Detected face saved to {face_path}")
        
        # Display the detected face
        cv2.imshow("Detected Face", face_region)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ No face detected in the image.")
        print("Try again with better lighting and center your face in the frame.")
        print("You can still register using force_register=True")

def register_new_face():
    print("\n=== Register New Face ===")
    name = input("Enter name to register: ")
    
    if not name:
        print("Name cannot be empty.")
        return
    
    print("\nCapturing from webcam...")
    print("Press 'c' to capture or 'q' to quit")
    
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    frame = None
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from webcam")
            break
        
        cv2.imshow("Registration - Press 'c' to capture", img)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            frame = img.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame is None:
        print("No frame captured")
        return
    
    # Try registering without force_register first
    print("\nAttempting to register face...")
    success = register_face(name, frame, force_register=False)
    
    if not success:
        print("\n❌ Face detection failed. Do you want to force registration?")
        force = input("Force register anyway? (y/n): ").lower()
        
        if force == 'y':
            print("Forcing registration...")
            success = register_face(name, frame, force_register=True)
            
            if success:
                print(f"✅ Face registration forced successfully for {name}")
            else:
                print("❌ Registration failed even with force_register enabled")
        else:
            print("Registration cancelled")
    else:
        print(f"✅ Face registered successfully for {name}")

def recognize_faces():
    print("\n=== Recognize Faces ===")
    
    # Check if there are registered faces
    registered_dir = "registered_faces"
    if not os.path.exists(registered_dir) or not os.listdir(registered_dir):
        print("❌ No registered faces found. Please register faces first.")
        return
    
    print("\nCapturing from webcam for recognition...")
    print("Press 'c' to capture or 'q' to quit")
    
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    frame = None
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from webcam")
            break
        
        cv2.imshow("Recognition - Press 'c' to capture", img)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            frame = img.copy()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame is None:
        print("No frame captured")
        return
    
    # First try to detect a face
    print("\nDetecting face in captured image...")
    face_detected, face_region, confidence = detect_face(frame)
    
    if face_detected:
        print(f"✅ Face detected in captured image (confidence: {confidence:.2f})")
        
        # Use the detected face region for recognition
        print("\nRecognizing face...")
        name, recognition_confidence = recognize_face_from_frame(face_region)
        
        if name:
            print(f"✅ Face recognized as: {name} (confidence: {recognition_confidence:.2f})")
            
            # Display result
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(face_region, f"Recognized: {name}", (10, 30), font, 0.8, (0, 255, 0), 2)
            cv2.putText(face_region, f"Confidence: {recognition_confidence:.2f}", (10, 60), font, 0.8, (0, 255, 0), 2)
            cv2.imshow("Recognition Result", face_region)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ Face not recognized in database")
    else:
        print("❌ No face detected in the captured image")
        print("Try again with better lighting and center your face in the frame")
        
        # Try recognition on full frame as fallback
        print("\nAttempting to recognize on full frame as fallback...")
        name, recognition_confidence = recognize_face_from_frame(frame)
        
        if name:
            print(f"✅ Face recognized as: {name} (confidence: {recognition_confidence:.2f}, fallback method)")
        else:
            print("❌ Face not recognized in database")

if __name__ == "__main__":
    main() 