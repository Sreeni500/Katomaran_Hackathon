import os
import cv2
from deepface import DeepFace

# Directory to store registered faces
REGISTERED_DIR = "registered_faces"
os.makedirs(REGISTERED_DIR, exist_ok=True)

# Use Facenet to avoid TensorFlow compatibility issues
MODEL_NAME = "Facenet"

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
                    enforce_detection=True
                )
                if result["verified"] and result["distance"] < threshold:
                    print(f"[MATCH] Face matches with: {name}")
                    return name
            except Exception as e:
                print(f"[SKIP] Could not verify against {name}: {e}")
    print("[NO MATCH] No matching face found.")
    return None

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
