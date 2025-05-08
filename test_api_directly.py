import requests
import base64
import cv2
import sys
import numpy as np

def encode_image_to_base64(image_path):
    """Encode an image file to base64 string with proper format prefix"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"

def capture_webcam_image():
    """Capture an image from webcam and encode it to base64"""
    print("Capturing image from webcam...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture image from webcam!")
        return None
    
    # Save the captured image temporarily
    temp_image_path = "temp_capture.jpg"
    cv2.imwrite(temp_image_path, frame)
    
    # Encode the image
    encoded_image = encode_image_to_base64(temp_image_path)
    return encoded_image

def test_register_face(name="test_user", image_path=None, force_register=True, use_nodejs=True):
    """Test the register-face API endpoint"""
    print("\n===== Testing Face Registration API =====")
    
    # Get image data
    if image_path:
        print(f"Using image from file: {image_path}")
        frame_data = encode_image_to_base64(image_path)
    else:
        print("Capturing image from webcam...")
        frame_data = capture_webcam_image()
        if not frame_data:
            print("Failed to capture image!")
            return False
    
    # Prepare the API payload
    payload = {
        "name": name,
        "frame_data": frame_data,
        "force_register": force_register
    }
    
    # Determine endpoint based on whether to use Node.js or Python directly
    if use_nodejs:
        endpoint = "http://localhost:3002/api/register-face"
        print("\nUsing Node.js server as intermediary")
    else:
        endpoint = "http://localhost:8001/register-face"
        print("\nConnecting directly to Python backend")
    
    # Log request details
    print(f"\nSending request to API:")
    print(f"Endpoint: {endpoint}")
    print(f"Method: POST")
    print(f"Payload:")
    print(f"  name: {name}")
    print(f"  frame_data: [base64 data - first 20 chars: {frame_data[:20]}...]")
    print(f"  force_register: {force_register}")
    
    # Send the API request
    try:
        response = requests.post(
            endpoint, 
            json=payload,
            timeout=30  # Longer timeout to allow for processing
        )
        
        # Print the response
        print(f"\nResponse status code: {response.status_code}")
        print("Response content:")
        print(response.text)
        
        # Interpret the response
        if response.status_code == 200:
            print("\n✅ API request successful!")
            return True
        else:
            print(f"\n❌ API request failed with status code {response.status_code}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error sending API request: {str(e)}")
        return False

if __name__ == "__main__":
    print("Katomaran Face Recognition API Test Tool")
    print("=======================================")
    
    # Get user preference for direct or Node.js-proxied connection
    use_nodejs = input("Use Node.js server as intermediary? (y/n, default: y): ").strip().lower()
    use_nodejs = use_nodejs != 'n'  # Default to True unless explicitly 'n'
    
    # Check if an image path was provided as a command line argument
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Get user name
    name = input("Enter name for registration (default: 'test_user'): ").strip()
    if not name:
        name = "test_user"
    
    # Run the test
    success = test_register_face(name=name, image_path=image_path, force_register=True, use_nodejs=use_nodejs)
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed. See error messages above.")
    
    print("\nPress Enter to exit...")
    input() 