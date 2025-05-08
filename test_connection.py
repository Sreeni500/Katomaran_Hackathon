import requests
import sys
import os
import time
import json

def test_node_backend():
    """Test connection to Node.js backend server on port 3002"""
    print("\n==== Testing Node.js Backend Connection ====")
    try:
        response = requests.get("http://localhost:3002/api", timeout=5)
        print(f"✅ Connection successful: Status {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed: Could not connect to Node.js backend server on port 3002")
        print("The Node.js server is probably not running or is using a different port")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_python_backend():
    """Test connection to Python backend server on port 8001"""
    print("\n==== Testing Python Backend Connection ====")
    try:
        response = requests.get("http://localhost:8001/", timeout=5)
        print(f"✅ Connection successful: Status {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed: Could not connect to Python backend server on port 8001")
        print("The Python server is probably not running or is using a different port")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_backend_communication():
    """Test if Node.js server can communicate with Python backend"""
    print("\n==== Testing Backend Communication ====")
    try:
        # Try to use Node.js server to proxy a request to Python backend
        response = requests.post(
            "http://localhost:3002/api/register-face", 
            json={
                "name": "test_user",
                "frame_data": "test_data",
                "force_register": True
            },
            timeout=5
        )
        print(f"Response status: {response.status_code}")
        print(f"Response data: {response.text}")
        
        if response.status_code == 200:
            print("✅ Node.js server successfully communicated with Python backend")
            return True
        else:
            print("❌ Communication test failed with status code", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed: Could not connect to Node.js server to proxy request")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def print_recommendations():
    """Print recommendations based on test results"""
    print("\n==== Recommendations ====")
    print("1. Make sure both servers are running:")
    print("   - Python backend: python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8001")
    print("   - Node.js server: cd node_ws && node server.js")
    print("\n2. Check your frontend configuration:")
    print("   - In frontend components, ensure API_URL is set to 'http://localhost:3002/api'")
    print("\n3. Check Node.js server configuration:")
    print("   - In node_ws/server.js, ensure PYTHON_API_PORT is set to 8001")
    print("\n4. Try restarting both servers")
    print("\n5. Use the 'Force Register' button which was added to bypass potential face detection issues")

if __name__ == "__main__":
    print("==== Connection Testing Tool ====")
    print("This tool will test connections between frontend, Node.js server, and Python backend")
    
    python_success = test_python_backend()
    node_success = test_node_backend()
    
    if python_success and node_success:
        comm_success = test_backend_communication()
    else:
        comm_success = False
        print("\n❌ Skipping communication test since one or both servers are not responding")
    
    print("\n==== Test Summary ====")
    print(f"Python Backend (8001): {'✅ CONNECTED' if python_success else '❌ FAILED'}")
    print(f"Node.js Server (3002): {'✅ CONNECTED' if node_success else '❌ FAILED'}")
    print(f"Backend Communication: {'✅ WORKING' if comm_success else '❌ FAILED'}")
    
    print_recommendations()
    
    print("\nPress Enter to exit...")
    input() 