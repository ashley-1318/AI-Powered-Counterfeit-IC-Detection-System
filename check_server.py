#!/usr/bin/env python3
"""
Diagnostic tool for CircuitCheck server connectivity
"""

import requests
import sys
import json

BASE_URL = "http://localhost:5000"
API_ENDPOINTS = [
    "/",
    "/health",
    "/api/auth/status",
    "/api/results/recent",
]

def check_server():
    """Check if the backend server is reachable"""
    print(f"Testing connection to {BASE_URL}...")
    
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"✅ Server is UP - Status: {r.status_code}")
            try:
                print(f"  Response: {r.json()}")
            except:
                print(f"  Response: {r.text[:100]}")
            return True
        else:
            print(f"❌ Server returned status code: {r.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection refused - server is not running or not reachable")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {str(e)}")
        return False

def check_endpoints():
    """Test a series of API endpoints"""
    print("\nTesting API endpoints:")
    
    for endpoint in API_ENDPOINTS:
        try:
            r = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if r.status_code == 200:
                print(f"✅ {endpoint} - Status: {r.status_code}")
            else:
                print(f"❌ {endpoint} - Status: {r.status_code}")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {str(e)}")

def main():
    if check_server():
        check_endpoints()
        print("\n💡 If browser still shows 'site can't be reached', try:")
        print("  - Ensure both backend and frontend are running")
        print("  - Check if you're accessing the correct port (backend:5000, frontend:3000)")
        print("  - Try using 'localhost' instead of '127.0.0.1' or vice versa")
        print("  - Use the start_project.bat script to launch both servers")
    else:
        print("\n💡 To fix backend connection issues:")
        print("  1. Ensure you're in the right directory")
        print("  2. Start the backend: cd backend && python app.py")
        print("  3. Start the frontend: cd frontend && npm start")
        print("  4. Or use the start_project.bat script to launch both")
        
if __name__ == "__main__":
    main()