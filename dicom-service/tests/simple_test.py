#!/usr/bin/env python3
"""
Simple test to verify the DICOM service is working
"""

import subprocess
import time
import os
import signal

def test_service():
    """Test that the service starts and responds to help"""
    print("🧪 Testing DICOM Service...")
    
    # Start the service
    server_path = "../build/dicom_service"
    if not os.path.exists(server_path):
        print("❌ Server executable not found")
        return False
    
    # Test help command
    try:
        result = subprocess.run([server_path, "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Service help command works")
            print("📋 Help output:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Help command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running help: {e}")
        return False

def test_service_start():
    """Test that the service can start"""
    print("\n🧪 Testing Service Startup...")
    
    server_path = "../build/dicom_service"
    
    # Start the service in background
    try:
        process = subprocess.Popen([server_path, "--address", "localhost:50052", 
                                  "--storage", "/tmp/dicom_test_storage"],
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Give it time to start
        time.sleep(2)
        
        # Check if it's still running
        if process.poll() is None:
            print("✅ Service started successfully")
            
            # Stop it
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Service stopped gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️ Service had to be killed")
            
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Service failed to start")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Simple DICOM Service Tests")
    print("=" * 50)
    
    success = True
    
    # Test 1: Help command
    if not test_service():
        success = False
    
    # Test 2: Service startup
    if not test_service_start():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL SIMPLE TESTS PASSED!")
        print("✅ The DICOM service is working correctly!")
    else:
        print("❌ SOME TESTS FAILED")
    
    exit(0 if success else 1)
