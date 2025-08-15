"""
Simple Camera Test Script
Tests camera initialization and basic pose detection
"""

import cv2
import numpy as np
import sys
import os

def test_camera():
    """Test camera initialization and basic functionality"""
    print("=" * 50)
    print("SIMPLE CAMERA TEST")
    print("=" * 50)
    
    # Test camera initialization
    print("1. Testing camera initialization...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("   ❌ Camera 0 failed to open")
        # Try alternative cameras
        for i in range(1, 5):
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                print(f"   ✅ Camera {i} opened successfully")
                break
        else:
            print("   ❌ No camera found")
            return False
    else:
        print("   ✅ Camera 0 opened successfully")
    
    # Set camera properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    print("2. Testing frame capture...")
    
    # Capture a few frames
    for i in range(3):
        ret, frame = camera.read()
        if ret:
            print(f"   ✅ Frame {i+1} captured successfully ({frame.shape})")
        else:
            print(f"   ❌ Frame {i+1} failed to capture")
            camera.release()
            return False
    
    print("3. Testing MediaPipe pose detection...")
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Test pose detection on a frame
        ret, frame = camera.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                print("   ✅ Pose detection working - landmarks detected")
                print(f"   📊 Number of landmarks: {len(results.pose_landmarks.landmark)}")
            else:
                print("   ⚠️  No pose detected (no person in frame)")
                
        pose.close()
        
    except ImportError as e:
        print(f"   ❌ Could not import MediaPipe: {e}")
        camera.release()
        return False
    except Exception as e:
        print(f"   ❌ Pose detection error: {e}")
        camera.release()
        return False
    
    # Clean up
    camera.release()
    
    print("=" * 50)
    print("✅ CAMERA TEST COMPLETED!")
    print("Camera is working properly.")
    print("You can now access the web interface at: http://localhost:5000")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_camera()
