"""
Slouching Detection System - Main Flask Application

Advanced Computer Vision and Machine Learning Implementation
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import json
import time
from datetime import datetime
import threading
from utils.pose_detector import PoseDetector
from utils.slouch_classifier import SlouchClassifier
from utils.data_logger import DataLogger

app = Flask(__name__)
CORS(app)

# Initialize components
pose_detector = PoseDetector()
slouch_classifier = SlouchClassifier()
data_logger = DataLogger()

# Global variables for real-time processing
camera = None
is_detecting = False
current_posture = "Unknown"
confidence_score = 0.0
session_data = {
    'start_time': None,
    'total_detections': 0,
    'slouch_count': 0,
    'good_posture_count': 0,
    'alerts': []
}

def generate_frames():
    """Generate video frames with real-time slouching detection"""
    global camera, is_detecting, current_posture, confidence_score
    
    # Initialize camera if not already done
    if camera is None:
        print("Initializing camera...")
        camera = cv2.VideoCapture(0)
        
        # Try different camera backends
        if not camera.isOpened():
            print("Camera 0 failed, trying alternative cameras...")
            # Try alternative camera indices
            for i in range(1, 5):
                camera = cv2.VideoCapture(i)
                if camera.isOpened():
                    print(f"Camera {i} opened successfully")
                    break
        
        if camera.isOpened():
            # Set camera properties
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time
            print("Camera initialized successfully")
        else:
            print("No camera available, showing placeholder")
            # Create a placeholder frame if camera fails
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera not available", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, "Please check camera connection", (150, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
    
    while True:
        if not is_detecting:
            # Show placeholder when not detecting
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Click 'Start Detection' to begin", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
            continue
            
        success, frame = camera.read()
        if not success:
            # Create error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Camera error - Please check connection", (100, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect pose and classify posture
        pose_landmarks = pose_detector.detect_pose(frame)
        
        if pose_landmarks is not None:
            # Extract features for classification
            features = pose_detector.extract_posture_features(pose_landmarks)
            
            if features is not None:
                # Classify posture
                prediction, confidence = slouch_classifier.predict(features)
                current_posture = "Slouching" if prediction == 1 else "Good Posture"
                confidence_score = confidence
                
                # Log data
                session_data['total_detections'] += 1
                if prediction == 1:
                    session_data['slouch_count'] += 1
                    # Add alert if slouching detected
                    if len(session_data['alerts']) < 10:  # Keep last 10 alerts
                        session_data['alerts'].append({
                            'timestamp': datetime.now().strftime('%H:%M:%S'),
                            'message': 'Slouching detected! Please sit up straight.'
                        })
                else:
                    session_data['good_posture_count'] += 1
                
                # Draw pose landmarks and posture status
                frame = pose_detector.draw_landmarks(frame, pose_landmarks)
                frame = draw_posture_status(frame, current_posture, confidence_score)
            else:
                # No features extracted - draw instruction
                cv2.putText(frame, "Please position yourself in frame", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # No pose detected - draw instruction
            cv2.putText(frame, "No person detected - Please sit in frame", (120, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def draw_posture_status(frame, posture, confidence):
    """Draw posture status and confidence on frame"""
    # Background rectangle for status
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
    
    # Posture text
    color = (0, 255, 0) if posture == "Good Posture" else (0, 0, 255)
    cv2.putText(frame, f"Posture: {posture}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Confidence text
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Session stats
    cv2.putText(frame, f"Session: {session_data['total_detections']} detections", (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    """Start real-time detection"""
    global is_detecting, session_data
    
    if not is_detecting:
        is_detecting = True
        session_data['start_time'] = datetime.now()
        session_data['total_detections'] = 0
        session_data['slouch_count'] = 0
        session_data['good_posture_count'] = 0
        session_data['alerts'] = []
        
        return jsonify({'status': 'success', 'message': 'Detection started'})
    return jsonify({'status': 'error', 'message': 'Detection already running'})

@app.route('/stop_detection')
def stop_detection():
    """Stop real-time detection"""
    global is_detecting, camera
    
    if is_detecting:
        is_detecting = False
        if camera:
            camera.release()
            camera = None
        
        # Save session data
        if session_data['start_time']:
            session_data['end_time'] = datetime.now()
            session_data['duration'] = (session_data['end_time'] - session_data['start_time']).total_seconds()
            data_logger.log_session(session_data)
        
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    return jsonify({'status': 'error', 'message': 'Detection not running'})

@app.route('/get_status')
def get_status():
    """Get current detection status and statistics"""
    global current_posture, confidence_score
    
    return jsonify({
        'is_detecting': is_detecting,
        'current_posture': current_posture,
        'confidence': confidence_score,
        'session_data': session_data
    })

@app.route('/get_analytics')
def get_analytics():
    """Get analytics data"""
    analytics = data_logger.get_analytics()
    return jsonify(analytics)

@app.route('/export_data')
def export_data():
    """Export session data"""
    data = data_logger.export_all_data()
    return jsonify(data)

@app.route('/camera_status')
def camera_status():
    """Check camera status"""
    global camera
    
    if camera is None:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Camera not initialized'
        })
    
    if camera.isOpened():
        return jsonify({
            'status': 'connected',
            'message': 'Camera is connected and working'
        })
    else:
        return jsonify({
            'status': 'disconnected',
            'message': 'Camera is not connected'
        })

if __name__ == '__main__':
    # Load pre-trained model
    slouch_classifier.load_model('models/slouch_classifier.pkl')
    
    print("=" * 60)
    print("SLOUCHING DETECTION SYSTEM - TEAM 4")
    print("Final Year Project - Advanced Computer Vision & ML")
    print("=" * 60)
    print("Starting Flask application...")
    print("Access the system at: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
