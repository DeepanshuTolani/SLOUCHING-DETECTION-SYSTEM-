"""
Pose Detector Utility
Handles MediaPipe pose detection and feature extraction for posture analysis
"""

import cv2
import mediapipe as mp
import numpy as np
import math

class PoseDetector:
    def __init__(self):
        """Initialize MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define key landmarks for posture analysis
        self.landmarks_of_interest = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST
        ]
    
    def detect_pose(self, frame):
        """Detect pose landmarks in the frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            return results.pose_landmarks
        return None
    
    def extract_posture_features(self, landmarks):
        """Extract posture-related features from landmarks"""
        features = []
        
        # Get landmark coordinates
        h, w, _ = (480, 640, 3)  # Default frame size
        
        # Extract key points
        nose = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.NOSE, h, w)
        left_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, h, w)
        right_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, h, w)
        left_ear = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_EAR, h, w)
        right_ear = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_EAR, h, w)
        left_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, h, w)
        right_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, h, w)
        
        if all([nose, left_shoulder, right_shoulder, left_ear, right_ear, left_hip, right_hip]):
            # Calculate posture features
            
            # 1. Head position relative to shoulders
            head_center = ((left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2)
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
            
            # Head forward distance (slouching indicator)
            head_forward = head_center[0] - shoulder_center[0]
            features.append(head_forward / w)  # Normalize by frame width
            
            # 2. Shoulder angle (should be horizontal)
            shoulder_angle = math.atan2(right_shoulder[1] - left_shoulder[1], 
                                      right_shoulder[0] - left_shoulder[0])
            features.append(shoulder_angle)
            
            # 3. Head tilt
            head_tilt = math.atan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
            features.append(head_tilt)
            
            # 4. Torso angle (shoulder to hip)
            torso_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
            torso_angle = math.atan2(torso_center[1] - shoulder_center[1],
                                   torso_center[0] - shoulder_center[0])
            features.append(torso_angle)
            
            # 5. Head height relative to shoulders
            head_height_ratio = (shoulder_center[1] - head_center[1]) / h
            features.append(head_height_ratio)
            
            # 6. Shoulder width ratio
            shoulder_width = math.sqrt((right_shoulder[0] - left_shoulder[0])**2 + 
                                     (right_shoulder[1] - left_shoulder[1])**2)
            shoulder_width_ratio = shoulder_width / w
            features.append(shoulder_width_ratio)
            
            # 7. Torso length ratio
            torso_length = math.sqrt((torso_center[0] - shoulder_center[0])**2 + 
                                   (torso_center[1] - shoulder_center[1])**2)
            torso_length_ratio = torso_length / h
            features.append(torso_length_ratio)
            
            # 8. Head-shoulder distance
            head_shoulder_dist = math.sqrt((head_center[0] - shoulder_center[0])**2 + 
                                         (head_center[1] - shoulder_center[1])**2)
            head_shoulder_ratio = head_shoulder_dist / h
            features.append(head_shoulder_ratio)
            
            return np.array(features)
        
        return None
    
    def _get_landmark_coords(self, landmarks, landmark_idx, h, w):
        """Get normalized coordinates of a landmark"""
        landmark = landmarks.landmark[landmark_idx]
        if landmark.visibility > 0.5:  # Only use visible landmarks
            return (int(landmark.x * w), int(landmark.y * h))
        return None
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on the frame"""
        # Draw the pose landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Draw additional posture indicators
        h, w, _ = frame.shape
        
        # Get key points for visualization
        nose = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.NOSE, h, w)
        left_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, h, w)
        right_shoulder = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, h, w)
        left_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.LEFT_HIP, h, w)
        right_hip = self._get_landmark_coords(landmarks, self.mp_pose.PoseLandmark.RIGHT_HIP, h, w)
        
        if all([nose, left_shoulder, right_shoulder, left_hip, right_hip]):
            # Draw posture reference lines
            # Shoulder line
            cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
            
            # Torso line
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                             (left_shoulder[1] + right_shoulder[1]) // 2)
            hip_center = ((left_hip[0] + right_hip[0]) // 2, 
                         (left_hip[1] + right_hip[1]) // 2)
            cv2.line(frame, shoulder_center, hip_center, (255, 0, 0), 2)
            
            # Head position indicator
            cv2.circle(frame, nose, 5, (0, 0, 255), -1)
        
        return frame
    
    def release(self):
        """Release resources"""
        if self.pose:
            self.pose.close()
