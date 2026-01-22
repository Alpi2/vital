"""
MediaPipe Hands Integration for Gesture Control
Real-time hand tracking at 30 FPS for touchless operation
"""

import mediapipe as mp
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging
from dataclasses import dataclass

@dataclass
class HandLandmark:
    x: float
    y: float
    z: float
    visibility: float

class HandTracker:
    """
    Real-time hand tracking using MediaPipe
    Supports touchless interaction for sterile environments
    """
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.logger = logging.getLogger(__name__)
        self.frame_count = 0
        self.fps = 0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        Process video frame and detect hands
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (annotated_frame, hand_landmarks_list)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        annotated_frame = frame.copy()
        hand_data = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark data
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                hand_data.append({
                    'landmarks': landmarks,
                    'handedness': self._get_handedness(results, len(hand_data))
                })
        
        self.frame_count += 1
        return annotated_frame, hand_data if hand_data else None
    
    def _get_handedness(self, results, index: int) -> str:
        """Determine if hand is left or right"""
        if results.multi_handedness and index < len(results.multi_handedness):
            return results.multi_handedness[index].classification[0].label
        return 'Unknown'
    
    def get_finger_states(self, landmarks: List[Dict]) -> Dict[str, bool]:
        """
        Determine which fingers are extended
        
        Returns:
            Dictionary with finger states (True = extended, False = folded)
        """
        if len(landmarks) < 21:
            return {}
        
        # Finger tip and pip landmark indices
        finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        finger_pips = {
            'thumb': 3,
            'index': 6,
            'middle': 10,
            'ring': 14,
            'pinky': 18
        }
        
        states = {}
        
        # Check each finger
        for finger, tip_idx in finger_tips.items():
            pip_idx = finger_pips[finger]
            
            if finger == 'thumb':
                # Thumb uses x-coordinate
                states[finger] = landmarks[tip_idx]['x'] < landmarks[pip_idx]['x']
            else:
                # Other fingers use y-coordinate
                states[finger] = landmarks[tip_idx]['y'] < landmarks[pip_idx]['y']
        
        return states
    
    def calculate_distance(self, p1: Dict, p2: Dict) -> float:
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt(
            (p1['x'] - p2['x'])**2 + 
            (p1['y'] - p2['y'])**2 + 
            (p1['z'] - p2['z'])**2
        )
    
    def close(self):
        """Release resources"""
        self.hands.close()
        self.logger.info("Hand tracker closed")

# Performance optimization notes:
# 1. Use GPU acceleration if available
# 2. Reduce frame resolution for faster processing
# 3. Skip frames if processing falls behind
# 4. Use threading for camera capture and processing
