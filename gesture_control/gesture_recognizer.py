"""
Gesture Recognition Engine
Recognizes gestures from hand landmarks and triggers actions
"""

import json
import time
from typing import Dict, List, Optional, Callable
import numpy as np
import logging
from collections import deque

class GestureRecognizer:
    """
    Recognizes gestures and maps them to actions
    Supports both static and dynamic gestures
    """
    
    def __init__(self, config_path: str = 'gestures/gesture_definitions.json'):
        self.logger = logging.getLogger(__name__)
        self.gestures = {}
        self.settings = {}
        self.load_config(config_path)
        
        # Gesture state tracking
        self.current_gesture = None
        self.gesture_start_time = None
        self.last_gesture_time = 0
        self.gesture_history = deque(maxlen=10)
        
        # Callbacks
        self.action_callbacks: Dict[str, Callable] = {}
        
    def load_config(self, config_path: str) -> None:
        """Load gesture definitions from JSON"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.gestures = {g['id']: g for g in config['gestures']}
            self.settings = config.get('settings', {})
            self.logger.info(f"Loaded {len(self.gestures)} gesture definitions")
        except Exception as e:
            self.logger.error(f"Failed to load gesture config: {e}")
    
    def register_action(self, action: str, callback: Callable) -> None:
        """Register callback for gesture action"""
        self.action_callbacks[action] = callback
        self.logger.info(f"Registered action: {action}")
    
    def recognize(self, hand_data: List[Dict]) -> Optional[Dict]:
        """
        Recognize gesture from hand landmarks
        
        Args:
            hand_data: List of hand landmark data
            
        Returns:
            Recognized gesture info or None
        """
        if not hand_data:
            return None
        
        current_time = time.time()
        
        # Check debounce
        debounce_time = self.settings.get('debounce_time_ms', 200) / 1000.0
        if current_time - self.last_gesture_time < debounce_time:
            return None
        
        # Try to match gestures
        for gesture_id, gesture_def in self.gestures.items():
            if self._matches_gesture(hand_data, gesture_def):
                # Check if gesture needs to be held
                hold_time = self.settings.get('gesture_hold_time_ms', 500) / 1000.0
                
                if gesture_def.get('continuous', False):
                    # Continuous gesture - trigger immediately
                    return self._trigger_gesture(gesture_id, gesture_def)
                else:
                    # Discrete gesture - check hold time
                    if self.current_gesture == gesture_id:
                        if current_time - self.gesture_start_time >= hold_time:
                            return self._trigger_gesture(gesture_id, gesture_def)
                    else:
                        self.current_gesture = gesture_id
                        self.gesture_start_time = current_time
        
        return None
    
    def _matches_gesture(self, hand_data: List[Dict], gesture_def: Dict) -> bool:
        """Check if hand data matches gesture definition"""
        detection_type = gesture_def.get('detection', 'finger_states')
        
        if detection_type == 'finger_states':
            return self._match_finger_states(hand_data[0], gesture_def)
        elif detection_type == 'distance_threshold':
            return self._match_distance(hand_data[0], gesture_def)
        elif detection_type == 'motion':
            return self._match_motion(hand_data[0], gesture_def)
        elif detection_type == 'two_hand_distance':
            if len(hand_data) >= 2:
                return self._match_two_hand_distance(hand_data, gesture_def)
        
        return False
    
    def _match_finger_states(self, hand: Dict, gesture_def: Dict) -> bool:
        """Match based on finger extension states"""
        # This would use the finger states from hand tracking
        # Placeholder implementation
        return False
    
    def _match_distance(self, hand: Dict, gesture_def: Dict) -> bool:
        """Match based on distance between landmarks"""
        landmarks = hand.get('landmarks', [])
        if len(landmarks) < 21:
            return False
        
        landmark_indices = gesture_def.get('landmarks', [])
        if len(landmark_indices) != 2:
            return False
        
        p1 = landmarks[landmark_indices[0]]
        p2 = landmarks[landmark_indices[1]]
        
        distance = np.sqrt(
            (p1['x'] - p2['x'])**2 + 
            (p1['y'] - p2['y'])**2
        )
        
        threshold = gesture_def.get('threshold', 0.05)
        return distance < threshold
    
    def _match_motion(self, hand: Dict, gesture_def: Dict) -> bool:
        """Match based on hand motion"""
        # Would track hand position over time
        # Placeholder implementation
        return False
    
    def _match_two_hand_distance(self, hands: List[Dict], gesture_def: Dict) -> bool:
        """Match based on distance between two hands"""
        # Calculate distance between hand centers
        # Placeholder implementation
        return False
    
    def _trigger_gesture(self, gesture_id: str, gesture_def: Dict) -> Dict:
        """Trigger gesture action"""
        action = gesture_def.get('action')
        
        # Execute callback if registered
        if action in self.action_callbacks:
            self.action_callbacks[action]()
        
        # Update state
        self.last_gesture_time = time.time()
        self.gesture_history.append({
            'gesture_id': gesture_id,
            'action': action,
            'timestamp': self.last_gesture_time
        })
        
        self.logger.info(f"Gesture recognized: {gesture_def['name']} -> {action}")
        
        return {
            'gesture_id': gesture_id,
            'name': gesture_def['name'],
            'action': action
        }
    
    def reset(self) -> None:
        """Reset gesture state"""
        self.current_gesture = None
        self.gesture_start_time = None
        self.gesture_history.clear()
