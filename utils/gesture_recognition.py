import json
import os
import numpy as np
import joblib
from collections import deque
from utils.hand_utils import HandDetector
from config.settings import (
    GESTURE_CONFIG,
    MODEL_CONFIG,
    ANGLE_THRESHOLDS,
    DISTANCE_THRESHOLDS
)

class GestureRecognizer:
    def __init__(self):
        self.hand_detector = HandDetector()
        self.gesture_history = deque(maxlen=GESTURE_CONFIG['smoothing_frames'])
        self.gesture_definitions = self._load_gesture_definitions()
        self.ml_model = None
        if MODEL_CONFIG['use_ml_classifier']:
            self._load_ml_model()
    
    def _load_gesture_definitions(self):
        definitions_path = GESTURE_CONFIG['gesture_definitions_path']
        if os.path.exists(definitions_path):
            try:
                with open(definitions_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading gesture definitions: {e}")
                return self._get_default_gestures()
        else:
            default_gestures = self._get_default_gestures()
            self._save_gesture_definitions(default_gestures)
            return default_gestures
    
    def _get_default_gestures(self):
        return {
            "thumbs_up": {
                "description": "Thumb extended upward, other fingers closed",
                "finger_states": {
                    "thumb": True,
                    "index": False,
                    "middle": False,
                    "ring": False,
                    "pinky": False
                },
                "special_conditions": ["thumb_upward"],
                "confidence_weight": 0.95
            },
            "thumbs_down": {
                "description": "Thumb pointing downward, other fingers closed",
                "finger_states": {
                    "thumb": True,
                    "index": False,
                    "middle": False,
                    "ring": False,
                    "pinky": False
                },
                "special_conditions": ["thumb_downward"],
                "confidence_weight": 0.95
            },
            "peace_sign": {
                "description": "Index and middle fingers extended in V shape",
                "finger_states": {
                    "thumb": False,
                    "index": True,
                    "middle": True,
                    "ring": False,
                    "pinky": False
                },
                "special_conditions": ["fingers_v_shape"],
                "confidence_weight": 0.95
            },
            "open_palm": {
                "description": "All fingers extended showing open palm",
                "finger_states": {
                    "thumb": True,
                    "index": True,
                    "middle": True,
                    "ring": True,
                    "pinky": True
                },
                "special_conditions": ["palm_facing_camera", "fingers_natural_spread"],
                "confidence_weight": 0.95
            },
            "fist": {
                "description": "All fingers closed",
                "finger_states": {
                    "thumb": False,
                    "index": False,
                    "middle": False,
                    "ring": False,
                    "pinky": False
                },
                "confidence_weight": 1.0
            },
            "pointing": {
                "description": "Only index finger extended",
                "finger_states": {
                    "thumb": False,
                    "index": True,
                    "middle": False,
                    "ring": False,
                    "pinky": False
                },
                "confidence_weight": 1.0
            }
        }
    
    def _save_gesture_definitions(self, definitions):
        try:
            os.makedirs(os.path.dirname(GESTURE_CONFIG['gesture_definitions_path']), exist_ok=True)
            with open(GESTURE_CONFIG['gesture_definitions_path'], 'w') as f:
                json.dump(definitions, f, indent=4)
        except Exception as e:
            print(f"Error saving gesture definitions: {e}")
    
    def _load_ml_model(self):
        model_path = MODEL_CONFIG['model_path']
        if os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                self.ml_model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                print(f"ML model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading ML model: {e}")
                self.ml_model = None
    
    def _check_special_conditions(self, landmarks, conditions):
        for condition in conditions:
            if condition == "thumb_upward":
                thumb_direction = self.hand_detector._get_thumb_direction(landmarks)
                if thumb_direction != "up":
                    return False
            
            elif condition == "thumb_downward":
                thumb_direction = self.hand_detector._get_thumb_direction(landmarks)
                if thumb_direction != "down":
                    return False
            
            elif condition == "fingers_v_shape":
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                index_mcp = landmarks[5]
                middle_mcp = landmarks[9]
                
                index_angle = self.hand_detector.calculate_angle(index_mcp, landmarks[6], index_tip)
                middle_angle = self.hand_detector.calculate_angle(middle_mcp, landmarks[10], middle_tip)
                
                if index_angle < 140 or middle_angle < 140:
                    return False
                
                finger_distance = self.hand_detector.calculate_distance(index_tip, middle_tip)
                palm_size = self.hand_detector.get_palm_size(landmarks)
                if finger_distance / palm_size < 0.3:
                    return False
        
        return True
    
    def _calculate_gesture_confidence(self, finger_states, target_states, landmarks, gesture_def):
        matches = 0
        total = len(target_states)
        for finger, expected_state in target_states.items():
            if finger_states.get(finger, False) == expected_state:
                matches += 1
        
        basic_confidence = matches / total
        confidence_weight = gesture_def.get('confidence_weight', 1.0)
        confidence = basic_confidence * confidence_weight
        
        special_conditions = gesture_def.get('special_conditions', [])
        if special_conditions:
            if not self._check_special_conditions(landmarks, special_conditions):
                confidence *= 0.5
        
        return min(confidence, 1.0)
    
    def _recognize_rule_based(self, hand_features):
        finger_states = hand_features['finger_states']
        landmarks = hand_features['landmarks']
        
        thumb_direction = self.hand_detector._get_thumb_direction(landmarks)
        
        if (finger_states['thumb'] and not finger_states['index'] and 
            not finger_states['middle'] and not finger_states['ring'] and 
            not finger_states['pinky'] and thumb_direction == "down"):
            return "thumbs_down", 0.95
        
        if (finger_states['thumb'] and not finger_states['index'] and 
            not finger_states['middle'] and not finger_states['ring'] and 
            not finger_states['pinky'] and thumb_direction == "up"):
            return "thumbs_up", 0.95
        
        if (not finger_states['thumb'] and finger_states['index'] and 
            not finger_states['middle'] and not finger_states['ring'] and 
            not finger_states['pinky']):
            return "pointing", 0.95
        
        if (not finger_states['thumb'] and not finger_states['index'] and 
            not finger_states['middle'] and not finger_states['ring'] and 
            not finger_states['pinky']):
            return "fist", 0.95
        
        best_gesture = "unknown"
        best_confidence = 0.0
        
        for gesture_name, gesture_def in self.gesture_definitions.items():
            target_states = gesture_def['finger_states']
            confidence = self._calculate_gesture_confidence(
                finger_states, target_states, landmarks, gesture_def
            )
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_gesture = gesture_name
        
        return best_gesture, best_confidence
    
    def _recognize_ml_based(self, hand_features):
        if self.ml_model is None:
            return "unknown", 0.0
        
        try:
            normalized_landmarks = hand_features['normalized_landmarks']
            feature_vector = normalized_landmarks.flatten()
            
            if len(feature_vector) != MODEL_CONFIG['feature_vector_size']:
                return "unknown", 0.0
            
            features_scaled = self.scaler.transform([feature_vector])
            prediction = self.ml_model.predict(features_scaled)[0]
            gesture_name = self.label_encoder.inverse_transform([prediction])[0]
            
            if hasattr(self.ml_model, 'predict_proba'):
                probabilities = self.ml_model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.8
            
            return gesture_name, confidence
        
        except Exception as e:
            print(f"Error in ML-based recognition: {e}")
            return "unknown", 0.0
    
    def _smooth_gesture_recognition(self, gesture_name, confidence):
        self.gesture_history.append((gesture_name, confidence))
        
        if len(self.gesture_history) < GESTURE_CONFIG['smoothing_frames']:
            return gesture_name, confidence
        
        gesture_counts = {}
        total_confidence = {}
        
        for hist_gesture, hist_confidence in self.gesture_history:
            if hist_gesture not in gesture_counts:
                gesture_counts[hist_gesture] = 0
                total_confidence[hist_gesture] = 0.0
            gesture_counts[hist_gesture] += 1
            total_confidence[hist_gesture] += hist_confidence
        
        most_frequent_gesture = max(gesture_counts, key=gesture_counts.get)
        avg_confidence = total_confidence[most_frequent_gesture] / gesture_counts[most_frequent_gesture]
        
        stability_ratio = gesture_counts[most_frequent_gesture] / len(self.gesture_history)
        if stability_ratio < 0.6:
            avg_confidence *= 0.7
        
        return most_frequent_gesture, avg_confidence
    
    def recognize(self, hand_landmarks, hand_type="Right"):
        try:
            image_shape = (480, 640, 3)
            hand_features = self.hand_detector.get_hand_features(hand_landmarks, image_shape)
            
            if MODEL_CONFIG['use_ml_classifier'] and self.ml_model is not None:
                gesture_name, confidence = self._recognize_ml_based(hand_features)
                if confidence < 0.5:
                    rule_gesture, rule_confidence = self._recognize_rule_based(hand_features)
                    if rule_confidence > confidence:
                        gesture_name, confidence = rule_gesture, rule_confidence
            else:
                gesture_name, confidence = self._recognize_rule_based(hand_features)
            
            if GESTURE_CONFIG['smoothing_frames'] > 1:
                gesture_name, confidence = self._smooth_gesture_recognition(gesture_name, confidence)
            
            return gesture_name, confidence
        
        except Exception as e:
            print(f"Error in gesture recognition: {e}")
            return "unknown", 0.0

def detect_gesture(fingers):
    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace Sign"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing"
    else:
        return "Unknown"
