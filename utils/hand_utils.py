import cv2
import mediapipe as mp
import numpy as np
import math
from config.settings import HAND_DETECTION_CONFIG, ANGLE_THRESHOLDS

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=HAND_DETECTION_CONFIG['static_image_mode'],
            max_num_hands=HAND_DETECTION_CONFIG['max_num_hands'],
            min_detection_confidence=HAND_DETECTION_CONFIG['min_detection_confidence'],
            min_tracking_confidence=HAND_DETECTION_CONFIG['min_tracking_confidence'],
            model_complexity=HAND_DETECTION_CONFIG['model_complexity']
        )
        
        self.LANDMARK_INDICES = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20],
            'palm': [0, 1, 5, 9, 13, 17]
        }
        
        self.JOINT_INDICES = {
            'thumb': {'pip': 2, 'dip': 3, 'tip': 4},
            'index': {'pip': 6, 'dip': 7, 'tip': 8},
            'middle': {'pip': 10, 'dip': 11, 'tip': 12},
            'ring': {'pip': 14, 'dip': 15, 'tip': 16},
            'pinky': {'pip': 18, 'dip': 19, 'tip': 20}
        }
    
    def detect_hands(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = self.hands.process(rgb_image)
        rgb_image.flags.writeable = True
        return results
    
    def get_landmarks_array(self, hand_landmarks, image_shape):
        height, width = image_shape[:2]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z
            landmarks.append([x, y, z])
        return np.array(landmarks)
    
    def calculate_distance(self, point1, point2):
        if len(point1) == 3 and len(point2) == 3:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
        else:
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_angle(self, point1, point2, point3):
        v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    def _get_thumb_direction(self, landmarks):
        if len(landmarks) <= 4:
            return "unknown"
        
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        thumb_vector = np.array([thumb_tip[0] - thumb_mcp[0], thumb_tip[1] - thumb_mcp[1]])
        middle_mcp = landmarks[9]
        palm_vector = np.array([middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]])
        
        cross_product = np.cross(palm_vector, thumb_vector)
        
        if thumb_tip[1] < wrist[1] - 20:
            return "up"
        elif thumb_tip[1] > wrist[1] + 20:
            return "down"
        elif cross_product > 50:
            return "right"
        elif cross_product < -50:
            return "left"
        else:
            return "neutral"
    
    def get_finger_angles(self, landmarks):
        angles = {}
        
        if len(landmarks) > 4:
            thumb_angle_1 = self.calculate_angle(landmarks[1], landmarks[2], landmarks[3])
            thumb_angle_2 = self.calculate_angle(landmarks[2], landmarks[3], landmarks[4])
            angles['thumb'] = min(thumb_angle_1, thumb_angle_2)
        
        for finger_name in ['index', 'middle', 'ring', 'pinky']:
            joints = self.JOINT_INDICES[finger_name]
            if len(landmarks) > joints['tip']:
                if finger_name == 'index':
                    mcp_idx = 5
                elif finger_name == 'middle':
                    mcp_idx = 9
                elif finger_name == 'ring':
                    mcp_idx = 13
                else:
                    mcp_idx = 17
                
                pip_angle = self.calculate_angle(
                    landmarks[mcp_idx], landmarks[joints['pip']], landmarks[joints['dip']]
                )
                dip_angle = self.calculate_angle(
                    landmarks[joints['pip']], landmarks[joints['dip']], landmarks[joints['tip']]
                )
                angles[finger_name] = min(pip_angle, dip_angle)
        
        return angles
    
    def is_finger_extended(self, finger_name, landmarks):
        if finger_name == 'thumb':
            return self._is_thumb_extended(landmarks)
        else:
            return self._is_regular_finger_extended(finger_name, landmarks)
    
    def _is_thumb_extended(self, landmarks):
        if len(landmarks) <= 4:
            return False
        
        thumb_tip = landmarks[4]
        palm_base = landmarks[0]
        index_mcp = landmarks[5]
        
        thumb_palm_dist = self.calculate_distance(thumb_tip, palm_base)
        index_palm_dist = self.calculate_distance(index_mcp, palm_base)
        distance_ratio = thumb_palm_dist / index_palm_dist if index_palm_dist > 0 else 0
        
        thumb_angle = self.calculate_angle(landmarks[2], landmarks[3], landmarks[4])
        
        thumb_y = thumb_tip[1]
        fingers_y = [landmarks[8][1], landmarks[12][1], landmarks[16][1], landmarks[20][1]]
        avg_finger_y = np.mean(fingers_y)
        
        distance_extended = distance_ratio > 0.8
        angle_extended = thumb_angle > ANGLE_THRESHOLDS['thumb_extended_min']
        position_extended = abs(thumb_y - avg_finger_y) > 30
        
        return sum([distance_extended, angle_extended, position_extended]) >= 2
    
    def _is_regular_finger_extended(self, finger_name, landmarks):
        joints = self.JOINT_INDICES[finger_name]
        if len(landmarks) <= joints['tip']:
            return False
        
        mcp_indices = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}
        mcp_idx = mcp_indices[finger_name]
        
        pip_angle = self.calculate_angle(
            landmarks[mcp_idx], landmarks[joints['pip']], landmarks[joints['dip']]
        )
        dip_angle = self.calculate_angle(
            landmarks[joints['pip']], landmarks[joints['dip']], landmarks[joints['tip']]
        )
        
        angle_extended = (pip_angle > ANGLE_THRESHOLDS['finger_extended_min'] and
                         dip_angle > ANGLE_THRESHOLDS['finger_extended_min'])
        
        tip_pos = landmarks[joints['tip']]
        mcp_pos = landmarks[mcp_idx]
        finger_length = self.calculate_distance(tip_pos, mcp_pos)
        palm_size = self.get_palm_size(landmarks)
        
        length_ratio = finger_length / palm_size if palm_size > 0 else 0
        distance_extended = length_ratio > 0.6
        
        wrist_y = landmarks[0][1]
        tip_y = tip_pos[1]
        y_extended = tip_y < wrist_y
        
        return angle_extended and (distance_extended or y_extended)
    
    def get_finger_states(self, landmarks):
        states = {}
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            is_extended = self.is_finger_extended(finger, landmarks)
            states[finger] = is_extended
        return states
    
    def get_hand_orientation(self, landmarks):
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        
        v1 = np.array([index_mcp[0] - wrist[0], index_mcp[1] - wrist[1], index_mcp[2] - wrist[2]])
        v2 = np.array([pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1], pinky_mcp[2] - wrist[2]])
        normal = np.cross(v1, v2)
        
        if normal[2] > 0.1:
            return 'palm_front'
        elif normal[2] < -0.1:
            return 'palm_back'
        else:
            return 'side'
    
    def get_hand_center(self, landmarks):
        palm_indices = self.LANDMARK_INDICES['palm']
        palm_points = landmarks[palm_indices]
        center_x = np.mean(palm_points[:, 0])
        center_y = np.mean(palm_points[:, 1])
        center_z = np.mean(palm_points[:, 2])
        return [center_x, center_y, center_z]
    
    def get_palm_size(self, landmarks):
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        primary_size = self.calculate_distance(wrist, middle_mcp)
        
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        secondary_size = self.calculate_distance(index_mcp, pinky_mcp)
        
        return (primary_size + secondary_size) / 2
    
    def normalize_landmarks(self, landmarks):
        center = self.get_hand_center(landmarks)
        palm_size = self.get_palm_size(landmarks)
        
        if palm_size == 0:
            palm_size = 1
        
        normalized = []
        for landmark in landmarks:
            norm_x = (landmark[0] - center[0]) / palm_size
            norm_y = (landmark[1] - center[1]) / palm_size
            norm_z = (landmark[2] - center[2]) / palm_size if len(landmark) > 2 else 0
            normalized.append([norm_x, norm_y, norm_z])
        
        return np.array(normalized)
    
    def get_hand_features(self, hand_landmarks, image_shape):
        landmarks = self.get_landmarks_array(hand_landmarks, image_shape)
        
        features = {
            'landmarks': landmarks,
            'normalized_landmarks': self.normalize_landmarks(landmarks),
            'finger_states': self.get_finger_states(landmarks),
            'finger_angles': self.get_finger_angles(landmarks),
            'hand_center': self.get_hand_center(landmarks),
            'palm_size': self.get_palm_size(landmarks),
            'hand_orientation': self.get_hand_orientation(landmarks)
        }
        
        return features
    
    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_finger_status(landmarks):
    fingers = []
    
    if landmarks[4][1] < landmarks[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id][1] < landmarks[tip_id - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers
