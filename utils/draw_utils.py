import cv2
import mediapipe as mp
import numpy as np
from config.settings import DRAWING_CONFIG, UI_CONFIG

class DrawingUtils:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.colors = {
            'landmark': DRAWING_CONFIG['landmark_color'],
            'connection': DRAWING_CONFIG['connection_color'],
            'text': DRAWING_CONFIG['text_color'],
            'confidence_bar_bg': (50, 50, 50),
            'confidence_bar_fill': (0, 255, 0),
            'confidence_bar_low': (0, 165, 255),
            'confidence_bar_medium': (0, 255, 255),
            'confidence_bar_high': (0, 255, 0),
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = DRAWING_CONFIG['text_scale']
        self.font_thickness = DRAWING_CONFIG['text_thickness']
        
        self.frame_count = 0
        self.fps_start_time = cv2.getTickCount()
    
    def draw_landmarks(self, image, hand_landmarks):
        if not DRAWING_CONFIG['show_landmarks']:
            return image
        
        if DRAWING_CONFIG['show_connections']:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            for landmark in hand_landmarks.landmark:
                height, width = image.shape[:2]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y),
                          DRAWING_CONFIG['landmark_thickness'],
                          self.colors['landmark'], -1)
        
        return image
    
    def draw_gesture_info(self, image, gesture_name, confidence, hand_type="Right", position=None):
        if not DRAWING_CONFIG['show_gesture_info']:
            return image
        
        if position is None:
            position = UI_CONFIG['gesture_info_position']
        
        gesture_text = f"{hand_type} Hand: {gesture_name.replace('_', ' ').title()}"
        confidence_text = f"Confidence: {confidence:.2f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            gesture_text, self.font, self.font_scale, self.font_thickness
        )
        
        bg_start = (position[0] - 10, position[1] - text_height - 15)
        bg_end = (position[0] + text_width + 10, position[1] + 40)
        cv2.rectangle(image, bg_start, bg_end, (0, 0, 0), -1)
        cv2.rectangle(image, bg_start, bg_end, (255, 255, 255), 2)
        
        cv2.putText(image, gesture_text, position, self.font,
                   self.font_scale, self.colors['text'], self.font_thickness)
        
        confidence_pos = (position[0], position[1] + 25)
        cv2.putText(image, confidence_text, confidence_pos, self.font,
                   self.font_scale * 0.7, self.colors['text'], 1)
        
        return image
    
    def draw_confidence_bar(self, image, confidence, position=None):
        if position is None:
            position = (UI_CONFIG['gesture_info_position'][0],
                       UI_CONFIG['gesture_info_position'][1] + 50)
        
        bar_length = UI_CONFIG['confidence_bar_length']
        bar_height = UI_CONFIG['confidence_bar_height']
        
        start_point = position
        end_point = (position[0] + bar_length, position[1] + bar_height)
        cv2.rectangle(image, start_point, end_point, self.colors['confidence_bar_bg'], -1)
        
        fill_length = int(bar_length * confidence)
        fill_end = (position[0] + fill_length, position[1] + bar_height)
        
        if confidence < 0.5:
            fill_color = self.colors['confidence_bar_low']
        elif confidence < 0.8:
            fill_color = self.colors['confidence_bar_medium']
        else:
            fill_color = self.colors['confidence_bar_high']
        
        cv2.rectangle(image, start_point, fill_end, fill_color, -1)
        cv2.rectangle(image, start_point, end_point, (255, 255, 255), 2)
        
        conf_text = f"{int(confidence * 100)}%"
        text_pos = (position[0] + bar_length + 10, position[1] + bar_height - 5)
        cv2.putText(image, conf_text, text_pos, self.font,
                   self.font_scale * 0.6, self.colors['text'], 1)
        
        return image
    
    def draw_fps(self, image):
        if not UI_CONFIG['show_fps']:
            return image
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = cv2.getTickCount()
            time_diff = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            fps = 30 / time_diff
            self.current_fps = fps
            self.fps_start_time = current_time
        
        if hasattr(self, 'current_fps'):
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(image, fps_text, UI_CONFIG['fps_position'],
                       self.font, self.font_scale * 0.7,
                       self.colors['text'], self.font_thickness)
        
        return image
