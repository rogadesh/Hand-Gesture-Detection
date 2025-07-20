CAMERA_CONFIG = {
    'device_id': 0,
    'width': 1280,
    'height': 720,
    'fps': 30,
}

HAND_DETECTION_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 2,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.5,
    'model_complexity': 1,
}

GESTURE_CONFIG = {
    'min_confidence': 0.75,
    'smoothing_frames': 7,
    'enable_custom_gestures': True,
    'gesture_definitions_path': 'gestures/gesture_definitions.json',
}

DRAWING_CONFIG = {
    'landmark_color': (0, 255, 0),
    'connection_color': (255, 0, 0),
    'text_color': (255, 255, 255),
    'text_font': 1,
    'text_scale': 1.0,
    'text_thickness': 2,
    'landmark_thickness': 2,
    'connection_thickness': 2,
    'show_landmarks': True,
    'show_connections': True,
    'show_gesture_info': True,
}

LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': 'logs/gesture_log.txt',
    'max_file_size': 10 * 1024 * 1024,
    'backup_count': 3,
}

ANGLE_THRESHOLDS = {
    'finger_extended_min': 140,
    'finger_closed_max': 100,
    'thumb_extended_min': 120,
    'thumb_closed_max': 60,
}

DISTANCE_THRESHOLDS = {
    'fingertip_proximity': 0.08,
    'palm_size_factor': 0.3,
    'gesture_stability': 0.03,
}

MODEL_CONFIG = {
    'use_ml_classifier': True,
    'model_path': 'models/trained_gesture_classifier.pkl',
    'feature_vector_size': 63,
    'training_data_path': 'data/training_data.csv',
    'validation_split': 0.2,
}

UI_CONFIG = {
    'window_name': 'Hand Gesture Detection',
    'show_fps': True,
    'fps_position': (10, 30),
    'gesture_info_position': (10, 70),
    'confidence_bar_length': 200,
    'confidence_bar_height': 20,
}

PERFORMANCE_CONFIG = {
    'frame_skip': 1,
    'roi_enabled': False,
    'roi_coordinates': (100, 100, 500, 500),
    'multi_threading': False,
    'gpu_acceleration': False,
}

DEBUG_CONFIG = {
    'debug_mode': False,
    'save_debug_frames': False,
    'debug_output_dir': 'debug/',
    'print_landmarks': False,
    'show_angles': False,
}
