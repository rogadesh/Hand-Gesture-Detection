import cv2
import logging
from utils.hand_utils import HandDetector
from utils.gesture_recognition import GestureRecognizer
from utils.draw_utils import DrawingUtils
from config.settings import CAMERA_CONFIG, GESTURE_CONFIG, LOG_CONFIG

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_CONFIG['level']),
        format=LOG_CONFIG['format'],
        handlers=[
            logging.FileHandler(LOG_CONFIG['file_path']),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    hand_detector = HandDetector()
    gesture_recognizer = GestureRecognizer()
    drawing_utils = DrawingUtils()
    
    cap = cv2.VideoCapture(CAMERA_CONFIG['device_id'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
    
    logger.info("Starting hand gesture detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame from camera")
                break
            
            frame = cv2.flip(frame, 1)
            results = hand_detector.detect_hands(frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_type = handedness.classification[0].label
                    gesture_name, confidence = gesture_recognizer.recognize(hand_landmarks, hand_type)
                    
                    frame = drawing_utils.draw_landmarks(frame, hand_landmarks)
                    frame = drawing_utils.draw_gesture_info(frame, gesture_name, confidence, hand_type)
                    
                    if confidence > GESTURE_CONFIG['min_confidence']:
                        logger.info(f"Detected {hand_type} hand gesture: {gesture_name} (confidence: {confidence:.2f})")
            
            cv2.imshow('Hand Gesture Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Hand gesture detection stopped")

if __name__ == "__main__":
    main()
