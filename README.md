
# Project Title

A real-time hand gesture recognition system using MediaPipe and Machine Learning that accurately detects 6 different hand gestures with high confidence.

---
## 🎯 Features

- **Real-time Detection**: Live gesture recognition using webcam  
- **6 Gesture Support**: Thumbs Up, Thumbs Down, Peace Sign, Open Palm, Fist, Pointing  
- **Machine Learning**: Random Forest classifier with synthetic data generation  
- **High Accuracy**: Enhanced thumb direction detection for reliable thumbs up/down recognition  
- **Confidence Scoring**: Only triggers on high-confidence gestures (75%+ threshold)  
- **Gesture Smoothing**: Temporal smoothing to reduce noise and false positives  
- **Visual Feedback**: Hand landmark visualization with gesture information overlay  

---

## 🛠️ Technologies Used

- **Computer Vision**: MediaPipe for hand landmark detection  
- **Machine Learning**: Scikit-learn (Random Forest, SVM)  
- **Image Processing**: OpenCV for camera handling and visualization  
- **Data Processing**: NumPy, Pandas for feature extraction  
- **Visualization**: Matplotlib, Seaborn for model analysis

---

## 📁 Project Structure 

    hand-gesture-detection/
    ├── main.py                              # Main application entry point
    ├── train_model.py                       # Model training script
    ├── requirements.txt                     # Python dependencies
    ├── README.md                            # Project documentation
    ├── config/
    │   ├── __init__.py
    │   └── settings.py                      # Configuration parameters
    ├── utils/
    │   ├── __init__.py
    │   ├── hand_utils.py                    # Hand detection utilities
    │   ├── gesture_recognition.py           # Gesture recognition logic
    │   └── draw_utils.py                    # Drawing and visualization
    ├── models/
    │   ├── __init__.py
    │   ├── gesture_classifier.py            # ML classifier implementation
    │   └── trained_gesture_classifier.pkl   # Trained model (generated)
    ├── gestures/
    │   └── gesture_definitions.json         # Gesture definitions
    └── logs/
        └── gesture_log.txt                  # Application logs (generated)

---

## 🚀 Installation

### 1. Clone the Repository

    git clone https://github.com/rogadesh/hand-gesture-detection.git
    cd hand-gesture-detection


### 2. Create Virtual Environment (Recommended)

    python -m venv venv

    On Linux/macOS:
    source venv/bin/activate

    On Windows:
    venv\Scripts\activate


### 3. Install Dependencies

    pip install -r requirements.txt

---

## 📋 Requirements

    opencv-python==4.8.1.78
    mediapipe==0.10.7
    numpy==1.24.3
    pandas==2.0.3
    scikit-learn==1.3.0
    matplotlib==3.7.2
    seaborn==0.12.2
    joblib==1.3.2

---

## 🎮 Usage

### Step 1: Train the Model (First Time Only)

    python train_model.py


### Step 2: Run the Application

    python main.py

---

### Controls

- **Q**: Quit the application
- Position your hand in front of the camera
- Make clear gestures for best detection

---

## 🤲 Supported Gestures

| Gesture        | Description                                 | Visual |
| -------------- | ------------------------------------------- | ------ |
| **Thumbs Up**  | Thumb pointing upward, other fingers closed | 👍     |
| **Thumbs Down**| Thumb pointing downward, other fingers closed | 👎   |
| **Peace Sign** | Index and middle fingers extended in V-shape| ✌️    |
| **Open Palm**  | All fingers extended showing open palm      | ✋     |
| **Fist**       | All fingers closed                          | ✊     |
| **Pointing**   | Only index finger extended                  | 👉     |

---

## ⚙️ Configuration

### Camera Settings (`config/settings.py`)

    CAMERA_CONFIG = {
    'device_id': 0, # Camera device (0 for default)
    'width': 1280, # Frame width
    'height': 720, # Frame height
    'fps': 30, # Frames per second
    }

---

## 🧠 How It Works

1. **Hand Detection**: Uses MediaPipe to detect 21 hand landmarks in real-time  
2. **Feature Extraction**: Extracts 60+ features from hand landmarks  
3. **Gesture Classification**: Machine Learning classifier with rule-based fallback  
4. **Confidence Scoring**: Only displays gestures above 75% confidence  

---

## 📊 Model Performance

- **Training Accuracy**: ~95%
- **Test Accuracy**: ~92%
- **Cross-validation Score**: ~90%

---

## 🐛 Troubleshooting

### Camera Not Working

- Check camera permissions
- Try different `device_id` values (0, 1, 2)
- Ensure camera is not used by another application

### Low Detection Accuracy

- Ensure good lighting conditions
- Keep hand clearly visible in frame
- Make distinct, clear gestures




