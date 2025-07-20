import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GestureClassifier:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
    
    def extract_features(self, landmarks):
        if len(landmarks) != 21:
            raise ValueError("Expected 21 landmarks")
        
        features = []
        landmarks = np.array(landmarks)
        
        wrist = landmarks[0]
        normalized_landmarks = landmarks - wrist
        features.extend(normalized_landmarks.flatten())
        
        wrist_distances = [np.linalg.norm(landmark - wrist) for landmark in landmarks[1:]]
        features.extend(wrist_distances)
        
        tip_indices = [4, 8, 12, 16, 20]
        tip_distances = [np.linalg.norm(landmarks[i] - wrist) for i in tip_indices]
        features.extend(tip_distances)
        
        for i in range(len(tip_indices)):
            for j in range(i+1, len(tip_indices)):
                dist = np.linalg.norm(landmarks[tip_indices[i]] - landmarks[tip_indices[j]])
                features.append(dist)
        
        finger_angles = []
        finger_indices = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20]
        ]
        
        for finger in finger_indices:
            if len(finger) >= 3:
                p1, p2, p3 = landmarks[finger[1]], landmarks[finger[2]], landmarks[finger[3]]
                v1 = p1 - p2
                v2 = p3 - p2
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                finger_angles.append(angle)
        
        features.extend(finger_angles)
        
        hand_span = np.linalg.norm(landmarks[4] - landmarks[20])
        features.append(hand_span)
        
        palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
        palm_distances = [np.linalg.norm(landmark - palm_center) for landmark in landmarks]
        features.extend(palm_distances)
        
        return np.array(features)
    
    def generate_synthetic_data(self, n_samples=1000):
        gestures = {
            'fist': self._generate_fist_data,
            'open_palm': self._generate_open_palm_data,
            'thumbs_up': self._generate_thumbs_up_data,
            'thumbs_down': self._generate_thumbs_down_data,
            'peace_sign': self._generate_peace_sign_data,
            'pointing': self._generate_pointing_data
        }
        
        X, y = [], []
        
        for gesture_name, generator in gestures.items():
            print(f"Generating {n_samples} samples for {gesture_name}...")
            for _ in range(n_samples):
                landmarks = generator()
                features = self.extract_features(landmarks)
                X.append(features)
                y.append(gesture_name)
        
        return np.array(X), np.array(y)
    
    def _generate_thumbs_down_data(self):
        base_landmarks = [
            (0, 0),
            (-15, -5),
            (-20, -10),
            (-25, -5),
            (-30, 15),
            (10, -15),
            (15, -25),
            (12, -20),
            (8, -15),
            (25, -15),
            (30, -25),
            (28, -20),
            (25, -15),
            (35, -10),
            (38, -20),
            (36, -15),
            (34, -10),
            (40, -5),
            (42, -15),
            (40, -10),
            (38, -5)
        ]
        
        return [(x + np.random.normal(0, 3), y + np.random.normal(0, 3)) for x, y in base_landmarks]
    
    def _generate_fist_data(self):
        base_landmarks = [
            (0, 0),
            (-20, -10),
            (-25, -15),
            (-30, -20),
            (-35, -25),
            (10, -15),
            (15, -25),
            (12, -30),
            (8, -35),
            (25, -15),
            (30, -25),
            (28, -30),
            (25, -35),
            (35, -10),
            (38, -20),
            (36, -25),
            (34, -30),
            (40, -5),
            (42, -15),
            (40, -20),
            (38, -25)
        ]
        
        return [(x + np.random.normal(0, 2), y + np.random.normal(0, 2)) for x, y in base_landmarks]
    
    def _generate_open_palm_data(self):
        base_landmarks = [
            (0, 0),
            (-15, -5),
            (-20, -15),
            (-25, -25),
            (-30, -35),
            (10, -15),
            (12, -30),
            (14, -45),
            (16, -60),
            (25, -15),
            (28, -32),
            (30, -48),
            (32, -65),
            (35, -10),
            (37, -25),
            (39, -40),
            (41, -55),
            (40, -5),
            (42, -20),
            (44, -35),
            (46, -50)
        ]
        
        return [(x + np.random.normal(0, 3), y + np.random.normal(0, 3)) for x, y in base_landmarks]
    
    def _generate_thumbs_up_data(self):
        base_landmarks = [
            (0, 0),
            (-15, -5),
            (-20, -15),
            (-25, -30),
            (-30, -45),
            (10, -15),
            (15, -25),
            (12, -30),
            (8, -35),
            (25, -15),
            (30, -25),
            (28, -30),
            (25, -35),
            (35, -10),
            (38, -20),
            (36, -25),
            (34, -30),
            (40, -5),
            (42, -15),
            (40, -20),
            (38, -25)
        ]
        
        return [(x + np.random.normal(0, 2), y + np.random.normal(0, 2)) for x, y in base_landmarks]
    
    def _generate_peace_sign_data(self):
        base_landmarks = [
            (0, 0),
            (-15, -5),
            (-18, -12),
            (-20, -18),
            (-22, -25),
            (10, -15),
            (12, -30),
            (14, -45),
            (16, -60),
            (25, -15),
            (28, -32),
            (30, -48),
            (32, -65),
            (35, -10),
            (38, -20),
            (36, -25),
            (34, -30),
            (40, -5),
            (42, -15),
            (40, -20),
            (38, -25)
        ]
        
        return [(x + np.random.normal(0, 2), y + np.random.normal(0, 2)) for x, y in base_landmarks]
    
    def _generate_pointing_data(self):
        base_landmarks = [
            (0, 0),
            (-15, -5),
            (-18, -12),
            (-20, -18),
            (-22, -25),
            (10, -15),
            (12, -30),
            (14, -45),
            (16, -60),
            (25, -15),
            (30, -25),
            (28, -30),
            (25, -35),
            (35, -10),
            (38, -20),
            (36, -25),
            (34, -30),
            (40, -5),
            (42, -15),
            (40, -20),
            (38, -25)
        ]
        
        return [(x + np.random.normal(0, 2), y + np.random.normal(0, 2)) for x, y in base_landmarks]
    
    def train_model(self, X=None, y=None, test_size=0.2):
        print("Training Gesture Classification Model...")
        
        if X is None or y is None:
            print("Generating synthetic training data...")
            X, y = self.generate_synthetic_data(n_samples=500)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        
        y_pred = self.model.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        self.is_trained = True
        print("Model training completed!")
        
        self._plot_confusion_matrix(y_test, y_pred)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores,
            'model': self.model
        }
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Gesture Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, landmarks, return_probabilities=False):
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        features = self.extract_features(landmarks).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        gesture = self.label_encoder.inverse_transform([prediction])[0]
        
        if return_probabilities:
            probabilities = self.model.predict_proba(features_scaled)[0]
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: prob
                for i, prob in enumerate(probabilities)
            }
            return gesture, prob_dict
        
        return gesture
    
    def save_model(self, filepath='trained_gesture_classifier.pkl'):
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='trained_gesture_classifier.pkl'):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            print(f"Model type: {self.model_type}")
            print(f"Training date: {model_data.get('training_date', 'Unknown')}")
        except FileNotFoundError:
            print(f"Model file {filepath} not found!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def hyperparameter_tuning(self, X, y):
        print("Performing hyperparameter tuning...")
        
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            model = SVC(random_state=42)
        
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_scaled, y_encoded)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return grid_search.best_params_

def main():
    print("Gesture Classifier Training Pipeline")
    print("=" * 50)
    
    classifier = GestureClassifier(model_type='random_forest')
    results = classifier.train_model()
    classifier.save_model()
    
    test_classifier = GestureClassifier()
    test_classifier.load_model()
    
    sample_landmarks = classifier._generate_peace_sign_data()
    prediction = test_classifier.predict(sample_landmarks, return_probabilities=True)
    print(f"\nSample prediction: {prediction}")
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()
