import numpy as np
from models.gesture_classifier import GestureClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_comprehensive_model():
    print("Training Comprehensive Gesture Recognition Model")
    print("=" * 60)
    
    os.makedirs('models', exist_ok=True)
    
    classifier = GestureClassifier(model_type='random_forest')
    
    print("Generating training data...")
    X, y = classifier.generate_synthetic_data(n_samples=1500)
    
    print("Adding additional thumbs_down samples...")
    for _ in range(500):
        landmarks = classifier._generate_thumbs_down_data()
        features = classifier.extract_features(landmarks)
        X = np.vstack([X, features])
        y = np.append(y, 'thumbs_down')
    
    print(f"Total samples: {len(X)}")
    print(f"Unique gestures: {np.unique(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training model with hyperparameter tuning...")
    best_params = classifier.hyperparameter_tuning(X_train, y_train)
    
    y_pred = classifier.model.predict(classifier.scaler.transform(X_test))
    y_test_encoded = classifier.label_encoder.transform(y_test)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_encoded, y_pred, 
                              target_names=classifier.label_encoder.classes_))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classifier.label_encoder.classes_,
                yticklabels=classifier.label_encoder.classes_)
    plt.title('Gesture Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('gesture_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    classifier.save_model('models/trained_gesture_classifier.pkl')
    print("Model saved successfully!")
    
    return classifier

if __name__ == "__main__":
    trained_model = train_comprehensive_model()
