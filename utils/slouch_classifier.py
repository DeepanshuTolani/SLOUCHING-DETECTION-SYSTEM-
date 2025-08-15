"""
Slouch Classifier
Machine learning model for posture classification
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

class SlouchClassifier:
    def __init__(self):
        """Initialize the slouch classifier"""
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'head_forward_ratio',
            'shoulder_angle',
            'head_tilt',
            'torso_angle',
            'head_height_ratio',
            'shoulder_width_ratio',
            'torso_length_ratio',
            'head_shoulder_ratio'
        ]
    
    def train_model(self, features, labels, test_size=0.2, random_state=42):
        """Train the slouch classification model"""
        print("Training Slouch Classifier...")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Good Posture', 'Slouching']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'test_predictions': y_pred,
            'test_labels': y_test
        }
    
    def predict(self, features):
        """Predict posture class and confidence"""
        if not self.is_trained or self.model is None:
            # Return default prediction if model not trained
            return 0, 0.5
        
        # Ensure features is a 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get prediction probability
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        if not self.is_trained:
            print("Warning: Model not trained. Nothing to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and scaler"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.is_trained = model_data.get('is_trained', True)
            
            print(f"Model loaded from {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained or self.model is None:
            return None
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return sorted_importance
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for demonstration"""
        print("Generating synthetic training data...")
        
        np.random.seed(42)
        
        # Generate features for good posture
        good_posture_features = []
        for _ in range(n_samples // 2):
            features = [
                np.random.normal(-0.05, 0.02),  # head_forward_ratio (slightly back)
                np.random.normal(0, 0.1),       # shoulder_angle (horizontal)
                np.random.normal(0, 0.1),       # head_tilt (straight)
                np.random.normal(1.57, 0.1),    # torso_angle (vertical)
                np.random.normal(0.15, 0.02),   # head_height_ratio
                np.random.normal(0.25, 0.02),   # shoulder_width_ratio
                np.random.normal(0.3, 0.02),    # torso_length_ratio
                np.random.normal(0.2, 0.02)     # head_shoulder_ratio
            ]
            good_posture_features.append(features)
        
        # Generate features for slouching posture
        slouching_features = []
        for _ in range(n_samples // 2):
            features = [
                np.random.normal(0.1, 0.05),    # head_forward_ratio (forward)
                np.random.normal(0.2, 0.2),     # shoulder_angle (tilted)
                np.random.normal(0.3, 0.2),     # head_tilt (tilted)
                np.random.normal(1.3, 0.2),     # torso_angle (leaning forward)
                np.random.normal(0.1, 0.02),    # head_height_ratio (lower)
                np.random.normal(0.22, 0.03),   # shoulder_width_ratio
                np.random.normal(0.28, 0.03),   # torso_length_ratio
                np.random.normal(0.18, 0.03)    # head_shoulder_ratio
            ]
            slouching_features.append(features)
        
        # Combine features and labels
        features = np.array(good_posture_features + slouching_features)
        labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
        
        # Shuffle the data
        indices = np.random.permutation(n_samples)
        features = features[indices]
        labels = labels[indices]
        
        print(f"Generated {n_samples} synthetic samples")
        print(f"Good posture samples: {np.sum(labels == 0)}")
        print(f"Slouching samples: {np.sum(labels == 1)}")
        
        return features, labels
