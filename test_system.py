"""
System Test Script
Tests all components of the slouching detection system
"""

import sys
import os
import numpy as np
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_pose_detector():
    """Test pose detector initialization"""
    print("\nTesting pose detector...")
    
    try:
        from utils.pose_detector import PoseDetector
        detector = PoseDetector()
        print("✓ PoseDetector initialized successfully")
        return True
    except Exception as e:
        print(f"✗ PoseDetector initialization failed: {e}")
        return False

def test_slouch_classifier():
    """Test slouch classifier"""
    print("\nTesting slouch classifier...")
    
    try:
        from utils.slouch_classifier import SlouchClassifier
        classifier = SlouchClassifier()
        print("✓ SlouchClassifier initialized successfully")
        
        # Test synthetic data generation
        features, labels = classifier.generate_synthetic_data(n_samples=100)
        print(f"✓ Generated {len(features)} synthetic samples")
        
        # Test model training
        results = classifier.train_model(features, labels, test_size=0.3)
        print(f"✓ Model trained with accuracy: {results['accuracy']:.3f}")
        
        # Test prediction
        test_features = np.array([[-0.03, 0.0, 0.05, 1.57, 0.15, 0.25, 0.3, 0.2]])
        prediction, confidence = classifier.predict(test_features)
        print(f"✓ Prediction test: {prediction} (confidence: {confidence:.3f})")
        
        return True
    except Exception as e:
        print(f"✗ SlouchClassifier test failed: {e}")
        return False

def test_data_logger():
    """Test data logger"""
    print("\nTesting data logger...")
    
    try:
        from utils.data_logger import DataLogger
        logger = DataLogger()
        print("✓ DataLogger initialized successfully")
        
        # Test session logging
        test_session = {
            'start_time': datetime.now(),
            'end_time': datetime.now(),
            'duration': 300,
            'total_detections': 1000,
            'slouch_count': 200,
            'good_posture_count': 800,
            'alerts': [{'timestamp': '10:30:00', 'message': 'Test alert'}]
        }
        
        session_id = logger.log_session(test_session)
        print(f"✓ Session logged with ID: {session_id}")
        
        # Test analytics
        analytics = logger.get_analytics()
        print(f"✓ Analytics generated with {analytics['summary']['total_sessions']} sessions")
        
        return True
    except Exception as e:
        print(f"✗ DataLogger test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app initialization"""
    print("\nTesting Flask app...")
    
    try:
        from app import app
        print("✓ Flask app imported successfully")
        
        # Test basic routes
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Main route working")
            else:
                print(f"✗ Main route failed: {response.status_code}")
                return False
            
            response = client.get('/get_status')
            if response.status_code == 200:
                print("✓ Status route working")
            else:
                print(f"✗ Status route failed: {response.status_code}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'utils/pose_detector.py',
        'utils/slouch_classifier.py',
        'utils/data_logger.py',
        'templates/index.html',
        'scripts/train_model.py'
    ]
    
    required_dirs = [
        'models',
        'data',
        'static',
        'templates',
        'utils',
        'scripts',
        'docs',
        'tests'
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_good = False
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/ directory exists")
        else:
            print(f"✗ {dir_path}/ directory missing")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("=" * 60)
    print("SLOUCHING DETECTION SYSTEM - COMPREHENSIVE TEST")
    print("Team 4 - Final Year Project")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Pose Detector", test_pose_detector),
        ("Slouch Classifier", test_slouch_classifier),
        ("Data Logger", test_data_logger),
        ("Flask App", test_flask_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for demonstration.")
        print("\nNext steps:")
        print("1. Run: python scripts/train_model.py")
        print("2. Run: python app.py")
        print("3. Open: http://localhost:5000")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
