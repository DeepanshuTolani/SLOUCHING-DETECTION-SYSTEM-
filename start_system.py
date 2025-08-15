"""
Startup Script for Slouching Detection System
Automatically trains the model and launches the web application
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print the system banner"""
    print("=" * 70)
    print("🎯 SLOUCHING DETECTION SYSTEM - TEAM 4")
    print("🚀 Final Year Project - Advanced Computer Vision & ML")
    print("=" * 70)
    print()

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'flask', 'opencv-python', 'mediapipe', 'numpy', 
        'scikit-learn', 'matplotlib', 'seaborn', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def train_model():
    """Train the machine learning model"""
    print("\n🤖 Training the machine learning model...")
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, 'scripts/train_model.py'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print(f"❌ Model training failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def start_application():
    """Start the Flask web application"""
    print("\n🌐 Starting the web application...")
    print("📱 The system will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the application")
    print()
    
    try:
        # Start the Flask app
        subprocess.run([
            sys.executable, 'app.py'
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Application error: {e}")

def main():
    """Main startup function"""
    print_banner()
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: Please run this script from the ml_team4 directory")
        print("Current directory:", os.getcwd())
        return
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Train model (if not already trained)
    model_path = os.path.join('models', 'slouch_classifier.pkl')
    if not os.path.exists(model_path):
        print(f"\n📋 Model file not found: {model_path}")
        if not train_model():
            print("❌ Failed to train model. Exiting.")
            return
    else:
        print(f"\n✅ Model already exists: {model_path}")
    
    # Step 3: Start application
    start_application()

if __name__ == "__main__":
    main()
