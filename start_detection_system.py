"""
Start Detection System
Simple script to start the slouching detection system
"""

import os
import sys
import subprocess
import time

def main():
    print("=" * 60)
    print("SLOUCHING DETECTION SYSTEM - TEAM 4")
    print("Final Year Project - Advanced Computer Vision & ML")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found!")
        print("Please run this script from the ml_team4 directory")
        return
    
    # Check if model exists
    if not os.path.exists('models/slouch_classifier.pkl'):
        print("⚠️  Model not found. Training model...")
        try:
            subprocess.run([sys.executable, 'scripts/train_model.py'], check=True)
            print("✅ Model trained successfully!")
        except subprocess.CalledProcessError:
            print("❌ Error training model!")
            return
    
    print("🚀 Starting Flask application...")
    print("📱 Access the system at: http://localhost:5000")
    print("📊 Analytics dashboard will show sample data")
    print("📷 Camera will be initialized when you start detection")
    print("=" * 60)
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Start the Flask application
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
