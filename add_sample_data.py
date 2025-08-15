"""
Add Sample Analytics Data
Creates realistic sample data for the analytics dashboard
"""

import json
import os
from datetime import datetime, timedelta
import random

def add_sample_data():
    """Add realistic sample data to sessions.json"""
    
    # Read existing data
    sessions_file = "data/sessions.json"
    
    if os.path.exists(sessions_file):
        with open(sessions_file, 'r') as f:
            sessions = json.load(f)
    else:
        sessions = []
    
    # Create sample sessions with realistic data
    sample_sessions = []
    
    # Generate sessions for the last 7 days
    for i in range(7):
        # Create 2-3 sessions per day
        for j in range(random.randint(2, 3)):
            # Random start time during the day
            start_hour = random.randint(9, 18)  # 9 AM to 6 PM
            start_minute = random.randint(0, 59)
            
            # Session start time
            session_date = datetime.now() - timedelta(days=i)
            start_time = session_date.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            
            # Session duration (5-30 minutes)
            duration_minutes = random.randint(5, 30)
            end_time = start_time + timedelta(minutes=duration_minutes)
            
            # Realistic detection data
            total_detections = random.randint(50, 300)
            slouch_count = random.randint(5, int(total_detections * 0.4))  # 5-40% slouch rate
            good_posture_count = total_detections - slouch_count
            slouch_percentage = (slouch_count / total_detections) * 100 if total_detections > 0 else 0
            
            # Generate alerts
            alerts = []
            if slouch_count > 0:
                for k in range(min(slouch_count, 5)):  # Max 5 alerts
                    alert_time = start_time + timedelta(minutes=random.randint(1, duration_minutes-1))
                    alerts.append({
                        'timestamp': alert_time.strftime('%H:%M:%S'),
                        'message': 'Slouching detected! Please sit up straight.'
                    })
            
            session_data = {
                "session_id": f"sample_session_{i}_{j}_{start_time.strftime('%Y%m%d_%H%M%S')}",
                "timestamp": end_time.isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": duration_minutes * 60,
                "total_detections": total_detections,
                "slouch_count": slouch_count,
                "good_posture_count": good_posture_count,
                "slouch_percentage": round(slouch_percentage, 1),
                "alerts": alerts
            }
            
            sample_sessions.append(session_data)
    
    # Add sample sessions to existing data
    sessions.extend(sample_sessions)
    
    # Save updated data
    with open(sessions_file, 'w') as f:
        json.dump(sessions, f, indent=2)
    
    print("=" * 50)
    print("SAMPLE DATA ADDED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Added {len(sample_sessions)} sample sessions")
    print("Analytics dashboard will now show realistic data")
    print("Access the system at: http://localhost:5000")
    print("=" * 50)

if __name__ == "__main__":
    add_sample_data()
