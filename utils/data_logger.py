"""
Data Logger Utility
Handles session data logging, analytics, and data export
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class DataLogger:
    def __init__(self, data_dir="data"):
        """Initialize the data logger"""
        self.data_dir = data_dir
        self.sessions_file = os.path.join(data_dir, "sessions.json")
        self.analytics_file = os.path.join(data_dir, "analytics.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize sessions data
        self.sessions = self._load_sessions()
    
    def _load_sessions(self):
        """Load existing sessions from file"""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading sessions: {e}")
                return []
        return []
    
    def _save_sessions(self):
        """Save sessions to file"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(self.sessions, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving sessions: {e}")
    
    def log_session(self, session_data):
        """Log a new session"""
        # Add session ID and timestamp
        session_id = f"session_{len(self.sessions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_record = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'start_time': session_data.get('start_time', datetime.now()).isoformat(),
            'end_time': session_data.get('end_time', datetime.now()).isoformat(),
            'duration': session_data.get('duration', 0),
            'total_detections': session_data.get('total_detections', 0),
            'slouch_count': session_data.get('slouch_count', 0),
            'good_posture_count': session_data.get('good_posture_count', 0),
            'slouch_percentage': 0,
            'alerts': session_data.get('alerts', [])
        }
        
        # Calculate slouch percentage
        if session_record['total_detections'] > 0:
            session_record['slouch_percentage'] = (
                session_record['slouch_count'] / session_record['total_detections'] * 100
            )
        
        # Add to sessions list
        self.sessions.append(session_record)
        
        # Save to file
        self._save_sessions()
        
        print(f"Session logged: {session_id}")
        return session_id
    
    def get_analytics(self):
        """Get comprehensive analytics data"""
        if not self.sessions:
            return self._get_empty_analytics()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.sessions)
        
        # Basic statistics
        total_sessions = len(df)
        total_duration = df['duration'].sum()
        total_detections = df['total_detections'].sum()
        total_slouch_count = df['slouch_count'].sum()
        total_good_posture_count = df['good_posture_count'].sum()
        
        # Average statistics
        avg_session_duration = df['duration'].mean()
        avg_slouch_percentage = df['slouch_percentage'].mean()
        avg_detections_per_session = df['total_detections'].mean()
        
        # Recent sessions (last 7 days)
        recent_date = datetime.now() - timedelta(days=7)
        recent_sessions = df[pd.to_datetime(df['timestamp']) >= recent_date]
        
        # Daily statistics
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_stats = df.groupby('date').agg({
            'duration': 'sum',
            'total_detections': 'sum',
            'slouch_count': 'sum',
            'good_posture_count': 'sum',
            'slouch_percentage': 'mean'
        }).reset_index()
        
        # Posture improvement trend
        df_sorted = df.sort_values('timestamp')
        posture_trend = df_sorted['slouch_percentage'].tolist()
        session_numbers = list(range(1, len(df_sorted) + 1))
        
        # Alert analysis
        all_alerts = []
        for session in self.sessions:
            all_alerts.extend(session.get('alerts', []))
        
        alert_count = len(all_alerts)
        
        analytics = {
            'summary': {
                'total_sessions': total_sessions,
                'total_duration_hours': round(total_duration / 3600, 2),
                'total_detections': total_detections,
                'total_slouch_count': total_slouch_count,
                'total_good_posture_count': total_good_posture_count,
                'overall_slouch_percentage': round(
                    (total_slouch_count / total_detections * 100) if total_detections > 0 else 0, 2
                )
            },
            'averages': {
                'avg_session_duration_minutes': round(avg_session_duration / 60, 2),
                'avg_slouch_percentage': round(avg_slouch_percentage, 2),
                'avg_detections_per_session': round(avg_detections_per_session, 2)
            },
            'recent_activity': {
                'sessions_last_7_days': len(recent_sessions),
                'detections_last_7_days': recent_sessions['total_detections'].sum(),
                'slouch_count_last_7_days': recent_sessions['slouch_count'].sum()
            },
            'daily_stats': daily_stats.to_dict('records'),
            'posture_trend': {
                'session_numbers': session_numbers,
                'slouch_percentages': posture_trend
            },
            'alerts': {
                'total_alerts': alert_count,
                'recent_alerts': all_alerts[-10:] if len(all_alerts) > 10 else all_alerts
            },
            'recommendations': self._generate_recommendations(df)
        }
        
        return analytics
    
    def _get_empty_analytics(self):
        """Return empty analytics structure"""
        return {
            'summary': {
                'total_sessions': 0,
                'total_duration_hours': 0,
                'total_detections': 0,
                'total_slouch_count': 0,
                'total_good_posture_count': 0,
                'overall_slouch_percentage': 0
            },
            'averages': {
                'avg_session_duration_minutes': 0,
                'avg_slouch_percentage': 0,
                'avg_detections_per_session': 0
            },
            'recent_activity': {
                'sessions_last_7_days': 0,
                'detections_last_7_days': 0,
                'slouch_count_last_7_days': 0
            },
            'daily_stats': [],
            'posture_trend': {
                'session_numbers': [],
                'slouch_percentages': []
            },
            'alerts': {
                'total_alerts': 0,
                'recent_alerts': []
            },
            'recommendations': []
        }
    
    def _generate_recommendations(self, df):
        """Generate personalized recommendations based on data"""
        recommendations = []
        
        avg_slouch_percentage = df['slouch_percentage'].mean()
        
        if avg_slouch_percentage > 50:
            recommendations.append({
                'type': 'warning',
                'message': 'High slouching detected. Consider taking more frequent breaks and practicing good posture exercises.',
                'priority': 'high'
            })
        elif avg_slouch_percentage > 30:
            recommendations.append({
                'type': 'info',
                'message': 'Moderate slouching detected. Try to maintain better posture during work sessions.',
                'priority': 'medium'
            })
        else:
            recommendations.append({
                'type': 'success',
                'message': 'Great posture! Keep up the good work.',
                'priority': 'low'
            })
        
        # Session duration recommendations
        avg_duration = df['duration'].mean()
        if avg_duration > 3600:  # More than 1 hour
            recommendations.append({
                'type': 'info',
                'message': 'Consider taking breaks every 30-45 minutes to maintain good posture.',
                'priority': 'medium'
            })
        
        # Consistency recommendations
        if len(df) >= 3:
            recent_trend = df.tail(3)['slouch_percentage'].mean()
            older_trend = df.head(3)['slouch_percentage'].mean()
            
            if recent_trend < older_trend:
                recommendations.append({
                    'type': 'success',
                    'message': 'Excellent improvement in posture! Your efforts are paying off.',
                    'priority': 'low'
                })
        
        return recommendations
    
    def export_all_data(self):
        """Export all session data"""
        return {
            'sessions': self.sessions,
            'analytics': self.get_analytics(),
            'export_timestamp': datetime.now().isoformat(),
            'total_sessions': len(self.sessions)
        }
    
    def clear_data(self):
        """Clear all session data"""
        self.sessions = []
        self._save_sessions()
        print("All session data cleared.")
    
    def get_session_by_id(self, session_id):
        """Get a specific session by ID"""
        for session in self.sessions:
            if session['session_id'] == session_id:
                return session
        return None
    
    def get_recent_sessions(self, limit=10):
        """Get recent sessions"""
        return self.sessions[-limit:] if self.sessions else []
