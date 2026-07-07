"""
Vercel Serverless Function - Suggest Improvements
"""
from http.server import BaseHTTPRequestHandler
import json
import numpy as np

OPTIMAL_RANGES = {
    'danceability': {'min': 0.55, 'max': 0.85, 'ideal': 0.70},
    'energy': {'min': 0.50, 'max': 0.90, 'ideal': 0.70},
    'loudness': {'min': -10, 'max': -4, 'ideal': -6},
    'speechiness': {'min': 0.02, 'max': 0.15, 'ideal': 0.08},
    'acousticness': {'min': 0.0, 'max': 0.35, 'ideal': 0.15},
    'instrumentalness': {'min': 0.0, 'max': 0.05, 'ideal': 0.0},
    'liveness': {'min': 0.05, 'max': 0.25, 'ideal': 0.12},
    'valence': {'min': 0.35, 'max': 0.85, 'ideal': 0.60},
    'tempo': {'min': 95, 'max': 140, 'ideal': 120},
    'duration_ms': {'min': 150000, 'max': 270000, 'ideal': 210000}
}

FEATURE_WEIGHTS = {
    'danceability': 0.15,
    'energy': 0.12,
    'loudness': 0.14,
    'speechiness': 0.08,
    'acousticness': 0.10,
    'instrumentalness': 0.09,
    'liveness': 0.06,
    'valence': 0.11,
    'tempo': 0.08,
    'duration_ms': 0.07
}

def calculate_improvement_score(current_value, optimal):
    """Calculate how much improvement is possible for a feature"""
    ideal = optimal['ideal']
    
    if current_value is None:
        return 0, 'increase', ideal
    
    distance_from_ideal = abs(current_value - ideal)
    max_distance = max(abs(optimal['max'] - ideal), abs(ideal - optimal['min']))
    
    if max_distance == 0:
        return 0, 'none', current_value
    
    improvement_potential = distance_from_ideal / max_distance
    
    if current_value < ideal:
        return improvement_potential, 'increase', ideal
    elif current_value > ideal:
        return improvement_potential, 'decrease', ideal
    else:
        return 0, 'none', current_value


def suggest_improvements(song_data):
    """Generate improvement suggestions for a song"""
    suggestions = []
    
    for feature, weight in FEATURE_WEIGHTS.items():
        if feature not in OPTIMAL_RANGES:
            continue
            
        current = song_data.get(feature)
        if current is None:
            continue
        
        optimal = OPTIMAL_RANGES[feature]
        improvement_score, direction, suggested = calculate_improvement_score(current, optimal)
        
        if improvement_score > 0.1:  # Only suggest if meaningful improvement possible
            suggestions.append({
                'feature': feature,
                'current': round(current, 4) if isinstance(current, float) else current,
                'suggested': round(suggested, 4) if isinstance(suggested, float) else suggested,
                'direction': direction.upper(),
                'improvement_potential': round(improvement_score * weight, 4),
                'weight': weight
            })
    
    # Sort by improvement potential
    suggestions.sort(key=lambda x: x['improvement_potential'], reverse=True)
    
    return suggestions[:5]  # Return top 5 suggestions


class handler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
        return

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            if not body:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No data provided'}).encode())
                return
            
            song_data = json.loads(body.decode())
            suggestions = suggest_improvements(song_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            response = {
                'suggestions': suggestions
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
