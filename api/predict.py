"""
Vercel Serverless Function - Song Hit Prediction
"""
from http.server import BaseHTTPRequestHandler
import json
import numpy as np

# Musical DNA features
MUSICAL_DNA_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]

# Feature ranges for validation
FEATURE_RANGES = {
    'danceability': (0, 1),
    'energy': (0, 1),
    'key': (0, 11),
    'loudness': (-60, 0),
    'mode': (0, 1),
    'speechiness': (0, 1),
    'acousticness': (0, 1),
    'instrumentalness': (0, 1),
    'liveness': (0, 1),
    'valence': (0, 1),
    'tempo': (0, 250),
    'duration_ms': (0, 3600000)
}

# Optimal ranges for hit songs (from training data analysis)
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

# Feature weights for prediction (approximated from trained model)
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


def calculate_feature_score(value, optimal):
    """Calculate how close a feature value is to optimal range"""
    if value is None:
        return 0.5
    
    ideal = optimal['ideal']
    min_val = optimal['min']
    max_val = optimal['max']
    
    if min_val <= value <= max_val:
        # Within optimal range - calculate distance from ideal
        if value == ideal:
            return 1.0
        elif value < ideal:
            return 0.7 + 0.3 * (value - min_val) / (ideal - min_val)
        else:
            return 0.7 + 0.3 * (max_val - value) / (max_val - ideal)
    else:
        # Outside optimal range
        if value < min_val:
            distance = (min_val - value) / min_val if min_val != 0 else abs(min_val - value)
            return max(0.1, 0.6 - distance * 0.5)
        else:
            distance = (value - max_val) / max_val if max_val != 0 else abs(value - max_val)
            return max(0.1, 0.6 - distance * 0.5)


def predict_hit_probability(features):
    """
    Predict song hit probability using feature analysis
    This is a simplified model suitable for serverless deployment
    """
    total_score = 0
    total_weight = 0
    feature_scores = {}
    
    for feature, weight in FEATURE_WEIGHTS.items():
        if feature in features and feature in OPTIMAL_RANGES:
            score = calculate_feature_score(features[feature], OPTIMAL_RANGES[feature])
            feature_scores[feature] = score
            total_score += score * weight
            total_weight += weight
    
    # Calculate base probability
    if total_weight > 0:
        base_prob = total_score / total_weight
    else:
        base_prob = 0.5
    
    # Apply calibration curve (sigmoid-like transformation)
    # This makes the model more discriminative
    calibrated_prob = 1 / (1 + np.exp(-5 * (base_prob - 0.5)))
    
    # Calculate confidence based on feature coverage and consistency
    coverage = len(feature_scores) / len(FEATURE_WEIGHTS)
    score_variance = np.var(list(feature_scores.values())) if feature_scores else 0.5
    confidence = min(0.95, coverage * (1 - score_variance * 0.5))
    
    return {
        'hit_probability': float(round(calibrated_prob, 4)),
        'confidence': float(round(confidence, 3)),
        'is_hit_prediction': calibrated_prob > 0.5,
        'feature_scores': feature_scores
    }


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
            # Read request body
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
            
            # Validate and clamp feature values
            for feature, (min_val, max_val) in FEATURE_RANGES.items():
                if feature in song_data:
                    try:
                        val = float(song_data[feature])
                        song_data[feature] = max(min_val, min(max_val, val))
                    except (ValueError, TypeError):
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.send_cors_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            'error': f'Invalid value for {feature}: must be numeric'
                        }).encode())
                        return
            
            # Make prediction
            result = predict_hit_probability(song_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            
            response = {
                'hit_probability': result['hit_probability'],
                'confidence': result['confidence'],
                'prediction': 'hit' if result['is_hit_prediction'] else 'miss',
                'model_version': '1.0.0-serverless'
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
