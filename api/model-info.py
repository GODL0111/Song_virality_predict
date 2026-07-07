"""
Vercel Serverless Function - Model Information
"""
from http.server import BaseHTTPRequestHandler
import json

MUSICAL_DNA_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'loaded': True,
            'active_model': 'serverless-v1',
            'metadata': {
                'model_type': 'Feature-Based Predictor',
                'version': '1.0.0-serverless',
                'description': 'Lightweight prediction model optimized for serverless deployment'
            },
            'features': MUSICAL_DNA_FEATURES,
            'improvements': {
                'bias_correction': 'enabled',
                'description': 'Calibrated predictions for balanced hit/miss distribution'
            }
        }
        
        self.wfile.write(json.dumps(response).encode())
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        return
