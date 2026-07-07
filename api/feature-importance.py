"""
Vercel Serverless Function - Feature Importance
"""
from http.server import BaseHTTPRequestHandler
import json

# Feature importance scores (from trained model analysis)
FEATURE_IMPORTANCE = [
    {'feature': 'danceability', 'importance': 0.152},
    {'feature': 'loudness', 'importance': 0.141},
    {'feature': 'energy', 'importance': 0.124},
    {'feature': 'valence', 'importance': 0.113},
    {'feature': 'acousticness', 'importance': 0.098},
    {'feature': 'instrumentalness', 'importance': 0.092},
    {'feature': 'speechiness', 'importance': 0.081},
    {'feature': 'tempo', 'importance': 0.076},
    {'feature': 'duration_ms', 'importance': 0.068},
    {'feature': 'liveness', 'importance': 0.055}
]

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'status': 'success',
            'features': FEATURE_IMPORTANCE
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
