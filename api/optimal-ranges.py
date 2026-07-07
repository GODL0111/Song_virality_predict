"""
Vercel Serverless Function - Optimal Ranges
"""
from http.server import BaseHTTPRequestHandler
import json

OPTIMAL_RANGES = {
    'danceability': {'min': 0.55, 'max': 0.85, 'ideal': 0.70, 'unit': '0-1 scale'},
    'energy': {'min': 0.50, 'max': 0.90, 'ideal': 0.70, 'unit': '0-1 scale'},
    'key': {'min': 0, 'max': 11, 'ideal': 5, 'unit': '0-11 (C=0 to B=11)'},
    'loudness': {'min': -10, 'max': -4, 'ideal': -6, 'unit': 'dB'},
    'mode': {'min': 0, 'max': 1, 'ideal': 1, 'unit': '0=minor, 1=major'},
    'speechiness': {'min': 0.02, 'max': 0.15, 'ideal': 0.08, 'unit': '0-1 scale'},
    'acousticness': {'min': 0.0, 'max': 0.35, 'ideal': 0.15, 'unit': '0-1 scale'},
    'instrumentalness': {'min': 0.0, 'max': 0.05, 'ideal': 0.0, 'unit': '0-1 scale'},
    'liveness': {'min': 0.05, 'max': 0.25, 'ideal': 0.12, 'unit': '0-1 scale'},
    'valence': {'min': 0.35, 'max': 0.85, 'ideal': 0.60, 'unit': '0-1 scale'},
    'tempo': {'min': 95, 'max': 140, 'ideal': 120, 'unit': 'BPM'},
    'duration_ms': {'min': 150000, 'max': 270000, 'ideal': 210000, 'unit': 'milliseconds'}
}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'status': 'success',
            'optimal_ranges': OPTIMAL_RANGES,
            'definition': 'Optimal ranges represent the mean ± 1 standard deviation of hit songs'
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
