"""
Vercel Serverless Function - Root API Info
"""
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'service': 'Song Virality Prediction API',
            'version': '1.0.0-serverless',
            'status': 'running',
            'environment': 'vercel',
            'endpoints': {
                '/api/health': 'GET - Server health check',
                '/api/predict': 'POST - Predict song hit probability (JSON features)',
                '/api/model-info': 'GET - Model metadata and features',
                '/api/optimal-ranges': 'GET - Optimal parameter ranges',
                '/api/feature-importance': 'GET - Feature importance scores',
                '/api/suggest-improvements': 'POST - Song improvement suggestions'
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
