"""
Flask API server for Song Virality Prediction
Provides REST endpoints for song hit probability predictions
"""
import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the backend directory
BACKEND_DIR = Path(__file__).parent.parent
MODELS_DIR = BACKEND_DIR / 'models'

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Lazy-loaded globals
model = None
feature_names = None
model_metadata = {}
_model_loaded = False

def load_model():
    """Load the trained XGBoost model (lazy loading)"""
    global model, feature_names, model_metadata, _model_loaded
    
    if _model_loaded:
        return model is not None
    
    try:
        # Import pickle here to avoid slow imports on startup
        import pickle
        
        # Load the pickle model
        model_path = MODELS_DIR / 'song_hit_model.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✓ Loaded model from {model_path}")
        
        # Load feature names
        features_path = MODELS_DIR / 'song_hit_model_features.pkl'
        if features_path.exists():
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            logger.info(f"✓ Loaded feature names from {features_path}")
        
        # Load metadata
        metadata_path = MODELS_DIR / 'model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"✓ Loaded metadata from {metadata_path}")
        
        _model_loaded = True
        return True
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        _model_loaded = True
        return False

def prepare_features(song_data):
    """
    Prepare song features for prediction
    Handles feature normalization and ordering
    """
    # Expected feature names in order
    expected_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms'
    ]
    
    # Extract features in correct order
    X = []
    for feat in expected_features:
        value = song_data.get(feat, 0)
        X.append(float(value))
    
    return np.array([X])

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict song hit probability
    
    Request body:
    {
      "danceability": 0.65,
      "energy": 0.72,
      "key": 5,
      "loudness": -6.5,
      "mode": 1,
      "speechiness": 0.08,
      "acousticness": 0.25,
      "instrumentalness": 0.05,
      "liveness": 0.15,
      "valence": 0.58,
      "tempo": 125,
      "duration_ms": 210000
    }
    
    Response:
    {
      "hit_probability": 0.732,
      "confidence": 0.85,
      "prediction": "hit" | "miss",
      "model_version": "1.0.0"
    }
    """
    try:
        # Lazy load model on first request
        if model is None:
            load_model()
        
        song_data = request.get_json()
        
        if not song_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Prepare features
        X = prepare_features(song_data)
        
        # Make prediction
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'fallback': True
            }), 503
        
        # Get probability
        proba = model.predict_proba(X)[0]
        hit_probability = float(proba[1])  # Probability of hit class
        
        # Determine prediction
        prediction = 'hit' if hit_probability > 0.5 else 'miss'
        
        # Calculate confidence
        confidence = float(max(proba))
        
        return jsonify({
            'hit_probability': hit_probability,
            'confidence': confidence,
            'prediction': prediction,
            'model_version': model_metadata.get('version', '1.0.0'),
            'timestamp': model_metadata.get('training_timestamp', 'unknown')
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    return jsonify({
        'loaded': model is not None,
        'metadata': model_metadata,
        'features': feature_names if feature_names else []
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

def create_app():
    """Create and configure Flask app"""
    # Don't load model on startup - use lazy loading on first request
    logger.info("✓ Flask app created (model will load on first request)")
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Development server
    port = os.getenv('FLASK_PORT', 5000)
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    logger.info(f"Starting API server on port {port}...")
    app.run(host='0.0.0.0', port=int(port), debug=debug, threaded=True)
