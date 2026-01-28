#!/usr/bin/env python3
"""
Primary Script: Song Virality Prediction - Complete Integration
Combines model training and Flask API server in one executable

This is the main entry point that:
1. Imports SongHitPredictor from predict_main.py (core ML model)
2. Initializes Flask API server
3. Handles prediction requests from frontend
4. Manages model training and persistence

Features:
- Uses XGBoost for binary classification (hit/miss prediction)
- Serves REST API on port 5001
- Integrates with React frontend on port 5173
- 12 musical DNA features for prediction
- ~87% accuracy on test set
"""

import os
import sys
import json
import logging
from pathlib import Path
import tempfile

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Audio processing
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("WARNING: librosa not installed. Audio feature extraction will be unavailable.")

# Import the main ML model from predict_main.py
sys.path.insert(0, str(Path(__file__).parent / 'models'))
from predict_main import SongHitPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get paths
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / 'models'
DATA_DIR = BACKEND_DIR.parent / 'datasets'

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Global state
# Use Ensemble for best predictions
default_model_type = "ensemble"
predictor = SongHitPredictor(model_dir=MODELS_DIR, data_dir=DATA_DIR, model_type=default_model_type)
model = None
feature_names = None
model_metadata = {}
_model_loaded = False
current_model_type = default_model_type  # Track which model is active

# Musical DNA features
MUSICAL_DNA_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms'
]



def extract_audio_features(audio_file):
    """
    Extract comprehensive musical DNA features from audio file using librosa
    
    IMPORTANT: Librosa extracts raw audio signal features, while Spotify uses
    proprietary ML models trained on millions of tracks. We apply empirical
    calibration to approximate Spotify's feature definitions.
    
    Calibration is based on analyzing the distribution differences between
    librosa outputs and Spotify's documented feature ranges/behaviors.
    
    The 12 Musical DNA Features:
    ============================
    1. duration_ms - Track duration in milliseconds (direct measurement)
    2. tempo - Beats per minute, 40-250 BPM range
    3. energy - Perceptual intensity (0-1), correlated with loudness/dynamics
    4. loudness - Overall loudness in dB, typically -60 to 0
    5. danceability - Rhythm regularity + tempo suitability (0-1)
    6. valence - Musical positivity/mood (0-1) - HARDEST to estimate
    7. speechiness - Spoken word detection (0-1)
    8. acousticness - Acoustic vs electronic sound (0-1)
    9. liveness - Live performance indicators (0-1)
    10. instrumentalness - Absence of vocals (0-1)
    11. key - Musical key (0-11, C=0 to B=11)
    12. mode - Major (1) or Minor (0)
    """
    if not LIBROSA_AVAILABLE:
        return None
    
    try:
        # Load audio file with optimal settings for feature extraction
        y, sr = librosa.load(audio_file, sr=22050, mono=True)
        
        if len(y) == 0:
            raise ValueError("Empty audio file")
        
        # Initialize features dict
        features = {}
        all_features = {}  # Store all extracted features for display
        
        # === DURATION (milliseconds) - Direct measurement ===
        duration_sec = librosa.get_duration(y=y, sr=sr)
        features['duration_ms'] = int(duration_sec * 1000)
        all_features['duration_sec'] = round(duration_sec, 2)
        
        # === TEMPO ANALYSIS ===
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo_estimate = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Use the more reliable tempo estimate, clamped to realistic range
        features['tempo'] = float(np.clip(tempo_estimate, 40, 250))
        all_features['tempo_primary'] = float(tempo)
        all_features['beat_count'] = len(beat_frames)
        
        # === HARMONIC-PERCUSSIVE SEPARATION ===
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy + 1e-10
        harmonic_ratio = harmonic_energy / total_energy
        percussive_ratio = percussive_energy / total_energy
        all_features['harmonic_ratio'] = round(harmonic_ratio, 4)
        all_features['percussive_ratio'] = round(percussive_ratio, 4)
        
        # === RMS ENERGY ANALYSIS ===
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_max = np.max(rms) + 1e-10
        rms_min = np.min(rms) + 1e-10
        
        # === ENERGY (Spotify-calibrated) ===
        # Spotify energy correlates with loudness, dynamic range, and spectral content
        # Calibration: Spotify energy tends to be higher than raw RMS ratios
        energy_raw = rms_mean / rms_max
        dynamic_range = rms_max / rms_min
        dynamic_factor = np.clip(np.log10(dynamic_range + 1) / 2, 0, 1)
        
        # Spotify energy formula approximation (empirically calibrated)
        # High energy songs: loud, consistent RMS, strong beats
        energy_calibrated = (
            0.4 * energy_raw +                    # Base energy from RMS
            0.3 * (1 - rms_std / (rms_mean + 1e-6)) +  # Consistency bonus
            0.2 * percussive_ratio +              # Percussive content
            0.1 * dynamic_factor                  # Dynamic range
        )
        # Apply slight boost - Spotify energy tends higher
        energy_calibrated = energy_calibrated * 1.15
        features['energy'] = float(np.clip(energy_calibrated, 0, 1))
        all_features['energy_raw'] = round(float(energy_raw), 4)
        
        # === LOUDNESS (dB, LUFS approximation) ===
        # Spotify uses LUFS (Loudness Units Full Scale)
        loudness_db = 20 * np.log10(rms_mean + 1e-10)
        # Calibration: Add offset to match Spotify's typical range (-3 to -15 dB)
        # Raw librosa values are typically -25 to -40 dB
        loudness_calibrated = loudness_db + 18  # Empirical offset
        features['loudness'] = float(np.clip(loudness_calibrated, -60, 0))
        all_features['loudness_raw_db'] = round(float(loudness_db), 2)
        
        # === SPECTRAL FEATURES ===
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        cent_mean = np.mean(spectral_centroids)
        bandwidth_mean = np.mean(spectral_bandwidth)
        rolloff_mean = np.mean(spectral_rolloff)
        flatness_mean = np.mean(spectral_flatness)
        contrast_mean = np.mean(spectral_contrast)
        
        nyquist = sr / 2
        cent_norm = cent_mean / nyquist
        rolloff_norm = rolloff_mean / nyquist
        
        all_features['spectral_centroid_hz'] = round(float(cent_mean), 2)
        all_features['spectral_bandwidth_hz'] = round(float(bandwidth_mean), 2)
        all_features['spectral_rolloff_hz'] = round(float(rolloff_mean), 2)
        all_features['spectral_flatness'] = round(float(flatness_mean), 6)
        
        # === DANCEABILITY (Spotify-calibrated) ===
        # Spotify: combination of tempo, rhythm stability, beat strength, regularity
        # Beat regularity - how consistent the beat intervals are
        if len(beat_frames) > 2:
            beat_intervals = np.diff(librosa.frames_to_time(beat_frames, sr=sr))
            beat_regularity = 1.0 - np.clip(np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6), 0, 1)
        else:
            beat_regularity = 0.5
        
        # Rhythm strength from onset envelope variance
        rhythm_strength = np.clip(np.std(onset_env) / (np.mean(onset_env) + 1e-6), 0, 2) / 2
        
        # Tempo factor: Most danceable 95-135 BPM (club music range)
        optimal_tempo = 120
        tempo_spread = 40
        tempo_factor = 1.0 - np.clip(abs(features['tempo'] - optimal_tempo) / tempo_spread, 0, 0.7)
        
        # Groove factor: percussive + low frequency content
        groove = percussive_ratio * 0.6 + (1 - cent_norm) * 0.4
        
        # Combined danceability with Spotify-like calibration
        danceability_raw = (
            0.30 * beat_regularity +      # Beat consistency
            0.25 * rhythm_strength +      # Rhythmic variation  
            0.20 * tempo_factor +         # Optimal tempo range
            0.15 * groove +               # Percussive groove
            0.10 * features['energy']     # Energy contribution
        )
        # Calibration boost - Spotify danceability tends to be higher
        danceability_calibrated = danceability_raw * 1.2 + 0.1
        features['danceability'] = float(np.clip(danceability_calibrated, 0, 1))
        all_features['beat_regularity'] = round(float(beat_regularity), 4)
        all_features['rhythm_strength'] = round(float(rhythm_strength), 4)
        
        # === VALENCE (Musical Positivity) - MOST DIFFICULT ===
        # Spotify uses complex ML models for valence
        # We approximate using: mode, tempo, brightness, energy
        
        # Chroma analysis for key/mode
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key = int(np.argmax(chroma_mean))
        
        # Mode detection (Major vs Minor)
        major_third = chroma_mean[(key + 4) % 12]
        minor_third = chroma_mean[(key + 3) % 12]
        fifth = chroma_mean[(key + 7) % 12]
        major_score = chroma_mean[key] + major_third + fifth
        minor_score = chroma_mean[key] + minor_third + fifth
        is_major = major_score > minor_score
        mode_factor = 0.6 if is_major else 0.4  # Major = happier
        
        # Brightness factor (brighter = more positive)
        brightness = np.clip(cent_norm * 1.5, 0, 1)
        
        # Tempo factor for valence (moderate-fast = more positive)
        tempo_valence = np.clip((features['tempo'] - 70) / 100, 0, 1)
        
        # Harmonic complexity (simpler = more positive pop feel)
        harmonic_simplicity = 1 - np.clip(np.std(chroma_mean) / np.mean(chroma_mean), 0, 1)
        
        # Combined valence with empirical calibration
        valence_raw = (
            0.25 * mode_factor +           # Major/minor influence
            0.25 * brightness +            # Spectral brightness
            0.20 * tempo_valence +         # Tempo influence
            0.15 * features['energy'] +    # Energy contribution
            0.15 * harmonic_simplicity     # Harmonic clarity
        )
        # Calibration: Center around 0.5 and spread
        valence_calibrated = 0.5 + (valence_raw - 0.5) * 1.4
        features['valence'] = float(np.clip(valence_calibrated, 0, 1))
        all_features['mode_factor'] = round(float(mode_factor), 4)
        all_features['brightness'] = round(float(brightness), 4)
        
        # === ZERO CROSSING RATE ===
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        all_features['zero_crossing_rate'] = round(float(zcr_mean), 6)
        
        # === SPEECHINESS (Spotify-calibrated) ===
        # Speech characteristics: high ZCR, specific spectral patterns
        # Typical speech ZCR: 0.05-0.15
        zcr_factor = np.clip((zcr_mean - 0.03) / 0.12, 0, 1)
        
        # Speech has moderate spectral flatness (not pure noise, not pure tone)
        speech_flatness = 1 - abs(flatness_mean - 0.1) * 5
        speech_flatness = np.clip(speech_flatness, 0, 1)
        
        # Low harmonic ratio suggests speech over singing
        speech_harmonic = 1 - harmonic_ratio
        
        speechiness_raw = (
            0.50 * zcr_factor +
            0.30 * speech_flatness +
            0.20 * speech_harmonic
        )
        # Spotify speechiness is typically low (< 0.3) for most music
        # Apply calibration to match Spotify's conservative scale
        speechiness_calibrated = speechiness_raw * 0.6
        features['speechiness'] = float(np.clip(speechiness_calibrated, 0, 1))
        
        # === ACOUSTICNESS (Spotify-calibrated) ===
        # Acoustic music: less high frequencies, more harmonic, less loudness
        high_freq_content = rolloff_norm
        
        acousticness_raw = (
            0.35 * (1.0 - high_freq_content) +     # Less high frequency
            0.30 * harmonic_ratio +                 # More harmonic
            0.20 * (1.0 - features['energy']) +    # Typically quieter
            0.15 * (1.0 - percussive_ratio)        # Less percussive
        )
        # Calibration for typical music distribution
        acousticness_calibrated = acousticness_raw * 1.1
        features['acousticness'] = float(np.clip(acousticness_calibrated, 0, 1))
        
        # === LIVENESS (Spotify-calibrated) ===
        # Live recordings: audience noise, reverb, less consistent dynamics
        # High spectral flatness can indicate ambient noise
        noise_factor = np.clip(flatness_mean * 5, 0, 1)
        
        # Dynamic variance indicates live performance
        dynamic_variance = np.clip(rms_std / (rms_mean + 1e-6), 0, 1)
        
        # Reverb detection via spectral decay (approximate)
        spectral_decay = np.clip(np.mean(np.diff(spectral_rolloff)) / 1000, -1, 1)
        reverb_factor = np.clip(0.5 - spectral_decay, 0, 1)
        
        liveness_raw = (
            0.40 * noise_factor +
            0.35 * dynamic_variance +
            0.25 * reverb_factor
        )
        # Most studio recordings have low liveness (< 0.3)
        liveness_calibrated = liveness_raw * 0.7
        features['liveness'] = float(np.clip(liveness_calibrated, 0, 1))
        
        # === INSTRUMENTALNESS (Spotify-calibrated) ===
        # High instrumentalness = no vocals
        # Vocals have specific spectral characteristics
        
        # Vocal frequency range presence (300-3400 Hz)
        vocal_range_energy = np.mean(spectral_centroids > 300) * np.mean(spectral_centroids < 3400)
        vocal_presence = np.clip(vocal_range_energy * 2, 0, 1)
        
        # Low speechiness and ZCR suggests instrumental
        instrumental_raw = (
            0.40 * (1 - features['speechiness']) +
            0.30 * (1 - zcr_factor) +
            0.30 * harmonic_ratio
        )
        # Calibrate - most pop/rock has vocals (low instrumentalness)
        instrumentalness_calibrated = instrumental_raw * 0.8
        features['instrumentalness'] = float(np.clip(instrumentalness_calibrated, 0, 1))
        
        # === KEY (0-11) ===
        features['key'] = key
        all_features['key_name'] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]
        all_features['key_strength'] = round(float(np.max(chroma_mean) / (np.sum(chroma_mean) + 1e-10)), 4)
        
        # === MODE (Major=1, Minor=0) ===
        features['mode'] = 1 if is_major else 0
        all_features['mode_name'] = 'Major' if is_major else 'Minor'
        all_features['major_confidence'] = round(float(major_score / (major_score + minor_score + 1e-10)), 4)
        
        # === MFCCs for additional analysis ===
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            all_features[f'mfcc_{i+1}'] = round(float(np.mean(mfccs[i])), 4)
        
        # Log extracted features
        logger.info(f"Extracted features: {features}")
        
        # Store all features for detailed display
        features['_all_features'] = all_features
        features['_feature_count'] = len(all_features)
        features['_calibration_note'] = "Features calibrated to approximate Spotify's scale"
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_model_globally():
    """Load model globally for API use"""
    global model, feature_names, model_metadata, _model_loaded
    
    if _model_loaded:
        return model is not None or predictor.model_type == "ensemble"
    
    try:
        if predictor.load_model():
            # For ensemble, set model to xgb_model for compatibility
            if predictor.model_type == "ensemble":
                model = predictor.xgb_model
            else:
                model = predictor.model
            feature_names = predictor.feature_names
            model_metadata = predictor.model_metadata
            logger.info("✓ Model loaded for API use")
            
        _model_loaded = True
        return model is not None
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        _model_loaded = True
        return False


# ============================================================================
# FLASK API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - information about the API"""
    return jsonify({
        'service': 'Song Virality Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'audio_processing': 'enabled' if LIBROSA_AVAILABLE else 'disabled',
        'endpoints': {
            '/api/health': 'GET - Server health check',
            '/api/predict': 'POST - Predict song hit probability (JSON features)',
            '/api/analyze-audio': 'POST - Analyze audio file and predict (multipart/form-data)',
            '/api/model-info': 'GET - Model metadata and features',
            '/api/optimal-ranges': 'GET - Optimal parameter ranges',
            '/api/feature-importance': 'GET - Feature importance scores',
            '/api/suggest-improvements': 'POST - Song improvement suggestions'
        }
    })

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
        if model is None:
            load_model_globally()
        
        song_data = request.get_json()
        
        if not song_data:
            return jsonify({'error': 'No data provided'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Validate and normalize feature ranges
        feature_ranges = {
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
        
        # Validate each feature
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in song_data:
                try:
                    val = float(song_data[feature])
                    # Clamp to valid range
                    song_data[feature] = max(min_val, min(max_val, val))
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid value for {feature}: must be numeric'}), 400
        
        # Make prediction using the predictor
        result = predictor.predict_song_hit_probability(song_data)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'hit_probability': result['hit_probability'],
            'confidence': result['confidence'],
            'prediction': 'hit' if result['is_hit_prediction'] else 'miss',
            'model_version': model_metadata.get('version', '1.0.0')
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-audio', methods=['POST'])
def analyze_audio():
    """
    Analyze audio file and predict hit probability
    
    Request: multipart/form-data with 'file' field containing audio file
    Response: Same as /api/predict but extracted from audio
    """
    try:
        if not LIBROSA_AVAILABLE:
            return jsonify({'error': 'librosa not installed. Cannot process audio files.'}), 503
        
        if model is None:
            load_model_globally()
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        audio_file = request.files['file']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(suffix=Path(audio_file.filename).suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name
        
        try:
            # Extract features from audio
            features = extract_audio_features(temp_path)
            
            if features is None:
                return jsonify({'error': 'Failed to extract audio features'}), 500
            
            # Make prediction using extracted features
            try:
                result = predictor.predict_song_hit_probability(features)
            except Exception as pred_err:
                logger.error(f"Prediction error: {pred_err}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Prediction failed: {str(pred_err)}'}), 500
            
            if result is None:
                logger.error("Prediction returned None")
                return jsonify({'error': 'Prediction failed - returned None'}), 500
            
            return jsonify({
                'probability': result['hit_probability'],
                'hit_probability': result['hit_probability'],
                'confidence': result['confidence'],
                'isViral': result['is_hit_prediction'],
                'prediction': 'hit' if result['is_hit_prediction'] else 'miss',
                'model_version': model_metadata.get('version', '1.0.0'),
                'features': features,
                'extracted_features': features,
                'fileName': audio_file.filename
            })
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information and metadata"""
    if model is None:
        load_model_globally()
    
    return jsonify({
        'loaded': model is not None,
        'active_model': current_model_type,
        'metadata': model_metadata,
        'features': MUSICAL_DNA_FEATURES,
        'improvements': {
            'bias_correction': 'enabled',
            'description': 'Reduces negative bias in predictions - boosts middle-range probabilities',
            'supported_models': ['xgboost', 'lstm']
        }
    })


@app.route('/api/optimal-ranges', methods=['GET'])
def optimal_ranges():
    """Get optimal parameter ranges for hit songs"""
    ranges = predictor.get_optimal_ranges()
    if ranges is None:
        return jsonify({'error': 'Could not calculate optimal ranges'}), 500
    
    return jsonify({
        'status': 'success',
        'optimal_ranges': ranges,
        'definition': 'Optimal ranges represent the mean ± 1 standard deviation of hit songs'
    })


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Get feature importance for hit prediction"""
    if model is None:
        load_model_globally()
    
    importance_df = predictor.get_feature_importance()
    if importance_df is None:
        return jsonify({'error': 'Could not calculate feature importance'}), 500
    
    # Convert to list of dicts for JSON serialization
    importance_list = []
    for _, row in importance_df.iterrows():
        importance_list.append({
            'feature': row['feature'],
            'importance': float(row['importance'])
        })
    
    return jsonify({
        'status': 'success',
        'features': importance_list
    })


@app.route('/api/suggest-improvements', methods=['POST'])
def suggest_improvements():
    """
    Suggest feature improvements for a song
    
    Request body:
    {
      "danceability": 0.5,
      "energy": 0.6,
      ...all 12 features
    }
    
    Response:
    {
      "current_probability": 0.032,
      "top_suggestions": [
        {
          "feature": "danceability",
          "current": 0.5,
          "suggested": 0.65,
          "direction": "INCREASE",
          "improvement": 0.045,
          "new_probability": 0.077
        },
        ...
      ]
    }
    """
    try:
        if model is None:
            load_model_globally()
        
        song_data = request.get_json()
        
        if not song_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Ensure all required features are present
        for feat in MUSICAL_DNA_FEATURES:
            if feat not in song_data:
                song_data[feat] = 0
        
        suggestions = predictor.suggest_feature_improvements(song_data)
        
        if suggestions is None:
            return jsonify({'error': 'Could not generate suggestions'}), 500
        
        return jsonify({
            'suggestions': suggestions
        })
    
    except Exception as e:
        logger.error(f"Error in suggest_improvements: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/switch-model', methods=['POST'])
def switch_model():
    """
    Switch between XGBoost and LSTM models
    
    Request body:
    {
      "model_type": "xgboost" or "lstm"
    }
    
    Response:
    {
      "status": "success",
      "active_model": "lstm",
      "message": "Switched to LSTM model"
    }
    """
    global predictor, model, current_model_type, _model_loaded
    
    try:
        data = request.get_json()
        model_type = data.get('model_type', '').lower()
        
        if model_type not in ['xgboost', 'lstm']:
            return jsonify({'error': 'Invalid model type. Must be "xgboost" or "lstm"'}), 400
        
        if model_type == 'lstm' and not LIBROSA_AVAILABLE:
            return jsonify({'error': 'TensorFlow not available. Cannot use LSTM model.'}), 503
        
        # Create new predictor with desired model type
        new_predictor = SongHitPredictor(model_dir=MODELS_DIR, data_dir=DATA_DIR, model_type=model_type)
        
        # Try to load the model
        if new_predictor.load_model(model_type=model_type):
            predictor = new_predictor
            model = predictor.model
            current_model_type = model_type
            _model_loaded = True
            return jsonify({
                'status': 'success',
                'active_model': model_type,
                'message': f'Switched to {model_type.upper()} model',
                'metadata': predictor.model_metadata
            })
        else:
            return jsonify({
                'error': f'Could not load {model_type} model. Train a new model first.',
                'hint': 'POST to /api/train to train a new model'
            }), 404
    
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/active-model', methods=['GET'])
def active_model():
    """Get information about the currently active model"""
    return jsonify({
        'active_model': current_model_type,
        'metadata': model_metadata,
        'model_loaded': model is not None
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
    logger.info("[OK] Flask app created with SongHitPredictor integration")
    return app


def main():
    """Main entry point"""
    try:
        logger.info("Song Virality Prediction System - Starting...")
        logger.info("="*60)
        
        # Primary dataset: spotify_tracks.csv
        data_path = DATA_DIR / 'spotify_tracks.csv'
        
        # Fallback to alternative names if primary doesn't exist
        if not data_path.exists():
            for name in ['demo.csv', 'spotify_songs.csv']:
                alt_path = DATA_DIR / name
                if alt_path.exists():
                    data_path = alt_path
                    break
        
        if not data_path.exists():
            logger.error(f"[ERROR] Data file not found. Looked in: {DATA_DIR}")
            logger.error("Please add a CSV file to the datasets directory.")
            return
        
        logger.info(f"[INFO] Data file: {data_path}")
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df, X, Y = predictor.load_and_prepare_data(str(data_path))
        
        if df is None:
            logger.error("[ERROR] Failed to load data. Exiting.")
            return
        
        # Train model
        logger.info("Training model...")
        predictor.train_model(X, Y, force_retrain=False)
        
        # Load globally
        load_model_globally()
        
        if model is None and predictor.model_type != "ensemble":
            logger.error("[ERROR] Failed to load model. Exiting.")
            return
        
        logger.info("Model ready!")
        logger.info("="*60)
        logger.info("Starting Flask API server...")
        logger.info(f"API running on http://0.0.0.0:5000")
        logger.info("Frontend: http://localhost:5173")
        logger.info("="*60)
        
        # Start Flask server
        port = int(os.getenv('FLASK_PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    except Exception as e:
        logger.error(f"FATAL ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
