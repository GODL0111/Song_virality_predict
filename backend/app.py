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
import uuid

# Flask imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd

# Audio processing
try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("WARNING: librosa not installed. Audio feature extraction will be unavailable.")

try:
    import soundfile as sf
except ImportError:
    print("WARNING: soundfile not installed. Cannot export mutated audio.")

try:
    import pedalboard
    from pedalboard import Pedalboard, PitchShift, Gain, time_stretch
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False
    print("WARNING: pedalboard not installed.")

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

# Security: CORS
frontend_url = os.environ.get('FRONTEND_URL', '*')
CORS(app, origins=[frontend_url] if frontend_url != '*' else '*')

# Security: Max upload size 15MB
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024 

# Security: Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day"],
    storage_uri="memory://"
)

# Setup User Database (Excel based)
USERS_DB_PATH = BACKEND_DIR / 'data' / 'users.xlsx'

def init_user_db():
    if not USERS_DB_PATH.exists():
        df = pd.DataFrame(columns=['id', 'username', 'email', 'password', 'created_at', 'auth_provider'])
        df.to_excel(USERS_DB_PATH, index=False)
        logger.info(f"Initialized new Excel user database at {USERS_DB_PATH}")

init_user_db()

def get_users_df():
    return pd.read_excel(USERS_DB_PATH)

def save_users_df(df):
    df.to_excel(USERS_DB_PATH, index=False)

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

# Hook Detection V3.0 Configurable Weights (Normalized 0-1)
HOOK_CONFIG = {
    'golden_hook': {
        'step_1_chorus_candidates': 5, # Select top 5 structurally repetitive sections
        'step_2_energy': 0.60,         # Rank by Energy
        'step_2_loudness': 0.40        # Rank by Loudness
    },
    'rhythm_hook': {
        'beat_density': 0.50,
        'beat_regularity': 0.30,
        'energy': 0.20
    },
    'high_energy_hook': {
        'energy': 0.60,
        'loudness': 0.25,
        'novelty': 0.15
    }
}



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
            
        return extract_features_from_array(y, sr)
    except Exception as e:
        logger.error(f"Error extracting features from {audio_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_features_from_array(y, sr):
    """
    Extract features directly from a loaded audio array.
    """
    try:
        # Initialize features dict
        features = {}
        all_features = {}  # Store all extracted features for display
        
        # === DURATION (milliseconds) - Direct measurement ===
        duration_sec = librosa.get_duration(y=y, sr=sr)
        features['duration_ms'] = int(duration_sec * 1000)
        all_features['duration_sec'] = round(float(duration_sec), 2)
        
        # === TEMPO ANALYSIS ===
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo_estimate = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Use the more reliable tempo estimate, clamped to realistic range
        features['tempo'] = float(np.clip(tempo_estimate, 40, 250))
        all_features['tempo_primary'] = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        all_features['beat_count'] = len(beat_frames)
        
        # === HARMONIC-PERCUSSIVE SEPARATION ===
        stft_complex = librosa.stft(y)
        S = np.abs(stft_complex)
        
        # HPSS using the precomputed STFT to save redundant STFT computation
        S_harm, S_perc = librosa.decompose.hpss(stft_complex)
        y_harmonic = librosa.istft(S_harm)
        y_percussive = librosa.istft(S_perc)
        
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy + 1e-10
        harmonic_ratio = harmonic_energy / total_energy
        percussive_ratio = percussive_energy / total_energy
        all_features['harmonic_ratio'] = round(float(harmonic_ratio), 4)
        all_features['percussive_ratio'] = round(float(percussive_ratio), 4)
        
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
        # NO artificial boost - let raw values through
        features['energy'] = float(np.clip(energy_calibrated, 0, 1))
        all_features['energy_raw'] = round(float(energy_raw), 4)
        
        # === LOUDNESS (dB, LUFS approximation) ===
        # Spotify uses LUFS (Loudness Units Full Scale)
        # Commercial music: -5 to -14 dB, Amateur: -20 to -40 dB
        loudness_db = 20 * np.log10(rms_mean + 1e-10)
        # Calibration: Shift but keep full range to differentiate amateur vs pro
        loudness_calibrated = loudness_db + 15  # Moderate offset
        # Allow wider range to differentiate amateur (quieter) from commercial (louder)
        features['loudness'] = float(np.clip(loudness_calibrated, -40, -3))
        all_features['loudness_raw_db'] = round(float(loudness_db), 2)
        
        # === SPECTRAL FEATURES ===
        spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
        spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(S=S)[0]
        
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
        # NO artificial boost - amateur recordings should have lower danceability
        features['danceability'] = float(np.clip(danceability_raw, 0, 1))
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
        # IMPORTANT: Most commercial pop/rock has LOW acousticness (< 0.3)
        high_freq_content = rolloff_norm
        
        acousticness_raw = (
            0.35 * (1.0 - high_freq_content) +     # Less high frequency
            0.30 * harmonic_ratio +                 # More harmonic
            0.20 * (1.0 - features['energy']) +    # Typically quieter
            0.15 * (1.0 - percussive_ratio)        # Less percussive
        )
        # CALIBRATION: Commercial music is mostly NOT acoustic
        # Reduce the base value significantly
        acousticness_calibrated = acousticness_raw * 0.4  # Reduced from 1.1
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
        # IMPORTANT: Most pop songs have vocals, so instrumentalness should be LOW
        
        # Vocal frequency range presence (300-3400 Hz)
        vocal_range_energy = np.mean(spectral_centroids > 300) * np.mean(spectral_centroids < 3400)
        vocal_presence = np.clip(vocal_range_energy * 2, 0, 1)
        
        # Low speechiness and ZCR suggests instrumental
        instrumental_raw = (
            0.40 * (1 - features['speechiness']) +
            0.30 * (1 - zcr_factor) +
            0.30 * harmonic_ratio
        )
        # CALIBRATION: Most pop/rock has vocals = LOW instrumentalness
        # Reduce significantly - typical value for vocal tracks is 0.0 to 0.1
        instrumentalness_calibrated = instrumental_raw * 0.15  # Reduced from 0.8
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
        # Skipped to improve performance, as they are not used by the ML model.
        # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # for i in range(13):
        #     all_features[f'mfcc_{i+1}'] = round(float(np.mean(mfccs[i])), 4)
        
        # Log extracted features
        logger.info(f"Extracted features: {features}")
        
        # Store all features for detailed display
        features['_all_features'] = all_features
        features['_feature_count'] = len(all_features)
        features['_calibration_note'] = "Features calibrated to approximate Spotify's scale"
        # Try to clean up warnings about empty arrays
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fill NaNs with means or defaults
            for k, v in features.items():
                if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)):
                    logger.warning(f"Feature {k} was NaN/Inf, using default")
                    features[k] = 0.5  # safe default for 0-1 range features
                    
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features from array: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_chorus_similarity(y, sr, start_sample, end_sample, full_chroma):
    """
    Calculate how similar a given segment is to the rest of the track's high-energy sections.
    """
    chunk_chroma = librosa.feature.chroma_stft(y=y[start_sample:end_sample], sr=sr)
    # Simple similarity based on mean chroma vector distance to the global mean chroma
    chunk_mean = np.mean(chunk_chroma, axis=1)
    full_mean = np.mean(full_chroma, axis=1)
    
    # Cosine similarity
    dot = np.dot(chunk_mean, full_mean)
    norm = np.linalg.norm(chunk_mean) * np.linalg.norm(full_mean)
    if norm == 0:
        return 0.0
    return max(0.0, float(dot / norm))


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
@limiter.limit("10 per minute")
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
        
        # Extract FULL features from audio
        y, sr = librosa.load(temp_path, sr=22050, mono=True)
        if len(y) == 0:
            raise ValueError("Empty audio file")
            
        features = extract_features_from_array(y, sr)
        
        # Add target_year if provided
        target_year = request.form.get('target_year', type=int)
        if target_year is not None:
            features['target_year'] = target_year
        else:
            features['target_year'] = 2024
        
        # Precompute DSP elements for cache
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo_track, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Cache the arrays for Phase 2 Hook Analysis
        analysis_id = str(uuid.uuid4())
        cache_path = os.path.join(tempfile.gettempdir(), f'cache_{analysis_id}.npz')
        
        # Precompute global features for hook analysis
        rms = librosa.feature.rms(y=y)[0]
        full_chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        np.savez(cache_path, y=y, sr=sr, beat_times=beat_times, onset_env=onset_env, rms=rms, full_chroma=full_chroma)
        
        # Calculate Global Confidence based on prediction
        result = predictor.predict_song_hit_probability(features)
        
        if result is None:
            logger.error("Prediction returned None")
            return jsonify({'error': 'Prediction failed - returned None'}), 500
            
        # Get prescriptive suggestions for improvement
        prescriptions = []
        if result['hit_probability'] < 0.95: # Suggest improvements if not already perfect
            prescriptions = predictor.suggest_feature_improvements(features)
        
        return jsonify({
            'probability': result['hit_probability'],
            'hit_probability': result['hit_probability'],
            'confidence': result['confidence'],
            'isViral': result['is_hit_prediction'],
            'prediction': 'hit' if result['is_hit_prediction'] else 'miss',
            'model_version': getattr(predictor, 'model_metadata', {}).get('version', '1.0.0'),
            'features': features,
            'prescriptions': prescriptions,
            'total_duration_sec': librosa.get_duration(y=y, sr=sr),
            'analysisId': analysis_id
        })
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/analyze-hooks', methods=['POST'])
@limiter.limit("10 per minute")
def analyze_hooks():
    try:
        data = request.json
        if not data or 'analysisId' not in data:
            return jsonify({'error': 'No analysisId provided'}), 400
            
        analysis_id = data['analysisId']
        cache_path = os.path.join(tempfile.gettempdir(), f'cache_{analysis_id}.npz')
        
        if not os.path.exists(cache_path):
            return jsonify({'error': 'Analysis cache expired or invalid. Please re-upload the song.'}), 404
            
        # Load precomputed arrays and ensure the file handle is closed
        with np.load(cache_path) as npz:
            y = npz['y']
            sr = int(npz['sr'])
            beat_times = npz['beat_times']
            onset_env = npz['onset_env']
            rms_global = npz['rms']
            full_chroma = npz['full_chroma']
        
        total_duration_sec = librosa.get_duration(y=y, sr=sr)
        
        # Calculate Global features needed for slicing
        # (Loaded from cache above)
        # rms_global = librosa.feature.rms(y=y)[0]
        # full_chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        window_sec = 15.0
        stride_sec = 5.0
        
        raw_segments = []
        
        # 1. Slice and Extract Raw Metrics
        for start_sec in np.arange(0, total_duration_sec - window_sec, stride_sec):
            start_idx = np.argmin(np.abs(beat_times - start_sec))
            snapped_start = beat_times[start_idx]
            
            end_idx = np.argmin(np.abs(beat_times - (snapped_start + window_sec)))
            snapped_end = beat_times[end_idx]
            
            if snapped_end - snapped_start < 5.0:
                continue
                
            start_sample = int(snapped_start * sr)
            end_sample = int(snapped_end * sr)
            
            start_frame = int(snapped_start * sr / 512)
            end_frame = int(snapped_end * sr / 512)
            
            if end_frame > start_frame:
                chunk_rms_mean = np.mean(rms_global[start_frame:end_frame])
                chunk_onset_mean = np.mean(onset_env[start_frame:end_frame])
            else:
                chunk_rms_mean = 0.0
                chunk_onset_mean = 0.0
            
            # Beats in chunk
            beats_in_chunk = [b for b in beat_times if snapped_start <= b <= snapped_end]
            beat_density_raw = len(beats_in_chunk)
            
            # Beat regularity (variance of beat intervals)
            if len(beats_in_chunk) > 2:
                intervals = np.diff(beats_in_chunk)
                beat_reg_raw = 1.0 / (np.var(intervals) + 1e-6) # Inverse of variance
            else:
                beat_reg_raw = 0.0
                
            # Chorus similarity
            chorus_sim = calculate_chorus_similarity(y, sr, start_sample, end_sample, full_chroma)
            
            raw_segments.append({
                'start_time': round(snapped_start, 1),
                'end_time': round(snapped_end, 1),
                'energy_raw': chunk_rms_mean,
                'loudness_raw': 20 * np.log10(chunk_rms_mean + 1e-10),
                'novelty_raw': chunk_onset_mean,
                'beat_density_raw': beat_density_raw,
                'beat_reg_raw': beat_reg_raw,
                'chorus_sim': chorus_sim
            })
            
        if not raw_segments:
            return jsonify({'temporal_segments': [], 'top_hooks': []})
            
        # 2. Min-Max Normalization
        import math
        def normalize_metric(metric):
            vals = [seg.get(metric, 0.0) for seg in raw_segments]
            # Filter out NaNs if any
            clean_vals = [v if not math.isnan(v) else 0.0 for v in vals]
            if not clean_vals:
                return [0.0 for _ in vals]
            min_v, max_v = min(clean_vals), max(clean_vals)
            if max_v - min_v <= 1e-9:
                return [0.0 for _ in vals]
            return [(v - min_v) / (max_v - min_v) for v in clean_vals]
            
        norm_energy = normalize_metric('energy_raw')
        norm_loudness = normalize_metric('loudness_raw')
        norm_novelty = normalize_metric('novelty_raw')
        norm_beat_density = normalize_metric('beat_density_raw')
        norm_beat_reg = normalize_metric('beat_reg_raw')
        norm_chorus_sim = normalize_metric('chorus_sim')
        
        temporal_segments = []
        for i, seg in enumerate(raw_segments):
            
            # Step 2: Rank candidates within Chorus bounds
            # We assign a pure energy/loudness score to all, but Golden Hook will only be selected
            # from the top 5 normalized chorus regions.
            hook_score = (
                HOOK_CONFIG['golden_hook']['step_2_energy'] * norm_energy[i] + 
                HOOK_CONFIG['golden_hook']['step_2_loudness'] * norm_loudness[i]
            )
            
            rhythm_score = (
                HOOK_CONFIG['rhythm_hook']['beat_density'] * norm_beat_density[i] +
                HOOK_CONFIG['rhythm_hook']['beat_regularity'] * norm_beat_reg[i] +
                HOOK_CONFIG['rhythm_hook']['energy'] * norm_energy[i]
            )
            
            high_energy_score = (
                HOOK_CONFIG['high_energy_hook']['energy'] * norm_energy[i] +
                HOOK_CONFIG['high_energy_hook']['loudness'] * norm_loudness[i] +
                HOOK_CONFIG['high_energy_hook']['novelty'] * norm_novelty[i]
            )
            
            temporal_segments.append({
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'hook_score': float(hook_score),
                'rhythm_score': float(rhythm_score),
                'high_energy_score': float(high_energy_score),
                'energy': float(norm_energy[i]),
                'novelty': float(norm_novelty[i]),
                'norm_chorus_sim': float(norm_chorus_sim[i])
            })
            
        # 3. Extract Top Hooks (Non-overlapping)
        top_hooks = []
        def is_overlapping(seg1, seg2):
            return not (seg1['end_time'] <= seg2['start_time'] or seg1['start_time'] >= seg2['end_time'])
            
        # --- HIERARCHICAL GOLDEN HOOK ---
        # 1. Structural Filter: Top N most repetitive structural sections
        sorted_by_chorus = sorted(temporal_segments, key=lambda x: x['norm_chorus_sim'], reverse=True)
        top_n = HOOK_CONFIG['golden_hook']['step_1_chorus_candidates']
        top_5_chorus_candidates = sorted_by_chorus[:top_n]
        
        # 2. Excitement Filter: Most energetic rendition among the 5
        golden = max(top_5_chorus_candidates, key=lambda x: x['hook_score'])
        golden_hook = {**golden, 'type': 'Golden Hook', 'description': 'Best overall viral potential (Chorus)'}
        top_hooks.append(golden_hook)
        
        rhythm_cands = [s for s in temporal_segments if not is_overlapping(s, golden_hook)]
        if rhythm_cands:
            rhythm = max(rhythm_cands, key=lambda x: x['rhythm_score'])
            top_hooks.append({**rhythm, 'hook_score': rhythm['rhythm_score'], 'type': 'Rhythm Hook', 'description': 'Most engaging & steady rhythm'})
            
            drop_cands = [s for s in rhythm_cands if not is_overlapping(s, rhythm)]
            if drop_cands:
                drop = max(drop_cands, key=lambda x: x['high_energy_score'])
                top_hooks.append({**drop, 'hook_score': drop['high_energy_score'], 'type': 'High-Energy Drop', 'description': 'Biggest energy spike / drop'})
                
        # Clean up cache
        os.remove(cache_path)
        
        # Log final scores
        logger.info(f"--- HOOK SCORES FOR ANALYSIS {analysis_id} ---")
        for h in top_hooks:
            logger.info(
                f"[{h.get('type')}] "
                f"Hook Score: {h.get('hook_score', 0):.3f} | "
                f"Rhythm Score: {h.get('rhythm_score', 0):.3f} | "
                f"Energy Score: {h.get('energy', 0):.3f} | "
                f"Novelty Score: {h.get('novelty', 0):.3f} | "
                f"Chorus Sim: {h.get('norm_chorus_sim', 0):.3f}"
            )
            
        
        return jsonify({
            'temporal_segments': temporal_segments,
            'top_hooks': top_hooks
        })
    except Exception as e:
        logger.error(f"Error in analyze_hooks: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/api/mutate-audio', methods=['POST'])
@limiter.limit("10 per minute")
def mutate_audio():
    """Mutate audio based on playground features"""
    try:
        data = request.json
        if not data or 'analysisId' not in data:
            return jsonify({'error': 'No analysisId provided'}), 400
            
        analysis_id = data['analysisId']
        cache_path = os.path.join(tempfile.gettempdir(), f'cache_{analysis_id}.npz')
        
        if not os.path.exists(cache_path):
            return jsonify({'error': 'Analysis cache expired. Please re-upload.'}), 404
            
        target_tempo = data.get('target_tempo')
        target_key = data.get('target_key')
        target_loudness = data.get('target_loudness')
        
        # Load audio from cache
        with np.load(cache_path) as npz:
            y = npz['y']
            sr = int(npz['sr'])
            
        original_tempo = data.get('original_tempo', 120)
        original_key = data.get('original_key', 0)
        original_loudness = data.get('original_loudness', -6)
        
        # 1 & 2. TIME STRETCH AND PITCH SHIFT
        rate = 1.0
        if target_tempo is not None and original_tempo > 0:
            rate = float(target_tempo) / float(original_tempo)
            rate = max(0.5, min(2.0, rate))
            
        n_steps = 0
        if target_key is not None:
            n_steps = target_key - original_key
            if n_steps > 6: n_steps -= 12
            elif n_steps < -6: n_steps += 12

        if abs(rate - 1.0) > 0.02 or n_steps != 0:
            if PEDALBOARD_AVAILABLE:
                y = time_stretch(y, sr, stretch_factor=rate, pitch_shift_in_semitones=float(n_steps))
            else:
                if abs(rate - 1.0) > 0.02:
                    y = librosa.effects.time_stretch(y, rate=rate)
                if n_steps != 0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                
        # 3. LOUDNESS / GAIN
        if target_loudness is not None:
            gain_db = target_loudness - original_loudness
            # Clip gain to avoid blowing out speakers
            gain_db = max(-12, min(12, gain_db))
            if abs(gain_db) > 0.5:
                if PEDALBOARD_AVAILABLE:
                    y_pedal = y.reshape(1, -1) if len(y.shape) == 1 else y
                    board = Pedalboard([Gain(gain_db=gain_db)])
                    y_pedal = board(y_pedal, sr)
                    y = y_pedal.flatten() if len(y.shape) == 1 else y_pedal
                else:
                    gain_linear = 10 ** (gain_db / 20)
                    y = y * gain_linear
                
                # Hard limit to avoid clipping
                y = np.clip(y, -1.0, 1.0)
                
        # Ensure proper shape for soundfile: (samples, channels)
        if len(y.shape) > 1:
            if y.shape[0] < y.shape[1]:
                y = y.T # Convert (channels, samples) to (samples, channels)
        else:
            y = y.flatten()

        # Write to temporary file
        out_path = os.path.join(tempfile.gettempdir(), f'mutated_{analysis_id}.wav')
        sf.write(out_path, y, sr)
        
        # Send file back
        return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name='mutated_hit.wav')
        
    except Exception as e:
        logger.error(f"Error mutating audio: {e}")
        import traceback
        traceback.print_exc()
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


@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')

        if not all([username, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400

        df = get_users_df()
        
        # Check if email already exists
        if email in df['email'].values:
            return jsonify({'error': 'Email already registered'}), 400

        # Check if username exists
        if username in df['username'].values:
            return jsonify({'error': 'Username already taken'}), 400

        new_user = {
            'id': str(uuid.uuid4()),
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'created_at': datetime.now().isoformat(),
            'auth_provider': 'local'
        }
        
        df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
        save_users_df(df)

        return jsonify({
            'status': 'success',
            'user': {
                'userId': new_user['id'],
                'username': new_user['username'],
                'email': new_user['email'],
                'name': new_user['username']
            }
        })

    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        identifier = data.get('username')  # Can be email or username
        password = data.get('password')

        if not identifier or not password:
            return jsonify({'error': 'Missing credentials'}), 400

        df = get_users_df()
        
        # Find user by email or username
        user_row = df[(df['email'] == identifier) | (df['username'] == identifier)]
        
        if user_row.empty:
            return jsonify({'error': 'Invalid credentials'}), 401
            
        user = user_row.iloc[0]
        
        if user['auth_provider'] == 'google' and pd.isna(user['password']):
             return jsonify({'error': 'Please login with Google'}), 401

        if not check_password_hash(str(user['password']), password):
            return jsonify({'error': 'Invalid credentials'}), 401

        return jsonify({
            'status': 'success',
            'user': {
                'userId': str(user['id']),
                'username': str(user['username']),
                'email': str(user['email']),
                'name': str(user['username'])
            }
        })

    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/google-login', methods=['POST'])
def google_login():
    try:
        data = request.get_json()
        email = data.get('email')
        name = data.get('name')

        if not email:
            return jsonify({'error': 'Email is required from Google Auth'}), 400

        df = get_users_df()
        
        user_row = df[df['email'] == email]
        
        if user_row.empty:
            # Create new user
            username = email.split('@')[0]
            # Handle username collision
            base_username = username
            counter = 1
            while username in df['username'].values:
                username = f"{base_username}{counter}"
                counter += 1
                
            new_user = {
                'id': str(uuid.uuid4()),
                'username': username,
                'email': email,
                'password': '', # No password for google auth
                'created_at': datetime.now().isoformat(),
                'auth_provider': 'google'
            }
            df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
            save_users_df(df)
            user_data = new_user
        else:
            user_data = user_row.iloc[0].to_dict()

        return jsonify({
            'status': 'success',
            'user': {
                'userId': str(user_data['id']),
                'username': str(user_data['username']),
                'email': str(user_data['email']),
                'name': str(name or user_data['username'])
            }
        })

    except Exception as e:
        logger.error(f"Google login error: {e}")
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
        
        # Use combined dataset from data pipeline
        combined_data_path = BACKEND_DIR / 'data' / 'combined_dataset.csv'
        
        # Fallback to individual datasets if combined doesn't exist
        if combined_data_path.exists():
            data_path = combined_data_path
            logger.info("[INFO] Using combined dataset from unified pipeline")
        else:
            # Try primary dataset: spotify_tracks.csv
            data_path = DATA_DIR / 'spotify_tracks.csv'
            
            # Fallback to alternative names if primary doesn't exist
            if not data_path.exists():
                for name in ['dataset.csv', 'spotify_songs.csv']:
                    alt_path = DATA_DIR / name
                    if alt_path.exists():
                        data_path = alt_path
                        break
        
        if not data_path.exists():
            logger.error(f"[ERROR] Data file not found. Looked in: {DATA_DIR}")
            logger.error("Please run data pipeline first: python backend/data_pipeline.py")
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
        port = int(os.environ.get('PORT', os.environ.get('FLASK_PORT', 5000)))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    
    except Exception as e:
        logger.error(f"FATAL ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
