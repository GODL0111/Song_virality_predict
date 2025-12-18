#!/usr/bin/env python3
"""
Debug script to test predictions and see what's happening
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'backend' / 'models'))

from predict_main import SongHitPredictor
import numpy as np

# Initialize predictor
predictor = SongHitPredictor(
    model_dir=str(Path(__file__).parent / 'backend' / 'models'),
    data_dir=str(Path(__file__).parent / 'datasets'),
    model_type="xgboost"
)

# Load model
if predictor.load_model():
    print("✓ Model loaded")
    print(f"Model metadata: {predictor.model_metadata}")
else:
    print("✗ Failed to load model")
    sys.exit(1)

# Test song with good features
test_song = {
    'duration_ms': 214482,
    'tempo': 95.703125,
    'energy': 0.5582396149635315,
    'loudness': -13.45650922741498,
    'danceability': 0.48821662253267006,
    'valence': 0.438522750076726,
    'speechiness': 0.3058045001792866,
    'acousticness': 0.796739005270769,
    'liveness': 0.010483223013579845,
    'instrumentalness': 0.8330345997848561,
    'key': 10,
    'mode': 0
}

print("\n" + "="*70)
print("TEST PREDICTION")
print("="*70)
print(f"\nInput features: {test_song}")

# Get raw prediction
import pandas as pd
song_df = pd.DataFrame([test_song])
song_df = song_df[predictor.feature_names]

raw_proba = predictor.model.predict_proba(song_df)[0]
print(f"\nRaw model probabilities: [non-hit: {raw_proba[0]:.6f}, hit: {raw_proba[1]:.6f}]")

# Test bias correction
raw_hit_prob = raw_proba[1]
print(f"Raw hit probability: {raw_hit_prob:.6f}")

# Manually test bias correction
if hasattr(predictor, 'Y_train') and predictor.Y_train is not None:
    hit_ratio = (predictor.Y_train == 1).sum() / len(predictor.Y_train)
    print(f"Training hit ratio: {hit_ratio:.6f}")
else:
    hit_ratio = 0.15
    print(f"Using default hit ratio: {hit_ratio:.6f}")

# Apply correction formula
numerator = raw_hit_prob * hit_ratio
denominator = numerator + (1 - raw_hit_prob) * (1 - hit_ratio)
corrected = numerator / denominator if denominator > 0 else raw_hit_prob
print(f"Corrected probability: {corrected:.6f}")

# Use predictor's method
result = predictor.predict_song_hit_probability(test_song)
print(f"\nPredictor result: {result}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
if result and result['hit_probability'] == 0.0:
    print("⚠ PROBLEM: Hit probability is 0.0!")
    print("  - Check if model is actually loaded")
    print("  - Check if bias correction is working")
    print("  - Check if features are being passed correctly")
elif result:
    print(f"✓ Hit probability: {result['hit_probability']:.2%}")
    print(f"✓ Prediction: {'HIT' if result['is_hit_prediction'] else 'MISS'}")
    print(f"✓ Confidence: {result['confidence']:.2%}")
