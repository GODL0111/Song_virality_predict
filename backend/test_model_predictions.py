#!/usr/bin/env python3
"""
Test script to compare XGBoost and LSTM model predictions
Demonstrates the bias fixes and model comparison

Usage:
    python test_model_predictions.py
"""

import os
import sys
from pathlib import Path
import logging
import json

# Add models to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
from predict_main import SongHitPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_predictions():
    """Test both models with sample songs"""
    
    backend_dir = Path(__file__).parent
    models_dir = backend_dir / 'models'
    data_dir = backend_dir.parent / 'datasets'
    
    logger.info("="*70)
    logger.info("MODEL COMPARISON TEST - XGBoost vs LSTM")
    logger.info("="*70)
    
    # Sample songs with various feature values
    test_songs = [
        {
            'name': 'Low Energy Song',
            'features': {
                'danceability': 0.3, 'energy': 0.2, 'key': 5, 'loudness': -15,
                'mode': 0, 'speechiness': 0.05, 'acousticness': 0.8,
                'instrumentalness': 0.1, 'liveness': 0.1, 'valence': 0.3,
                'tempo': 80, 'duration_ms': 180000
            }
        },
        {
            'name': 'High Energy Dance Song',
            'features': {
                'danceability': 0.85, 'energy': 0.9, 'key': 7, 'loudness': -3,
                'mode': 1, 'speechiness': 0.08, 'acousticness': 0.05,
                'instrumentalness': 0.02, 'liveness': 0.25, 'valence': 0.8,
                'tempo': 140, 'duration_ms': 210000
            }
        },
        {
            'name': 'Acoustic Ballad',
            'features': {
                'danceability': 0.4, 'energy': 0.4, 'key': 0, 'loudness': -12,
                'mode': 0, 'speechiness': 0.15, 'acousticness': 0.9,
                'instrumentalness': 0.05, 'liveness': 0.2, 'valence': 0.5,
                'tempo': 90, 'duration_ms': 240000
            }
        },
        {
            'name': 'Pop Hit Potential',
            'features': {
                'danceability': 0.7, 'energy': 0.75, 'key': 2, 'loudness': -5,
                'mode': 1, 'speechiness': 0.1, 'acousticness': 0.2,
                'instrumentalness': 0.01, 'liveness': 0.15, 'valence': 0.7,
                'tempo': 120, 'duration_ms': 200000
            }
        },
    ]
    
    # Test both models
    for model_type in ['xgboost', 'lstm']:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing {model_type.upper()} Model")
        logger.info(f"{'='*70}")
        
        try:
            # Initialize predictor
            predictor = SongHitPredictor(
                model_dir=str(models_dir),
                data_dir=str(data_dir),
                model_type=model_type
            )
            
            # Try to load model
            if not predictor.load_model(model_type=model_type):
                logger.warning(f"⚠ {model_type.upper()} model not found. Skipping.")
                logger.info(f"  Hint: Train first with 'python train_lstm_model.py'")
                continue
            
            logger.info(f"✓ {model_type.upper()} model loaded")
            if predictor.model_metadata:
                logger.info(f"  Accuracy: {predictor.model_metadata.get('accuracy', 'N/A'):.2%}")
                logger.info(f"  Training Time: {predictor.model_metadata.get('training_time', 'N/A'):.2f}s")
            
            # Make predictions
            logger.info(f"\nPredictions:")
            logger.info(f"{'-'*70}")
            
            for song in test_songs:
                result = predictor.predict_song_hit_probability(song['features'])
                
                if result:
                    prob = result['hit_probability']
                    confidence = result['confidence']
                    prediction = result['is_hit_prediction']
                    
                    # Visual representation
                    bar_length = int(prob * 40)
                    bar = '█' * bar_length + '░' * (40 - bar_length)
                    
                    logger.info(f"\n{song['name']}")
                    logger.info(f"  Probability: |{bar}| {prob:.1%}")
                    logger.info(f"  Prediction:  {('HIT' if prediction else 'MISS')} (confidence: {confidence:.1%})")
                else:
                    logger.error(f"  Failed to predict for {song['name']}")
            
        except Exception as e:
            logger.error(f"Error testing {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'='*70}")
    logger.info("TEST COMPLETE")
    logger.info("="*70)
    logger.info("\nKey Observations:")
    logger.info("1. LSTM and XGBoost may give different predictions")
    logger.info("2. Both models apply bias correction to reduce false negatives")
    logger.info("3. High-energy, danceable songs should score higher")
    logger.info("4. Confidence increases with clearer predictions")
    logger.info("\nTo use LSTM model in API:")
    logger.info("  POST /api/switch-model")
    logger.info("  {\"model_type\": \"lstm\"}")

if __name__ == '__main__':
    test_predictions()
