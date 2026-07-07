#!/usr/bin/env python3
"""
Script to train LSTM model for song hit prediction
Run this after the initial XGBoost training to create an LSTM alternative

Usage:
    python train_lstm_model.py
"""

import os
import sys
from pathlib import Path
import logging

# Add models to path
sys.path.insert(0, str(Path(__file__).parent / 'models'))
from predict_main import SongHitPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Train LSTM model"""
    backend_dir = Path(__file__).parent
    models_dir = backend_dir / 'models'
    data_dir = backend_dir.parent / 'datasets'
    
    logger.info("="*60)
    logger.info("LSTM Model Training Script")
    logger.info("="*60)
    
    # Find data file
    data_path = data_dir / 'spotify_tracks.csv'
    if not data_path.exists():
        for name in ['demo.csv', 'spotify_songs.csv']:
            alt_path = data_dir / name
            if alt_path.exists():
                data_path = alt_path
                break
    
    if not data_path.exists():
        logger.error(f"✗ Data file not found in {data_dir}")
        logger.error("Expected: spotify_tracks.csv, demo.csv, or spotify_songs.csv")
        return False
    
    logger.info(f"✓ Data file: {data_path}")
    
    # Initialize LSTM predictor
    logger.info("\nInitializing LSTM predictor...")
    predictor = SongHitPredictor(
        model_dir=str(models_dir),
        data_dir=str(data_dir),
        model_type="lstm"
    )
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    df, X, Y = predictor.load_and_prepare_data(str(data_path))
    
    if df is None:
        logger.error("✗ Failed to load data")
        return False
    
    logger.info(f"✓ Loaded {len(df)} samples")
    logger.info(f"  - Features: {list(X.columns)}")
    logger.info(f"  - Target distribution: {(Y.value_counts().to_dict())}")
    
    # Train LSTM model
    logger.info("\nTraining LSTM model...")
    logger.info("(This may take a few minutes...)")
    
    try:
        predictor.train_model(X, Y, force_retrain=True)
        logger.info("✓ LSTM model training completed")
    except Exception as e:
        logger.error(f"✗ Error training LSTM model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display metadata
    logger.info("\n" + "="*60)
    logger.info("LSTM Model Information")
    logger.info("="*60)
    if predictor.model_metadata:
        for key, value in predictor.model_metadata.items():
            if key != 'feature_names':
                logger.info(f"{key}: {value}")
    
    logger.info("\n✓ LSTM model saved successfully!")
    logger.info("You can now switch to LSTM model via POST /api/switch-model")
    logger.info("="*60)
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
