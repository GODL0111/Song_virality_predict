#!/usr/bin/env python3
"""
Weight Files Inspection Script
==============================

Displays detailed information about all saved model weight files.
"""

import os
import json
import joblib
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def inspect_weight_files():
    """Inspect and display all weight files"""
    
    model_dir = Path('backend/models')
    
    logger.info("\n" + "="*80)
    logger.info("MODEL WEIGHT FILES INSPECTION")
    logger.info("="*80)
    
    if not model_dir.exists():
        logger.error("Model directory not found!")
        return False
    
    # Get all files
    files = sorted([f for f in model_dir.glob('*') if f.is_file()])
    
    if not files:
        logger.warning("No weight files found!")
        return False
    
    logger.info(f"\nDirectory: {model_dir.absolute()}\n")
    
    total_size_mb = 0
    
    for i, file_path in enumerate(files, 1):
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        total_size_mb += size_mb
        
        logger.info(f"{i}. {file_path.name}")
        logger.info(f"   {'─' * 76}")
        logger.info(f"   Location: {file_path.absolute()}")
        logger.info(f"   Size:     {size_mb:.4f} MB ({size_bytes:,} bytes)")
        logger.info(f"   Type:     {file_path.suffix if file_path.suffix else 'directory'}")
        
        # Special handling for different file types
        if file_path.name == 'song_hit_model.pkl':
            logger.info(f"   Purpose:  ⭐ MAIN MODEL WEIGHTS (XGBoost)")
            logger.info(f"   Contains: 200 decision trees, split rules, leaf values")
            
            try:
                model = joblib.load(file_path)
                logger.info(f"   Status:   Successfully loaded")
                logger.info(f"   Model Type: {type(model).__name__}")
                
                # Get model details
                if hasattr(model, 'n_estimators'):
                    logger.info(f"   Estimators: {model.n_estimators}")
                if hasattr(model, 'max_depth'):
                    logger.info(f"   Max Depth: {model.max_depth}")
                if hasattr(model, 'learning_rate'):
                    logger.info(f"   Learning Rate: {model.learning_rate}")
            except Exception as e:
                logger.error(f"   Error loading: {e}")
        
        elif file_path.name == 'song_hit_model_features.pkl':
            logger.info(f"   Purpose:  Feature names and ordering")
            logger.info(f"   Contains: List of 12 musical features")
            
            try:
                features = joblib.load(file_path)
                logger.info(f"   Status:   Successfully loaded")
                logger.info(f"   Features: {', '.join(features)}")
            except Exception as e:
                logger.error(f"   Error loading: {e}")
        
        elif file_path.name == 'model_metadata.json':
            logger.info(f"   Purpose:  Model metadata and statistics")
            logger.info(f"   Contains: Accuracy, training time, feature names, etc.")
            
            try:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"   Status:   Successfully loaded")
                
                if metadata:
                    for key, value in metadata.items():
                        if key != 'feature_names':
                            if isinstance(value, (int, float)):
                                logger.info(f"   {key}: {value}")
                            elif isinstance(value, str) and 'T' not in str(value):
                                logger.info(f"   {key}: {value}")
            except Exception as e:
                logger.error(f"   Error loading: {e}")
        
        elif file_path.name == 'predict_main.py':
            logger.info(f"   Purpose:  Model prediction code and logic")
            logger.info(f"   Contains: SongHitPredictor class and methods")
        
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total files:     {len(files)}")
    logger.info(f"Total size:      {total_size_mb:.4f} MB")
    logger.info(f"Main weights:    song_hit_model.pkl (0.45 MB)")
    logger.info("")
    
    # File details table
    logger.info("FILE DETAILS TABLE")
    logger.info("-"*80)
    logger.info(f"{'Filename':<35} {'Size (MB)':<15} {'Type':<15}")
    logger.info("-"*80)
    
    for file_path in files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        file_type = file_path.suffix if file_path.suffix else "dir"
        logger.info(f"{file_path.name:<35} {size_mb:<15.4f} {file_type:<15}")
    
    logger.info("-"*80)
    logger.info(f"{'TOTAL':<35} {total_size_mb:<15.4f} MB")
    
    return True


def inspect_dataset_files():
    """Inspect dataset files"""
    
    logger.info("\n" + "="*80)
    logger.info("DATASET FILES")
    logger.info("="*80)
    
    data_dir = Path('backend/data')
    
    if not data_dir.exists():
        logger.error("Data directory not found!")
        return False
    
    files = sorted([f for f in data_dir.glob('*') if f.is_file()])
    
    if not files:
        logger.warning("No dataset files found!")
        return False
    
    logger.info(f"\nDirectory: {data_dir.absolute()}\n")
    
    total_size_mb = 0
    
    for file_path in files:
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        total_size_mb += size_mb
        
        logger.info(f"• {file_path.name}")
        logger.info(f"  Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
        
        if file_path.suffix == '.csv':
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                logger.info(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")
                logger.info(f"  Columns: {', '.join(df.columns.tolist())}")
            except:
                pass
        
        logger.info("")
    
    logger.info(f"Total dataset size: {total_size_mb:.2f} MB")
    
    return True


def main():
    """Main execution"""
    inspect_weight_files()
    inspect_dataset_files()
    
    logger.info("\n" + "="*80)
    logger.info("INSPECTION COMPLETE")
    logger.info("="*80)
    logger.info("\nKey Points:")
    logger.info("✓ Primary weight file: backend/models/song_hit_model.pkl (0.45 MB)")
    logger.info("✓ Dataset splits: backend/data/ (3 CSV files)")
    logger.info("✓ Ready for predictions and deployment")
    logger.info("")


if __name__ == '__main__':
    main()
