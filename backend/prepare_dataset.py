#!/usr/bin/env python3
"""
Dataset Preparation Script for Song Virality Prediction
========================================================

This script prepares the spotify_tracks.csv dataset for:
1. Training
2. Testing
3. Validation

Creates train/test/validation splits with proper data cleaning and preprocessing.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetPreparation:
    def __init__(self, dataset_path='datasets/spotify_tracks.csv', output_dir='backend/data'):
        """Initialize dataset preparation"""
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define features for the model
        self.musical_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms'
        ]
        
        self.df = None
        self.stats = {}
        
    def load_dataset(self):
        """Load the primary dataset"""
        logger.info(f"Loading dataset from: {self.dataset_path}")
        
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset not found at {self.dataset_path}")
            return False
        
        try:
            self.df = pd.read_csv(self.dataset_path, on_bad_lines='skip', engine='python')
            logger.info(f"Dataset loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns")
            self.stats['initial_records'] = len(self.df)
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def clean_dataset(self):
        """Clean and prepare the dataset"""
        logger.info("Cleaning dataset...")
        
        if self.df is None:
            return False
        
        # Handle popularity column (use 'popularity' if 'track_popularity' doesn't exist)
        if 'popularity' not in self.df.columns and 'track_popularity' not in self.df.columns:
            logger.error("Neither 'popularity' nor 'track_popularity' column found!")
            return False
        
        popularity_col = 'track_popularity' if 'track_popularity' in self.df.columns else 'popularity'
        
        # Define target: hit if popularity >= 50
        self.df['is_hit'] = (self.df[popularity_col] >= 50).astype(int)
        
        # Ensure all musical features exist
        missing_features = [f for f in self.musical_features if f not in self.df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
        
        # Convert features to numeric
        for feature in self.musical_features:
            self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')
        
        # Remove rows with NaN in required columns
        required_cols = self.musical_features + ['is_hit']
        self.df = self.df.dropna(subset=required_cols)
        
        logger.info(f"After cleaning: {len(self.df)} records")
        self.stats['after_cleaning'] = len(self.df)
        
        # Log class distribution
        class_dist = self.df['is_hit'].value_counts()
        logger.info(f"Class distribution:")
        logger.info(f"  - Non-hits (0): {class_dist[0]} ({class_dist[0]/len(self.df)*100:.1f}%)")
        logger.info(f"  - Hits (1): {class_dist[1]} ({class_dist[1]/len(self.df)*100:.1f}%)")
        self.stats['non_hits'] = int(class_dist[0])
        self.stats['hits'] = int(class_dist[1])
        
        return True
    
    def create_splits(self, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, random_state=42):
        """Create train/test/validation splits"""
        logger.info("Creating train/test/validation splits...")
        
        if self.df is None:
            logger.error("Dataset not loaded!")
            return False
        
        # Extract features and target
        X = self.df[self.musical_features]
        y = self.df['is_hit']
        
        # First split: train + temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(test_ratio + val_ratio),
            random_state=random_state,
            stratify=y
        )
        
        # Second split: test and validation
        test_split_ratio = test_ratio / (test_ratio + val_ratio)
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp,
            test_size=(1 - test_split_ratio),
            random_state=random_state,
            stratify=y_temp
        )
        
        # Log split information
        logger.info(f"\nDataset Splits:")
        logger.info(f"  Train: {len(X_train):6d} samples ({len(X_train)/len(X)*100:5.1f}%)")
        logger.info(f"  Test:  {len(X_test):6d} samples ({len(X_test)/len(X)*100:5.1f}%)")
        logger.info(f"  Valid: {len(X_val):6d} samples ({len(X_val)/len(X)*100:5.1f}%)")
        
        # Log class distribution per split
        logger.info(f"\nClass Distribution per Split:")
        for name, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test), ('Valid', X_val, y_val)]:
            hits = (y == 1).sum()
            non_hits = (y == 0).sum()
            logger.info(f"  {name}:")
            logger.info(f"    - Hits:     {hits:5d} ({hits/len(y)*100:5.1f}%)")
            logger.info(f"    - Non-hits: {non_hits:5d} ({non_hits/len(y)*100:5.1f}%)")
        
        self.stats['train_samples'] = len(X_train)
        self.stats['test_samples'] = len(X_test)
        self.stats['val_samples'] = len(X_val)
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def save_splits(self, X_train, X_test, X_val, y_train, y_test, y_val):
        """Save the splits to CSV files"""
        logger.info("Saving splits to CSV files...")
        
        try:
            # Create train dataset
            train_df = X_train.copy()
            train_df['is_hit'] = y_train.values
            train_path = self.output_dir / 'train_dataset.csv'
            train_df.to_csv(train_path, index=False)
            logger.info(f"  Train: {train_path}")
            
            # Create test dataset
            test_df = X_test.copy()
            test_df['is_hit'] = y_test.values
            test_path = self.output_dir / 'test_dataset.csv'
            test_df.to_csv(test_path, index=False)
            logger.info(f"  Test:  {test_path}")
            
            # Create validation dataset
            val_df = X_val.copy()
            val_df['is_hit'] = y_val.values
            val_path = self.output_dir / 'validation_dataset.csv'
            val_df.to_csv(val_path, index=False)
            logger.info(f"  Valid: {val_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving splits: {e}")
            return False
    
    def save_stats(self):
        """Save dataset statistics"""
        stats_path = self.output_dir / 'dataset_stats.json'
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Statistics saved: {stats_path}")
        return True
    
    def create_summary_report(self):
        """Create a summary report"""
        report_path = self.output_dir / 'DATASET_SUMMARY.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATASET PREPARATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("PRIMARY DATASET\n")
            f.write("-"*70 + "\n")
            f.write(f"Source: {self.dataset_path}\n")
            f.write(f"Initial Records: {self.stats['initial_records']}\n")
            f.write(f"After Cleaning: {self.stats['after_cleaning']}\n\n")
            
            f.write("CLASS DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            f.write(f"Non-hits (popularity < 50): {self.stats['non_hits']}\n")
            f.write(f"Hits (popularity >= 50): {self.stats['hits']}\n\n")
            
            f.write("TRAIN/TEST/VALIDATION SPLITS\n")
            f.write("-"*70 + "\n")
            f.write(f"Train: {self.stats['train_samples']} samples (70%)\n")
            f.write(f"Test:  {self.stats['test_samples']} samples (15%)\n")
            f.write(f"Valid: {self.stats['val_samples']} samples (15%)\n\n")
            
            f.write("SAVED FILES\n")
            f.write("-"*70 + "\n")
            f.write(f"{self.output_dir}/train_dataset.csv\n")
            f.write(f"{self.output_dir}/test_dataset.csv\n")
            f.write(f"{self.output_dir}/validation_dataset.csv\n")
            f.write(f"{self.output_dir}/dataset_stats.json\n\n")
            
            f.write("MUSICAL FEATURES\n")
            f.write("-"*70 + "\n")
            for i, feature in enumerate(self.musical_features, 1):
                f.write(f"{i:2d}. {feature}\n")
        
        logger.info(f"Summary report saved: {report_path}")
        return True


def main():
    """Main execution"""
    logger.info("="*70)
    logger.info("SPOTIFY DATASET PREPARATION")
    logger.info("="*70 + "\n")
    
    # Initialize preparation
    prep = DatasetPreparation(
        dataset_path='datasets/spotify_tracks.csv',
        output_dir='backend/data'
    )
    
    # Load dataset
    if not prep.load_dataset():
        return False
    
    # Clean dataset
    if not prep.clean_dataset():
        return False
    
    # Create splits
    splits = prep.create_splits(train_ratio=0.7, test_ratio=0.15, val_ratio=0.15)
    if not splits:
        return False
    
    X_train, X_test, X_val, y_train, y_test, y_val = splits
    
    # Save splits
    if not prep.save_splits(X_train, X_test, X_val, y_train, y_test, y_val):
        return False
    
    # Save statistics
    if not prep.save_stats():
        return False
    
    # Create summary report
    if not prep.create_summary_report():
        return False
    
    logger.info("\n" + "="*70)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("="*70)
    
    return True


if __name__ == '__main__':
    # Use the new unified data pipeline instead
    from data_pipeline import DataPipeline
    
    print("=" * 60)
    print("NOTE: Using unified DataPipeline for multi-dataset support")
    print("=" * 60)
    
    pipeline = DataPipeline(
        datasets_dir='datasets',
        output_dir='backend/data'
    )
    
    success = pipeline.run(
        deduplicate=True,
        hit_threshold=50,
        save_combined=True
    )
    
    exit(0 if success else 1)
