#!/usr/bin/env python3
"""
Unified Data Pipeline for Song Virality Prediction
===================================================

This module combines multiple Spotify datasets into a unified format
for training the virality prediction model.

Supported datasets:
- dataset.csv (Kaggle Spotify dataset)
- spotify_songs.csv (Spotify songs with playlist info)
- spotify_tracks.csv (Spotify tracks with audio features)
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional, List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Unified data pipeline that combines multiple Spotify datasets
    into a single training dataset.
    """
    
    # Common audio features across all datasets
    AUDIO_FEATURES = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms'
    ]
    
    # Column mapping for each dataset type
    COLUMN_MAPPINGS = {
        'dataset.csv': {
            'track_id': 'track_id',
            'track_name': 'track_name',
            'artists': 'artist_name',
            'album_name': 'album_name',
            'popularity': 'popularity',
            'track_genre': 'genre',
            # Audio features have same names
        },
        'spotify_songs.csv': {
            'track_id': 'track_id',
            'track_name': 'track_name',
            'track_artist': 'artist_name',
            'track_album_name': 'album_name',
            'track_popularity': 'popularity',
            'playlist_genre': 'genre',
            # Audio features have same names
        },
        'spotify_tracks.csv': {
            'track_id': 'track_id',
            'track_name': 'track_name',
            'artist_name': 'artist_name',
            'album_name': 'album_name',
            'popularity': 'popularity',
            'language': 'language',
            # Audio features have same names
        }
    }
    
    def __init__(self, datasets_dir: str = 'datasets', output_dir: str = 'backend/data'):
        """
        Initialize the data pipeline.
        
        Args:
            datasets_dir: Directory containing the raw datasets
            output_dir: Directory to save processed data
        """
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.combined_df: Optional[pd.DataFrame] = None
        self.stats: Dict = {}
        
    def discover_datasets(self) -> List[str]:
        """
        Discover available datasets in the datasets directory.
        
        Returns:
            List of dataset filenames found
        """
        available = []
        for dataset_name in self.COLUMN_MAPPINGS.keys():
            path = self.datasets_dir / dataset_name
            if path.exists():
                available.append(dataset_name)
                logger.info(f"Found dataset: {dataset_name}")
            else:
                logger.warning(f"Dataset not found: {dataset_name}")
        return available
    
    def load_dataset(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load a single dataset and standardize column names.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            Standardized DataFrame or None if loading fails
        """
        path = self.datasets_dir / filename
        
        if not path.exists():
            logger.error(f"Dataset not found: {path}")
            return None
        
        try:
            logger.info(f"Loading {filename}...")
            df = pd.read_csv(path, on_bad_lines='skip', engine='python')
            logger.info(f"  Loaded {len(df):,} records with {len(df.columns)} columns")
            
            # Get column mapping for this dataset
            mapping = self.COLUMN_MAPPINGS.get(filename, {})
            
            # Standardize column names
            standardized = pd.DataFrame()
            
            # Map metadata columns
            for src_col, dst_col in mapping.items():
                if src_col in df.columns:
                    standardized[dst_col] = df[src_col]
            
            # Copy audio features directly (they have same names)
            for feature in self.AUDIO_FEATURES:
                if feature in df.columns:
                    standardized[feature] = df[feature]
            
            # Add source column to track origin
            standardized['source'] = filename
            
            logger.info(f"  Standardized to {len(standardized.columns)} columns")
            return standardized
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None
    
    def load_all_datasets(self) -> bool:
        """
        Load all available datasets.
        
        Returns:
            True if at least one dataset was loaded successfully
        """
        available = self.discover_datasets()
        
        if not available:
            logger.error("No datasets found!")
            return False
        
        for filename in available:
            df = self.load_dataset(filename)
            if df is not None:
                self.datasets[filename] = df
                self.stats[f'{filename}_records'] = len(df)
        
        logger.info(f"\nLoaded {len(self.datasets)} datasets successfully")
        return len(self.datasets) > 0
    
    def combine_datasets(self, deduplicate: bool = True) -> bool:
        """
        Combine all loaded datasets into a single DataFrame.
        
        Args:
            deduplicate: Whether to remove duplicate tracks by track_id
            
        Returns:
            True if combination was successful
        """
        if not self.datasets:
            logger.error("No datasets loaded! Call load_all_datasets() first.")
            return False
        
        logger.info("\nCombining datasets...")
        
        # Concatenate all datasets
        dfs = list(self.datasets.values())
        self.combined_df = pd.concat(dfs, ignore_index=True)
        
        initial_count = len(self.combined_df)
        logger.info(f"  Combined total: {initial_count:,} records")
        self.stats['combined_total'] = initial_count
        
        # Log source distribution
        source_counts = self.combined_df['source'].value_counts()
        logger.info("  Records by source:")
        for source, count in source_counts.items():
            logger.info(f"    - {source}: {count:,}")
        
        # Deduplicate by track_id if requested
        if deduplicate and 'track_id' in self.combined_df.columns:
            # Keep first occurrence (preserves order)
            before = len(self.combined_df)
            self.combined_df = self.combined_df.drop_duplicates(subset=['track_id'], keep='first')
            removed = before - len(self.combined_df)
            logger.info(f"  Removed {removed:,} duplicate tracks")
            self.stats['duplicates_removed'] = removed
        
        self.stats['unique_tracks'] = len(self.combined_df)
        logger.info(f"  Final unique tracks: {len(self.combined_df):,}")
        
        return True
    
    def clean_data(self, hit_threshold: int = 50) -> bool:
        """
        Clean the combined dataset and create target variable.
        
        Args:
            hit_threshold: Popularity score threshold for hit classification
            
        Returns:
            True if cleaning was successful
        """
        if self.combined_df is None:
            logger.error("No combined data! Call combine_datasets() first.")
            return False
        
        logger.info("\nCleaning combined dataset...")
        
        # Ensure popularity column exists
        if 'popularity' not in self.combined_df.columns:
            logger.error("No 'popularity' column found!")
            return False
        
        # Convert popularity to numeric
        self.combined_df['popularity'] = pd.to_numeric(
            self.combined_df['popularity'], errors='coerce'
        )
        
        # Create binary target variable
        self.combined_df['is_hit'] = (
            self.combined_df['popularity'] >= hit_threshold
        ).astype(int)
        
        # Convert audio features to numeric
        for feature in self.AUDIO_FEATURES:
            if feature in self.combined_df.columns:
                self.combined_df[feature] = pd.to_numeric(
                    self.combined_df[feature], errors='coerce'
                )
        
        # Check for missing features
        missing_features = [
            f for f in self.AUDIO_FEATURES 
            if f not in self.combined_df.columns
        ]
        if missing_features:
            logger.warning(f"Missing audio features: {missing_features}")
        
        # Drop rows with NaN in required columns
        required_cols = [f for f in self.AUDIO_FEATURES if f in self.combined_df.columns]
        required_cols.append('is_hit')
        
        before = len(self.combined_df)
        self.combined_df = self.combined_df.dropna(subset=required_cols)
        removed = before - len(self.combined_df)
        
        logger.info(f"  Removed {removed:,} rows with missing values")
        logger.info(f"  Clean records: {len(self.combined_df):,}")
        
        self.stats['after_cleaning'] = len(self.combined_df)
        self.stats['rows_with_missing'] = removed
        
        # Log class distribution
        class_dist = self.combined_df['is_hit'].value_counts().sort_index()
        non_hits = class_dist.get(0, 0)
        hits = class_dist.get(1, 0)
        total = len(self.combined_df)
        
        logger.info(f"\nClass Distribution (threshold={hit_threshold}):")
        logger.info(f"  - Non-hits (0): {non_hits:,} ({non_hits/total*100:.1f}%)")
        logger.info(f"  - Hits (1):     {hits:,} ({hits/total*100:.1f}%)")
        
        self.stats['non_hits'] = int(non_hits)
        self.stats['hits'] = int(hits)
        self.stats['hit_threshold'] = hit_threshold
        
        return True
    
    def create_splits(
        self, 
        train_ratio: float = 0.7, 
        test_ratio: float = 0.15, 
        val_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Create train/test/validation splits with stratification.
        
        Args:
            train_ratio: Proportion for training set
            test_ratio: Proportion for test set
            val_ratio: Proportion for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, X_val, y_train, y_test, y_val)
        """
        if self.combined_df is None:
            raise ValueError("No data available! Run the pipeline first.")
        
        logger.info("\nCreating train/test/validation splits...")
        
        # Get available features
        available_features = [
            f for f in self.AUDIO_FEATURES 
            if f in self.combined_df.columns
        ]
        
        X = self.combined_df[available_features]
        y = self.combined_df['is_hit']
        
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
        total = len(X)
        logger.info(f"\nDataset Splits:")
        logger.info(f"  Train: {len(X_train):>7,} samples ({len(X_train)/total*100:5.1f}%)")
        logger.info(f"  Test:  {len(X_test):>7,} samples ({len(X_test)/total*100:5.1f}%)")
        logger.info(f"  Valid: {len(X_val):>7,} samples ({len(X_val)/total*100:5.1f}%)")
        
        # Log class distribution per split
        logger.info(f"\nClass Distribution per Split:")
        for name, X_split, y_split in [
            ('Train', X_train, y_train), 
            ('Test', X_test, y_test), 
            ('Valid', X_val, y_val)
        ]:
            hits = (y_split == 1).sum()
            non_hits = (y_split == 0).sum()
            logger.info(f"  {name}:")
            logger.info(f"    - Hits:     {hits:>6,} ({hits/len(y_split)*100:5.1f}%)")
            logger.info(f"    - Non-hits: {non_hits:>6,} ({non_hits/len(y_split)*100:5.1f}%)")
        
        self.stats['train_samples'] = len(X_train)
        self.stats['test_samples'] = len(X_test)
        self.stats['val_samples'] = len(X_val)
        self.stats['features_used'] = available_features
        
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def save_splits(
        self,
        X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series, y_val: pd.Series
    ) -> bool:
        """
        Save the data splits to CSV files.
        
        Args:
            X_train, X_test, X_val: Feature DataFrames
            y_train, y_test, y_val: Target Series
            
        Returns:
            True if saving was successful
        """
        logger.info("\nSaving splits to CSV files...")
        
        try:
            # Save train dataset
            train_df = X_train.copy()
            train_df['is_hit'] = y_train.values
            train_path = self.output_dir / 'train_dataset.csv'
            train_df.to_csv(train_path, index=False)
            logger.info(f"  Train: {train_path}")
            
            # Save test dataset
            test_df = X_test.copy()
            test_df['is_hit'] = y_test.values
            test_path = self.output_dir / 'test_dataset.csv'
            test_df.to_csv(test_path, index=False)
            logger.info(f"  Test:  {test_path}")
            
            # Save validation dataset
            val_df = X_val.copy()
            val_df['is_hit'] = y_val.values
            val_path = self.output_dir / 'validation_dataset.csv'
            val_df.to_csv(val_path, index=False)
            logger.info(f"  Valid: {val_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving splits: {e}")
            return False
    
    def save_combined_dataset(self, filename: str = 'combined_dataset.csv') -> bool:
        """
        Save the combined dataset before splitting.
        
        Args:
            filename: Name of the output file
            
        Returns:
            True if saving was successful
        """
        if self.combined_df is None:
            logger.error("No combined data to save!")
            return False
        
        try:
            path = self.output_dir / filename
            self.combined_df.to_csv(path, index=False)
            logger.info(f"Combined dataset saved: {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving combined dataset: {e}")
            return False
    
    def save_stats(self) -> bool:
        """Save pipeline statistics to JSON."""
        stats_path = self.output_dir / 'pipeline_stats.json'
        
        try:
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            logger.info(f"Statistics saved: {stats_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
            return False
    
    def run(
        self,
        deduplicate: bool = True,
        hit_threshold: int = 50,
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        save_combined: bool = False
    ) -> bool:
        """
        Run the complete data pipeline.
        
        Args:
            deduplicate: Remove duplicate tracks
            hit_threshold: Popularity threshold for hit classification
            train_ratio: Training set proportion
            test_ratio: Test set proportion
            val_ratio: Validation set proportion
            save_combined: Whether to save the combined dataset
            
        Returns:
            True if pipeline completed successfully
        """
        logger.info("=" * 60)
        logger.info("UNIFIED DATA PIPELINE FOR SONG VIRALITY PREDICTION")
        logger.info("=" * 60)
        
        # Step 1: Load all datasets
        if not self.load_all_datasets():
            return False
        
        # Step 2: Combine datasets
        if not self.combine_datasets(deduplicate=deduplicate):
            return False
        
        # Step 3: Clean data
        if not self.clean_data(hit_threshold=hit_threshold):
            return False
        
        # Step 4: Optionally save combined dataset
        if save_combined:
            self.save_combined_dataset()
        
        # Step 5: Create splits
        X_train, X_test, X_val, y_train, y_test, y_val = self.create_splits(
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio
        )
        
        # Step 6: Save splits
        if not self.save_splits(X_train, X_test, X_val, y_train, y_test, y_val):
            return False
        
        # Step 7: Save statistics
        self.save_stats()
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"\nOutput files saved to: {self.output_dir}")
        logger.info(f"  - train_dataset.csv ({self.stats['train_samples']:,} samples)")
        logger.info(f"  - test_dataset.csv ({self.stats['test_samples']:,} samples)")
        logger.info(f"  - validation_dataset.csv ({self.stats['val_samples']:,} samples)")
        logger.info(f"  - pipeline_stats.json")
        
        return True


def main():
    """Main entry point for the data pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified Data Pipeline for Song Virality Prediction'
    )
    parser.add_argument(
        '--datasets-dir', 
        default='datasets',
        help='Directory containing raw datasets (default: datasets)'
    )
    parser.add_argument(
        '--output-dir',
        default='backend/data',
        help='Output directory for processed data (default: backend/data)'
    )
    parser.add_argument(
        '--hit-threshold',
        type=int,
        default=50,
        help='Popularity threshold for hit classification (default: 50)'
    )
    parser.add_argument(
        '--no-deduplicate',
        action='store_true',
        help='Do not remove duplicate tracks'
    )
    parser.add_argument(
        '--save-combined',
        action='store_true',
        help='Save combined dataset before splitting'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = DataPipeline(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir
    )
    
    success = pipeline.run(
        deduplicate=not args.no_deduplicate,
        hit_threshold=args.hit_threshold,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        save_combined=args.save_combined
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
