#!/usr/bin/env python3
"""
Model Training Script for Song Virality Prediction
===================================================

Trains XGBoost and LSTM models using the prepared train/test/validation splits.
Saves model weights and generates performance reports.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import joblib
from pathlib import Path
from datetime import datetime
import logging

# ML imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, data_dir='backend/data', model_dir='backend/models', output_dir='backend/training_reports'):
        """Initialize model trainer"""
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        
        self.results = {}
    
    def load_splits(self):
        """Load train/test/validation splits"""
        logger.info("Loading dataset splits...")
        
        train_path = self.data_dir / 'train_dataset.csv'
        test_path = self.data_dir / 'test_dataset.csv'
        val_path = self.data_dir / 'validation_dataset.csv'
        
        if not all([train_path.exists(), test_path.exists(), val_path.exists()]):
            logger.error("Dataset splits not found!")
            return False
        
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        val_df = pd.read_csv(val_path)
        
        # Separate features and labels
        self.y_train = train_df.pop('is_hit')
        self.y_test = test_df.pop('is_hit')
        self.y_val = val_df.pop('is_hit')
        
        self.X_train = train_df
        self.X_test = test_df
        self.X_val = val_df
        
        logger.info(f"Train: {len(self.X_train)} samples")
        logger.info(f"Test:  {len(self.X_test)} samples")
        logger.info(f"Valid: {len(self.X_val)} samples")
        
        return True
    
    def train_xgboost(self):
        """Train XGBoost model"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING XGBOOST MODEL")
        logger.info("="*70)
        
        # Calculate class weights
        n_samples = len(self.y_train)
        n_hits = (self.y_train == 1).sum()
        n_non_hits = (self.y_train == 0).sum()
        
        weight_hits = n_samples / (2 * n_hits)
        weight_non_hits = n_samples / (2 * n_non_hits)
        
        logger.info(f"Class weights: hits={weight_hits:.2f}, non_hits={weight_non_hits:.2f}")
        
        # Initialize model
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='auc',
            random_state=42,
            scale_pos_weight=weight_hits / weight_non_hits,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            base_score=0.4,
            n_jobs=-1,
            verbose=1
        )
        
        logger.info("Training XGBoost...")
        model.fit(self.X_train, self.y_train, verbose=False)
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        val_pred = model.predict(self.X_val)
        
        train_proba = model.predict_proba(self.X_train)[:, 1]
        test_proba = model.predict_proba(self.X_test)[:, 1]
        val_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        results = {
            'model_type': 'XGBoost',
            'training_timestamp': datetime.now().isoformat(),
            'train_metrics': self._calculate_metrics(self.y_train, train_pred, train_proba),
            'test_metrics': self._calculate_metrics(self.y_test, test_pred, test_proba),
            'val_metrics': self._calculate_metrics(self.y_val, val_pred, val_proba),
        }
        
        # Save model
        model_path = self.model_dir / 'song_hit_model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save features
        features_path = self.model_dir / 'song_hit_model_features.pkl'
        joblib.dump(self.X_train.columns.tolist(), features_path)
        
        self.results['xgboost'] = results
        
        # Log results
        self._log_results('XGBoost', results)
        
        return model, results
    
    def train_lstm(self):
        """Train LSTM model (optional)"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping LSTM training")
            return None, None
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING LSTM MODEL")
        logger.info("="*70)
        
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        X_val_scaled = scaler.transform(self.X_val)
        
        # Reshape for LSTM
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
        
        # Build model
        model = models.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Calculate class weights
        n_samples = len(self.y_train)
        n_hits = (self.y_train == 1).sum()
        n_non_hits = (self.y_train == 0).sum()
        class_weight = {0: n_samples / (2 * n_non_hits), 1: n_samples / (2 * n_hits)}
        
        logger.info(f"Class weights: {class_weight}")
        logger.info("Training LSTM...")
        
        # Train with early stopping
        history = model.fit(
            X_train_lstm, self.y_train,
            validation_data=(X_val_lstm, self.y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ],
            verbose=0
        )
        
        # Evaluate
        train_proba = model.predict(X_train_lstm, verbose=0).flatten()
        test_proba = model.predict(X_test_lstm, verbose=0).flatten()
        val_proba = model.predict(X_val_lstm, verbose=0).flatten()
        
        train_pred = (train_proba > 0.5).astype(int)
        test_pred = (test_proba > 0.5).astype(int)
        val_pred = (val_proba > 0.5).astype(int)
        
        results = {
            'model_type': 'LSTM',
            'training_timestamp': datetime.now().isoformat(),
            'train_metrics': self._calculate_metrics(self.y_train, train_pred, train_proba),
            'test_metrics': self._calculate_metrics(self.y_test, test_pred, test_proba),
            'val_metrics': self._calculate_metrics(self.y_val, val_pred, val_proba),
            'epochs_trained': len(history.history['loss'])
        }
        
        # Save model
        model_path = self.model_dir / 'song_hit_model_lstm.h5'
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / 'song_hit_model_lstm_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        
        self.results['lstm'] = results
        
        # Log results
        self._log_results('LSTM', results)
        
        return model, results
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate performance metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_true, y_proba)),
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
        
        # Calculate additional rates
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        
        return metrics
    
    def _log_results(self, model_name, results):
        """Log training results"""
        logger.info(f"\n{model_name} Results:")
        logger.info("-" * 70)
        
        for split_name in ['train_metrics', 'test_metrics', 'val_metrics']:
            metrics = results[split_name]
            split_label = split_name.replace('_metrics', '').upper()
            
            logger.info(f"\n{split_label}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            logger.info(f"  Specificity: {metrics['specificity']:.4f}")
            logger.info(f"  FPR: {metrics['false_positive_rate']:.4f}")
            logger.info(f"  FNR: {metrics['false_negative_rate']:.4f}")
    
    def save_training_report(self):
        """Save comprehensive training report"""
        report_path = self.output_dir / 'TRAINING_REPORT.json'
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Training report saved: {report_path}")
        
        # Also save text version
        text_report_path = self.output_dir / 'TRAINING_REPORT.txt'
        with open(text_report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL TRAINING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("DATASET INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Train samples: {len(self.X_train)}\n")
            f.write(f"Test samples:  {len(self.X_test)}\n")
            f.write(f"Valid samples: {len(self.X_val)}\n")
            f.write(f"Total samples: {len(self.X_train) + len(self.X_test) + len(self.X_val)}\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{'='*70}\n")
                f.write(f"{model_name.upper()} MODEL\n")
                f.write(f"{'='*70}\n")
                f.write(f"Training timestamp: {results['training_timestamp']}\n\n")
                
                for split_name, metrics in results.items():
                    if split_name.endswith('_metrics'):
                        split_label = split_name.replace('_metrics', '').upper()
                        f.write(f"{split_label} SET METRICS:\n")
                        f.write("-"*70 + "\n")
                        for key, value in metrics.items():
                            if key != 'confusion_matrix':
                                f.write(f"  {key:.<30} {value:.4f}\n")
                        
                        cm = metrics['confusion_matrix']
                        f.write(f"\n  Confusion Matrix:\n")
                        f.write(f"    True Negatives:  {cm['tn']}\n")
                        f.write(f"    False Positives: {cm['fp']}\n")
                        f.write(f"    False Negatives: {cm['fn']}\n")
                        f.write(f"    True Positives:  {cm['tp']}\n\n")
        
        logger.info(f"Text report saved: {text_report_path}")
    
    def display_weight_files(self):
        """Display all saved weight files"""
        logger.info("\n" + "="*70)
        logger.info("SAVED MODEL WEIGHT FILES")
        logger.info("="*70)
        
        if not self.model_dir.exists():
            logger.error("Model directory does not exist!")
            return
        
        weight_files = list(self.model_dir.glob('*'))
        
        if not weight_files:
            logger.warning("No weight files found!")
            return
        
        logger.info("\nModel files in backend/models/:\n")
        
        total_size = 0
        for i, file in enumerate(sorted(weight_files), 1):
            size = file.stat().st_size / (1024 * 1024)  # Convert to MB
            total_size += size
            
            file_type = file.suffix
            logger.info(f"{i}. {file.name}")
            logger.info(f"   Size: {size:.2f} MB")
            logger.info(f"   Type: {file_type}")
            logger.info(f"   Path: {file.absolute()}\n")
        
        logger.info(f"Total size: {total_size:.2f} MB")
        
        return weight_files


def main():
    """Main execution"""
    logger.info("="*70)
    logger.info("SONG VIRALITY PREDICTION - MODEL TRAINING")
    logger.info("="*70 + "\n")
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_dir='backend/data',
        model_dir='backend/models',
        output_dir='backend/training_reports'
    )
    
    # Load splits
    if not trainer.load_splits():
        logger.error("Failed to load dataset splits")
        return False
    
    # Train XGBoost
    xgb_model, xgb_results = trainer.train_xgboost()
    
    # Train LSTM (optional)
    if TENSORFLOW_AVAILABLE:
        lstm_model, lstm_results = trainer.train_lstm()
    else:
        logger.info("\nSkipping LSTM training (TensorFlow not available)")
    
    # Save training report
    trainer.save_training_report()
    
    # Display weight files
    trainer.display_weight_files()
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
