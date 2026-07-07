#!/usr/bin/env python3
"""
Complete Model Retraining with Unified Data Pipeline
=====================================================

This script:
1. Runs the unified data pipeline to combine all 3 datasets
2. Trains a new ensemble model (XGBoost + RandomForest + Logistic Regression)
3. Saves new weights with improved accuracy

Run this script to retrain the model with all available data.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
from datetime import datetime
import logging

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# Import our data pipeline
from data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedModelTrainer:
    """
    Trains ensemble models using the unified data pipeline.
    Combines all available datasets for maximum training data.
    """
    
    def __init__(self, 
                 datasets_dir='datasets', 
                 data_dir='backend/data', 
                 model_dir='backend/models'):
        """Initialize the trainer"""
        self.datasets_dir = Path(datasets_dir)
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.feature_names = None
        
        # Models
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        
        # Metrics
        self.metrics = {}
        self.training_start = None
        
    def run_data_pipeline(self, hit_threshold=50):
        """Run the data pipeline to prepare unified dataset"""
        logger.info("="*70)
        logger.info("STEP 1: RUNNING UNIFIED DATA PIPELINE")
        logger.info("="*70)
        
        pipeline = DataPipeline(
            datasets_dir=str(self.datasets_dir),
            output_dir=str(self.data_dir)
        )
        
        success = pipeline.run(
            deduplicate=True,
            hit_threshold=hit_threshold,
            train_ratio=0.7,
            test_ratio=0.15,
            val_ratio=0.15,
            save_combined=True
        )
        
        if success:
            self.metrics['pipeline_stats'] = pipeline.stats
            logger.info(f"Pipeline complete: {pipeline.stats.get('unique_tracks', 0):,} unique tracks")
        
        return success
    
    def load_data(self):
        """Load the prepared dataset splits"""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: LOADING PREPARED DATA")
        logger.info("="*70)
        
        train_path = self.data_dir / 'train_dataset.csv'
        test_path = self.data_dir / 'test_dataset.csv'
        val_path = self.data_dir / 'validation_dataset.csv'
        
        if not all([p.exists() for p in [train_path, test_path, val_path]]):
            logger.error("Dataset splits not found! Run data pipeline first.")
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
        
        self.feature_names = list(self.X_train.columns)
        
        # Log data stats
        logger.info(f"\nDataset loaded:")
        logger.info(f"  Train: {len(self.X_train):>7,} samples")
        logger.info(f"  Test:  {len(self.X_test):>7,} samples")
        logger.info(f"  Valid: {len(self.X_val):>7,} samples")
        logger.info(f"  Total: {len(self.X_train) + len(self.X_test) + len(self.X_val):>7,} samples")
        logger.info(f"\nFeatures: {self.feature_names}")
        
        # Class distribution
        train_hits = (self.y_train == 1).sum()
        train_total = len(self.y_train)
        logger.info(f"\nTrain class distribution:")
        logger.info(f"  Hits:     {train_hits:>6,} ({train_hits/train_total*100:.1f}%)")
        logger.info(f"  Non-hits: {train_total - train_hits:>6,} ({(train_total-train_hits)/train_total*100:.1f}%)")
        
        return True
    
    def train_xgboost(self):
        """Train XGBoost model with optimized parameters for hit detection"""
        logger.info("\n" + "-"*50)
        logger.info("Training XGBoost Classifier...")
        logger.info("-"*50)
        
        # Calculate optimal scale_pos_weight for BETTER HIT RECALL
        n_hits = (self.y_train == 1).sum()
        n_non_hits = (self.y_train == 0).sum()
        # Use actual ratio for better hit detection - true hits should score high
        scale_pos_weight = n_non_hits / n_hits
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        self.xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='auc',                        # Optimize for AUC
            random_state=42,
            scale_pos_weight=scale_pos_weight,        # Full weight for better recall
            max_depth=8,                              # Deeper trees capture complex patterns
            learning_rate=0.08,                       # Slightly lower for better generalization
            n_estimators=300,                         # More trees for accuracy
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,                       # Allow capturing smaller patterns
            gamma=0.05,                               # Less regularization for sensitivity
            reg_alpha=0.05,
            reg_lambda=0.8,
            n_jobs=-1
        )
        
        self.xgb_model.fit(
            self.X_train, self.y_train, 
            eval_set=[(self.X_val, self.y_val)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.xgb_model.predict(self.X_test)
        y_proba = self.xgb_model.predict_proba(self.X_test)[:, 1]
        
        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        
        logger.info(f"XGBoost - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        self.metrics['xgboost'] = {
            'accuracy': float(acc),
            'auc': float(auc),
            'precision': float(precision_score(self.y_test, y_pred)),
            'recall': float(recall_score(self.y_test, y_pred)),
            'f1': float(f1_score(self.y_test, y_pred))
        }
        
        return True
    
    def train_random_forest(self):
        """Train Random Forest model optimized for hit detection"""
        logger.info("\n" + "-"*50)
        logger.info("Training Random Forest Classifier...")
        logger.info("-"*50)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=300,          # More trees for better patterns
            max_depth=15,              # Deeper for complex patterns
            min_samples_split=8,       # Balance between overfit/underfit
            min_samples_leaf=3,
            class_weight='balanced',   # Balanced for minority class
            random_state=42,
            n_jobs=-1,
            max_features='sqrt'        # Standard for RF
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict(self.X_test)
        y_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        
        logger.info(f"Random Forest - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        self.metrics['random_forest'] = {
            'accuracy': float(acc),
            'auc': float(auc),
            'precision': float(precision_score(self.y_test, y_pred)),
            'recall': float(recall_score(self.y_test, y_pred)),
            'f1': float(f1_score(self.y_test, y_pred))
        }
        
        return True
    
    def train_logistic_regression(self):
        """Train calibrated Logistic Regression"""
        logger.info("\n" + "-"*50)
        logger.info("Training Calibrated Logistic Regression...")
        logger.info("-"*50)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        X_val_scaled = self.scaler.transform(self.X_val)
        
        # Base model - balanced for minority class
        base_lr = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            solver='lbfgs',
            C=0.5  # Regularization
        )
        
        # Calibrate using isotonic regression
        self.lr_model = CalibratedClassifierCV(
            base_lr,
            method='isotonic',
            cv=5
        )
        
        self.lr_model.fit(X_train_scaled, self.y_train)
        
        # Evaluate
        y_pred = self.lr_model.predict(X_test_scaled)
        y_proba = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        
        logger.info(f"Logistic Regression - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
        
        self.metrics['logistic_regression'] = {
            'accuracy': float(acc),
            'auc': float(auc),
            'precision': float(precision_score(self.y_test, y_pred)),
            'recall': float(recall_score(self.y_test, y_pred)),
            'f1': float(f1_score(self.y_test, y_pred))
        }
        
        return True
    
    def evaluate_ensemble(self):
        """Evaluate the ensemble model"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: EVALUATING ENSEMBLE MODEL")
        logger.info("="*70)
        
        # Get predictions from all models
        xgb_proba = self.xgb_model.predict_proba(self.X_test)[:, 1]
        rf_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        
        X_test_scaled = self.scaler.transform(self.X_test)
        lr_proba = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # Weighted ensemble - balanced weights for better accuracy
        # XGB=4, RF=4, LR=2 (more balanced approach)
        ensemble_proba = (4 * xgb_proba + 4 * rf_proba + 2 * lr_proba) / 10
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(self.y_test, ensemble_pred)
        auc = roc_auc_score(self.y_test, ensemble_proba)
        precision = precision_score(self.y_test, ensemble_pred)
        recall = recall_score(self.y_test, ensemble_pred)
        f1 = f1_score(self.y_test, ensemble_pred)
        
        logger.info(f"\nEnsemble Performance on Test Set:")
        logger.info(f"  Accuracy:  {acc:.4f}")
        logger.info(f"  AUC-ROC:   {auc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, ensemble_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {cm[0,0]:>6,}")
        logger.info(f"  False Positives: {cm[0,1]:>6,}")
        logger.info(f"  False Negatives: {cm[1,0]:>6,}")
        logger.info(f"  True Positives:  {cm[1,1]:>6,}")
        
        # Classification report
        logger.info(f"\nClassification Report:")
        logger.info(classification_report(self.y_test, ensemble_pred, 
                                          target_names=['Non-Hit', 'Hit']))
        
        self.metrics['ensemble'] = {
            'accuracy': float(acc),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        # Also evaluate on validation set
        xgb_val_proba = self.xgb_model.predict_proba(self.X_val)[:, 1]
        rf_val_proba = self.rf_model.predict_proba(self.X_val)[:, 1]
        X_val_scaled = self.scaler.transform(self.X_val)
        lr_val_proba = self.lr_model.predict_proba(X_val_scaled)[:, 1]
        
        ensemble_val_proba = (5 * xgb_val_proba + 4 * rf_val_proba + 1 * lr_val_proba) / 10
        ensemble_val_pred = (ensemble_val_proba > 0.5).astype(int)
        
        val_acc = accuracy_score(self.y_val, ensemble_val_pred)
        val_auc = roc_auc_score(self.y_val, ensemble_val_proba)
        
        logger.info(f"\nValidation Set Performance:")
        logger.info(f"  Accuracy: {val_acc:.4f}")
        logger.info(f"  AUC-ROC:  {val_auc:.4f}")
        
        self.metrics['validation'] = {
            'accuracy': float(val_acc),
            'auc': float(val_auc)
        }
        
        return True
    
    def save_models(self):
        """Save all trained models and metadata"""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: SAVING MODELS")
        logger.info("="*70)
        
        try:
            # Save individual models
            joblib.dump(self.xgb_model, self.model_dir / 'ensemble_xgb.pkl')
            logger.info(f"  Saved: ensemble_xgb.pkl")
            
            joblib.dump(self.rf_model, self.model_dir / 'ensemble_rf.pkl')
            logger.info(f"  Saved: ensemble_rf.pkl")
            
            joblib.dump(self.lr_model, self.model_dir / 'ensemble_lr.pkl')
            logger.info(f"  Saved: ensemble_lr.pkl")
            
            joblib.dump(self.scaler, self.model_dir / 'ensemble_scaler.pkl')
            logger.info(f"  Saved: ensemble_scaler.pkl")
            
            joblib.dump(self.feature_names, self.model_dir / 'ensemble_features.pkl')
            logger.info(f"  Saved: ensemble_features.pkl")
            
            # Also save as the default model
            joblib.dump(self.xgb_model, self.model_dir / 'song_hit_model.pkl')
            joblib.dump(self.feature_names, self.model_dir / 'song_hit_model_features.pkl')
            logger.info(f"  Saved: song_hit_model.pkl (XGBoost default)")
            
            # Save metadata
            training_time = (datetime.now() - self.training_start).total_seconds()
            
            metadata = {
                'model_type': 'Ensemble (XGBoost + RandomForest + LogisticRegression)',
                'training_date': datetime.now().isoformat(),
                'training_time_seconds': training_time,
                'data_source': 'unified_pipeline',
                'datasets_combined': ['dataset.csv', 'spotify_songs.csv', 'spotify_tracks.csv'],
                'total_samples': len(self.X_train) + len(self.X_test) + len(self.X_val),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'val_samples': len(self.X_val),
                'feature_names': self.feature_names,
                'metrics': self.metrics,
                'ensemble_weights': {'xgboost': 4, 'random_forest': 4, 'logistic_regression': 2},
                'hit_threshold': 50,
                'bias_correction': 'enabled',
                'version': '2.0.0'
            }
            
            with open(self.model_dir / 'ensemble_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"  Saved: ensemble_metadata.json")
            
            # Also save as model_metadata.json for compatibility
            with open(self.model_dir / 'model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"  Saved: model_metadata.json")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def train(self):
        """Run the complete training pipeline"""
        self.training_start = datetime.now()
        
        logger.info("\n" + "="*70)
        logger.info("UNIFIED MODEL TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Started at: {self.training_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Run data pipeline
        if not self.run_data_pipeline():
            logger.error("Data pipeline failed!")
            return False
        
        # Step 2: Load data
        if not self.load_data():
            logger.error("Failed to load data!")
            return False
        
        # Step 3: Train models
        logger.info("\n" + "="*70)
        logger.info("STEP 3: TRAINING MODELS")
        logger.info("="*70)
        
        self.train_xgboost()
        self.train_random_forest()
        self.train_logistic_regression()
        
        # Step 4: Evaluate ensemble
        self.evaluate_ensemble()
        
        # Step 5: Save models
        self.save_models()
        
        # Final summary
        training_time = (datetime.now() - self.training_start).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nTotal training time: {training_time:.1f} seconds")
        logger.info(f"\nFinal Ensemble Performance:")
        logger.info(f"  Test Accuracy:  {self.metrics['ensemble']['accuracy']:.4f}")
        logger.info(f"  Test AUC-ROC:   {self.metrics['ensemble']['auc']:.4f}")
        logger.info(f"  Test F1 Score:  {self.metrics['ensemble']['f1']:.4f}")
        logger.info(f"\n  Val Accuracy:   {self.metrics['validation']['accuracy']:.4f}")
        logger.info(f"  Val AUC-ROC:    {self.metrics['validation']['auc']:.4f}")
        logger.info(f"\nModels saved to: {self.model_dir}")
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train ensemble model with unified data pipeline'
    )
    parser.add_argument(
        '--datasets-dir',
        default='datasets',
        help='Directory containing raw datasets'
    )
    parser.add_argument(
        '--data-dir',
        default='backend/data',
        help='Directory for processed data'
    )
    parser.add_argument(
        '--model-dir',
        default='backend/models',
        help='Directory to save models'
    )
    
    args = parser.parse_args()
    
    trainer = UnifiedModelTrainer(
        datasets_dir=args.datasets_dir,
        data_dir=args.data_dir,
        model_dir=args.model_dir
    )
    
    success = trainer.train()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
