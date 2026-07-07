#!/usr/bin/env python3
"""
Advanced Ensemble Model Training for Song Virality Prediction
===============================================================

Uses multiple models with probability calibration for accurate predictions:
1. XGBoost (gradient boosting)
2. Random Forest (bagging)
3. Logistic Regression with calibration
4. Voting ensemble with optimized weights

This addresses severe underprediction issues by combining multiple models.
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModelTrainer:
    def __init__(self, data_dir='backend/data', model_dir='backend/models'):
        """Initialize ensemble model trainer"""
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        
        self.scaler = StandardScaler()
        self.ensemble_model = None
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
    
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
    
    def train_ensemble(self):
        """Train ensemble model with calibration"""
        logger.info("\n" + "="*70)
        logger.info("TRAINING ENSEMBLE MODEL")
        logger.info("="*70)
        
        # Calculate class weights
        n_samples = len(self.y_train)
        n_hits = (self.y_train == 1).sum()
        n_non_hits = (self.y_train == 0).sum()
        scale_pos_weight = n_non_hits / n_hits
        
        logger.info(f"Class distribution: Hits={n_hits}, Non-hits={n_non_hits}")
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Sample weights
        sample_weights = np.where(self.y_train == 1, n_samples / (2 * n_hits), n_samples / (2 * n_non_hits))
        
        # Scale features for LogisticRegression
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # === MODEL 1: XGBoost with aggressive settings ===
        logger.info("\n1. Training XGBoost...")
        self.xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=scale_pos_weight * 1.5,  # Extra boost for minority class
            max_depth=7,
            learning_rate=0.03,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        self.xgb_model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        
        # === MODEL 2: Random Forest with balanced weights ===
        logger.info("2. Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        
        # === MODEL 3: Calibrated Logistic Regression ===
        logger.info("3. Training Calibrated Logistic Regression...")
        base_lr = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            C=0.1
        )
        # Use isotonic calibration for better probability estimates
        self.lr_model = CalibratedClassifierCV(
            base_lr, 
            method='isotonic',
            cv=5
        )
        self.lr_model.fit(X_train_scaled, self.y_train)
        
        # === ENSEMBLE: Weighted Voting ===
        logger.info("4. Creating Weighted Ensemble...")
        # XGBoost gets highest weight, then RF, then calibrated LR
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', self.xgb_model),
                ('rf', self.rf_model),
                ('lr', self.lr_model)
            ],
            voting='soft',  # Use probability voting
            weights=[3, 2, 2],  # XGBoost gets 3x weight
            n_jobs=-1
        )
        
        # Note: VotingClassifier needs to be fit, but we already trained individual models
        # So we'll use a custom prediction approach
        
        logger.info("✓ Ensemble training complete")
        
        return True
    
    def evaluate_models(self):
        """Evaluate all models"""
        logger.info("\n" + "="*70)
        logger.info("MODEL EVALUATION")
        logger.info("="*70)
        
        X_test_scaled = self.scaler.transform(self.X_test)
        
        models = {
            'XGBoost': (self.xgb_model, self.X_test),
            'Random Forest': (self.rf_model, self.X_test),
            'Calibrated LogReg': (self.lr_model, X_test_scaled)
        }
        
        results = {}
        
        for name, (model, X_test_data) in models.items():
            logger.info(f"\n{name}:")
            logger.info("-" * 70)
            
            y_pred = model.predict(X_test_data)
            y_pred_proba = model.predict_proba(X_test_data)[:, 1]
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
            logger.info(f"  AUC-ROC:   {auc:.4f}")
            
            # Check average predicted probability for hits
            hit_mask = self.y_test == 1
            avg_hit_prob = y_pred_proba[hit_mask].mean() if hit_mask.sum() > 0 else 0
            logger.info(f"  Avg probability for actual hits: {avg_hit_prob:.4f}")
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'avg_hit_probability': avg_hit_prob
            }
        
        # Ensemble prediction
        logger.info(f"\nWeighted Ensemble:")
        logger.info("-" * 70)
        
        xgb_proba = self.xgb_model.predict_proba(self.X_test)[:, 1]
        rf_proba = self.rf_model.predict_proba(self.X_test)[:, 1]
        lr_proba = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # Weighted average (3:2:2)
        ensemble_proba = (3 * xgb_proba + 2 * rf_proba + 2 * lr_proba) / 7
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(self.y_test, ensemble_pred)
        precision = precision_score(self.y_test, ensemble_pred, zero_division=0)
        recall = recall_score(self.y_test, ensemble_pred)
        f1 = f1_score(self.y_test, ensemble_pred)
        auc = roc_auc_score(self.y_test, ensemble_proba)
        
        hit_mask = self.y_test == 1
        avg_hit_prob = ensemble_proba[hit_mask].mean() if hit_mask.sum() > 0 else 0
        
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        logger.info(f"  AUC-ROC:   {auc:.4f}")
        logger.info(f"  Avg probability for actual hits: {avg_hit_prob:.4f}")
        
        results['Ensemble'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'avg_hit_probability': avg_hit_prob
        }
        
        return results
    
    def save_models(self):
        """Save all models"""
        logger.info("\n" + "="*70)
        logger.info("SAVING MODELS")
        logger.info("="*70)
        
        # Save individual models
        joblib.dump(self.xgb_model, self.model_dir / 'ensemble_xgb.pkl')
        joblib.dump(self.rf_model, self.model_dir / 'ensemble_rf.pkl')
        joblib.dump(self.lr_model, self.model_dir / 'ensemble_lr.pkl')
        joblib.dump(self.scaler, self.model_dir / 'ensemble_scaler.pkl')
        
        # Save feature names
        feature_names = list(self.X_train.columns)
        joblib.dump(feature_names, self.model_dir / 'ensemble_features.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'Ensemble (XGBoost + RF + Calibrated LR)',
            'model_framework': 'ensemble',
            'created_at': datetime.now().isoformat(),
            'feature_names': feature_names,
            'weights': {'xgb': 3, 'rf': 2, 'lr': 2},
            'bias_correction': 'ensemble voting with calibrated probabilities',
            'data_size': len(self.X_train) + len(self.X_test)
        }
        
        with open(self.model_dir / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✓ Models saved:")
        logger.info(f"  - ensemble_xgb.pkl")
        logger.info(f"  - ensemble_rf.pkl")
        logger.info(f"  - ensemble_lr.pkl")
        logger.info(f"  - ensemble_scaler.pkl")
        logger.info(f"  - ensemble_features.pkl")
        logger.info(f"  - ensemble_metadata.json")
        
        return True


def main():
    """Main training function"""
    logger.info("\n" + "#"*70)
    logger.info("# ENSEMBLE MODEL TRAINING")
    logger.info("#"*70 + "\n")
    
    trainer = EnsembleModelTrainer()
    
    if not trainer.load_splits():
        return False
    
    if not trainer.train_ensemble():
        return False
    
    results = trainer.evaluate_models()
    
    if not trainer.save_models():
        return False
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
