#!/usr/bin/env python3
"""
Retrain Models with Proper Probability Calibration
===================================================

This fixes the issue where hit songs get low probabilities.
Uses isotonic regression calibration on all models.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from xgboost import XGBClassifier
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Retrain all models with proper calibration"""
    
    data_dir = Path('backend/data')
    model_dir = Path('backend/models')
    
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(data_dir / 'train_dataset.csv')
    test_df = pd.read_csv(data_dir / 'test_dataset.csv')
    val_df = pd.read_csv(data_dir / 'validation_dataset.csv')
    
    y_train = train_df.pop('is_hit')
    y_test = test_df.pop('is_hit')
    y_val = val_df.pop('is_hit')
    
    X_train = train_df
    X_test = test_df
    X_val = val_df
    
    feature_names = list(X_train.columns)
    
    # Class balance
    n_hits = (y_train == 1).sum()
    n_non_hits = (y_train == 0).sum()
    scale_weight = n_non_hits / n_hits
    
    logger.info(f"Train: {len(X_train)} samples ({n_hits} hits, {n_non_hits} non-hits)")
    logger.info(f"Class weight ratio: {scale_weight:.2f}")
    
    # Scaler for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    # ============================================================
    # 1. Train XGBoost with CALIBRATION
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("Training CALIBRATED XGBoost...")
    
    base_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        scale_pos_weight=scale_weight,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    # Calibrate XGBoost using isotonic regression
    xgb_calibrated = CalibratedClassifierCV(base_xgb, method='isotonic', cv=3)
    xgb_calibrated.fit(X_train, y_train)
    
    xgb_proba = xgb_calibrated.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba > 0.5).astype(int)
    logger.info(f"XGBoost - Acc: {accuracy_score(y_test, xgb_pred):.4f}, AUC: {roc_auc_score(y_test, xgb_proba):.4f}, Recall: {recall_score(y_test, xgb_pred):.4f}")
    
    # ============================================================
    # 2. Train Random Forest with CALIBRATION
    # ============================================================
    logger.info("\nTraining CALIBRATED Random Forest...")
    
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Calibrate RF
    rf_calibrated = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
    rf_calibrated.fit(X_train, y_train)
    
    rf_proba = rf_calibrated.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba > 0.5).astype(int)
    logger.info(f"Random Forest - Acc: {accuracy_score(y_test, rf_pred):.4f}, AUC: {roc_auc_score(y_test, rf_proba):.4f}, Recall: {recall_score(y_test, rf_pred):.4f}")
    
    # ============================================================
    # 3. Train Logistic Regression with CALIBRATION
    # ============================================================
    logger.info("\nTraining CALIBRATED Logistic Regression...")
    
    base_lr = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        C=1.0
    )
    
    lr_calibrated = CalibratedClassifierCV(base_lr, method='isotonic', cv=3)
    lr_calibrated.fit(X_train_scaled, y_train)
    
    lr_proba = lr_calibrated.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_proba > 0.5).astype(int)
    logger.info(f"Logistic Regression - Acc: {accuracy_score(y_test, lr_pred):.4f}, AUC: {roc_auc_score(y_test, lr_proba):.4f}, Recall: {recall_score(y_test, lr_pred):.4f}")
    
    # ============================================================
    # 4. Evaluate Ensemble
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE EVALUATION")
    
    # Equal weight ensemble
    ensemble_proba = (xgb_proba + rf_proba + lr_proba) / 3
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_proba)
    recall = recall_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred)
    
    logger.info(f"Ensemble - Acc: {acc:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    # Check calibration by looking at probability distribution
    logger.info(f"\nProbability distribution on test set:")
    logger.info(f"  Mean probability for HITS: {ensemble_proba[y_test==1].mean():.3f}")
    logger.info(f"  Mean probability for NON-HITS: {ensemble_proba[y_test==0].mean():.3f}")
    
    # ============================================================
    # 5. Save Models
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("SAVING CALIBRATED MODELS")
    
    joblib.dump(xgb_calibrated, model_dir / 'ensemble_xgb.pkl')
    joblib.dump(rf_calibrated, model_dir / 'ensemble_rf.pkl')
    joblib.dump(lr_calibrated, model_dir / 'ensemble_lr.pkl')
    joblib.dump(scaler, model_dir / 'ensemble_scaler.pkl')
    joblib.dump(feature_names, model_dir / 'ensemble_features.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'calibrated_ensemble',
        'calibration_method': 'isotonic',
        'accuracy': float(acc),
        'auc': float(auc),
        'recall': float(recall),
        'precision': float(precision),
        'hit_mean_prob': float(ensemble_proba[y_test==1].mean()),
        'non_hit_mean_prob': float(ensemble_proba[y_test==0].mean()),
        'training_samples': len(X_train),
        'features': feature_names
    }
    
    with open(model_dir / 'ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✓ All calibrated models saved!")
    logger.info(f"\nHit songs should now have mean probability ~{ensemble_proba[y_test==1].mean()*100:.1f}%")
    logger.info(f"Non-hit songs should now have mean probability ~{ensemble_proba[y_test==0].mean()*100:.1f}%")
    
    return True


if __name__ == '__main__':
    main()
