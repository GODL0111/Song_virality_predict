#!/usr/bin/env python3
"""
Retrain with Balanced Data for Better Hit Detection
=====================================================

This creates a balanced dataset so models learn to give
higher probabilities to hit-like songs.
Includes LSTM neural network with visible epochs.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.utils import resample
from xgboost import XGBClassifier
import logging
import json

# TensorFlow for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Retrain with balanced data"""
    
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
    
    logger.info(f"Original: {len(X_train)} samples ({(y_train==1).sum()} hits, {(y_train==0).sum()} non-hits)")
    
    # Manual oversampling: duplicate hits to match non-hits
    logger.info("\nBalancing dataset with oversampling...")
    
    # Separate hits and non-hits
    hits_X = X_train[y_train == 1]
    hits_y = y_train[y_train == 1]
    non_hits_X = X_train[y_train == 0]
    non_hits_y = y_train[y_train == 0]
    
    # Oversample hits to match non-hits count
    hits_X_upsampled, hits_y_upsampled = resample(
        hits_X, hits_y,
        replace=True,
        n_samples=len(non_hits_X),
        random_state=42
    )
    
    # Combine into balanced dataset
    X_train_balanced = pd.concat([non_hits_X, hits_X_upsampled])
    y_train_balanced = pd.concat([non_hits_y, hits_y_upsampled])
    
    logger.info(f"After balancing: {len(X_train_balanced)} samples ({(y_train_balanced==1).sum()} hits, {(y_train_balanced==0).sum()} non-hits)")
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced.iloc[shuffle_idx].reset_index(drop=True)
    y_train_balanced = y_train_balanced.iloc[shuffle_idx].reset_index(drop=True)
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================================
    # 1. Train XGBoost on BALANCED data
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost on BALANCED data...")
    
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train_balanced, y_train_balanced)
    
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba > 0.5).astype(int)
    logger.info(f"XGBoost - Acc: {accuracy_score(y_test, xgb_pred):.4f}, AUC: {roc_auc_score(y_test, xgb_proba):.4f}, Recall: {recall_score(y_test, xgb_pred):.4f}")
    logger.info(f"  Hit mean prob: {xgb_proba[y_test==1].mean():.3f}, Non-hit mean prob: {xgb_proba[y_test==0].mean():.3f}")
    
    # ============================================================
    # 2. Train Random Forest on BALANCED data
    # ============================================================
    logger.info("\nTraining Random Forest on BALANCED data...")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba > 0.5).astype(int)
    logger.info(f"Random Forest - Acc: {accuracy_score(y_test, rf_pred):.4f}, AUC: {roc_auc_score(y_test, rf_proba):.4f}, Recall: {recall_score(y_test, rf_pred):.4f}")
    logger.info(f"  Hit mean prob: {rf_proba[y_test==1].mean():.3f}, Non-hit mean prob: {rf_proba[y_test==0].mean():.3f}")
    
    # ============================================================
    # 3. Train Logistic Regression on BALANCED data
    # ============================================================
    logger.info("\nTraining Logistic Regression on BALANCED data...")
    
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0
    )
    
    lr_model.fit(X_train_scaled, y_train_balanced)
    
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_proba > 0.5).astype(int)
    logger.info(f"Logistic Regression - Acc: {accuracy_score(y_test, lr_pred):.4f}, AUC: {roc_auc_score(y_test, lr_proba):.4f}, Recall: {recall_score(y_test, lr_pred):.4f}")
    logger.info(f"  Hit mean prob: {lr_proba[y_test==1].mean():.3f}, Non-hit mean prob: {lr_proba[y_test==0].mean():.3f}")
    
    # ============================================================
    # 4. Train LSTM Neural Network (with visible epochs!)
    # ============================================================
    lstm_proba = None
    lstm_model = None
    
    if TENSORFLOW_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info("Training LSTM Neural Network on BALANCED data...")
        logger.info("(You will see epochs below!)")
        logger.info("="*60)
        
        # Reshape for LSTM: (samples, features, 1)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        # Build LSTM model
        lstm_model = models.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], 1), return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        # Train with VISIBLE EPOCHS (verbose=1)
        logger.info("\nStarting LSTM training with 30 epochs...")
        history = lstm_model.fit(
            X_train_lstm, y_train_balanced,
            validation_split=0.15,
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            verbose=1  # THIS SHOWS EPOCHS!
        )
        
        # Evaluate LSTM
        lstm_proba = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        lstm_pred = (lstm_proba > 0.5).astype(int)
        
        logger.info(f"\nLSTM - Acc: {accuracy_score(y_test, lstm_pred):.4f}, AUC: {roc_auc_score(y_test, lstm_proba):.4f}, Recall: {recall_score(y_test, lstm_pred):.4f}")
        logger.info(f"  Hit mean prob: {lstm_proba[y_test==1].mean():.3f}, Non-hit mean prob: {lstm_proba[y_test==0].mean():.3f}")
    else:
        logger.warning("TensorFlow not available - skipping LSTM training")
    
    # ============================================================
    # 5. Evaluate Ensemble (now with optional LSTM)
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE EVALUATION")
    
    # Include LSTM if available
    if lstm_proba is not None:
        # 4-model ensemble: XGB + RF + LR + LSTM
        ensemble_proba = (xgb_proba + rf_proba + lr_proba + lstm_proba) / 4
        logger.info("Using 4-model ensemble: XGBoost + RandomForest + LogisticRegression + LSTM")
    else:
        # 3-model ensemble: XGB + RF + LR
        ensemble_proba = (xgb_proba + rf_proba + lr_proba) / 3
        logger.info("Using 3-model ensemble: XGBoost + RandomForest + LogisticRegression")
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    
    acc = accuracy_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_proba)
    recall = recall_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred)
    
    logger.info(f"Ensemble - Acc: {acc:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    logger.info(f"\n*** Mean probability for HITS: {ensemble_proba[y_test==1].mean()*100:.1f}% ***")
    logger.info(f"*** Mean probability for NON-HITS: {ensemble_proba[y_test==0].mean()*100:.1f}% ***")
    
    # ============================================================
    # 6. Save Models
    # ============================================================
    logger.info("\n" + "="*60)
    logger.info("SAVING BALANCED MODELS")
    
    joblib.dump(xgb_model, model_dir / 'ensemble_xgb.pkl')
    joblib.dump(rf_model, model_dir / 'ensemble_rf.pkl')
    joblib.dump(lr_model, model_dir / 'ensemble_lr.pkl')
    joblib.dump(scaler, model_dir / 'ensemble_scaler.pkl')
    joblib.dump(feature_names, model_dir / 'ensemble_features.pkl')
    
    # Save LSTM if trained
    if lstm_model is not None:
        lstm_model.save(model_dir / 'song_hit_model_lstm.h5')
        logger.info("✓ LSTM model saved!")
    
    # Save metadata
    metadata = {
        'model_type': 'smote_balanced_ensemble',
        'accuracy': float(acc),
        'auc': float(auc),
        'recall': float(recall),
        'precision': float(precision),
        'hit_mean_prob': float(ensemble_proba[y_test==1].mean()),
        'non_hit_mean_prob': float(ensemble_proba[y_test==0].mean()),
        'training_samples': len(X_train_balanced),
        'features': feature_names
    }
    
    with open(model_dir / 'ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n✓ All SMOTE-balanced models saved!")
    
    return True


if __name__ == '__main__':
    main()
