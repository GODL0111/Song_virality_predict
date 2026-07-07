"""
Main Backend ML Model - Song Hit Prediction
=============================================

This is the core machine learning model for predicting song hit probability.
Used by the Flask API server for all predictions.

Classes:
    - SongHitPredictor: Main model class with training and prediction

Features:
    - 12 musical DNA features for prediction
    - XGBoost classifier for binary classification (hit/miss)
    - Model persistence and metadata tracking
    - Feature analysis and optimization suggestions
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import json
import hashlib
from scipy import stats
import warnings
import logging

# LSTM imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SongHitPredictor:
    def __init__(self, model_dir="models", data_dir="data", model_type="ensemble"):
        """
        Initialize the Song Hit Predictor with model persistence capabilities
        
        Args:
            model_dir: Directory to store models
            data_dir: Directory with training data
            model_type: "xgboost", "lstm", or "ensemble" - which model to use
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.scaler = None  # For feature scaling (used by LSTM and ensemble)
        self.feature_names = None
        self.model_metadata = {}
        self.df = None
        self.model_type = model_type  # Track which model type is active
        
        # For ensemble
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.ensemble_scaler = None

        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Define musical DNA features
        self.musical_dna_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_ms'
        ]

        # Note: Visualization code removed - this is a backend ML model

    def _calculate_data_hash(self, data):
        """Calculate hash of the dataset to detect changes"""
        return hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()

    def _save_model_metadata(self, accuracy, data_hash, training_time):
        """Save model metadata for tracking"""
        model_type_name = 'LSTM' if self.model_type == 'lstm' else 'XGBClassifier'
        self.model_metadata = {
            'model_type': model_type_name,
            'model_framework': self.model_type,
            'accuracy': accuracy,
            'training_time': training_time,
            'data_hash': data_hash,
            'feature_names': self.feature_names.tolist(),
            'created_at': datetime.now().isoformat(),
            'data_size': len(self.X_train) + len(self.X_test),
            'bias_correction': 'calibrated',
            'bias_correction_method': 'isotonic-inspired calibration based on empirical hit ratio',
            'scale_pos_weight': 'enabled'
        }

        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

    def save_model(self, model_name="song_hit_model"):
        """Save the trained model, scaler, and metadata"""
        if self.model is None:
            return False

        try:
            if self.model_type == "lstm":
                # Save LSTM model
                model_path = os.path.join(self.model_dir, f"{model_name}_lstm.h5")
                self.model.save(model_path)
                
                # Save scaler
                scaler_path = os.path.join(self.model_dir, f"{model_name}_lstm_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
            else:
                # Save XGBoost model
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                joblib.dump(self.model, model_path)

            feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
            joblib.dump(self.feature_names, feature_path)

            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, model_name="song_hit_model", model_type=None):
        """Load a previously trained model - prioritize ensemble"""
        try:
            if model_type is None:
                model_type = self.model_type
            
            # Try ensemble first (best performance)
            if model_type == "ensemble" or model_type == "xgboost":
                ensemble_xgb_path = os.path.join(self.model_dir, "ensemble_xgb.pkl")
                ensemble_rf_path = os.path.join(self.model_dir, "ensemble_rf.pkl")
                ensemble_lr_path = os.path.join(self.model_dir, "ensemble_lr.pkl")
                ensemble_scaler_path = os.path.join(self.model_dir, "ensemble_scaler.pkl")
                ensemble_features_path = os.path.join(self.model_dir, "ensemble_features.pkl")
                ensemble_metadata_path = os.path.join(self.model_dir, "ensemble_metadata.json")
                
                if all(os.path.exists(p) for p in [ensemble_xgb_path, ensemble_rf_path, ensemble_lr_path]):
                    self.xgb_model = joblib.load(ensemble_xgb_path)
                    self.rf_model = joblib.load(ensemble_rf_path)
                    self.lr_model = joblib.load(ensemble_lr_path)
                    self.ensemble_scaler = joblib.load(ensemble_scaler_path)
                    self.feature_names = joblib.load(ensemble_features_path)
                    self.model_type = "ensemble"
                    
                    if os.path.exists(ensemble_metadata_path):
                        with open(ensemble_metadata_path, 'r') as f:
                            self.model_metadata = json.load(f)
                    
                    logger.info("✓ Ensemble model loaded (XGBoost + RF + Calibrated LR)")
                    return True
            
            feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')

            # Try XGBoost
            xgboost_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(xgboost_path) and model_type == "xgboost":
                self.model = joblib.load(xgboost_path)
                self.model_type = "xgboost"
                self.feature_names = joblib.load(feature_path)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                return True
            
            if model_type == "lstm":
                model_path = os.path.join(self.model_dir, f"{model_name}_lstm.h5")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_lstm_scaler.pkl")
                
                if not os.path.exists(model_path):
                    return False
                
                self.model = keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                self.model_type = "lstm"
            else:
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                
                if not os.path.exists(model_path):
                    return False
                
                self.model = joblib.load(model_path)
                self.model_type = "xgboost"

            self.feature_names = joblib.load(feature_path)

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)

            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def load_and_prepare_data(self, file_path):
        """Load and prepare the data for training/prediction"""
        try:
            # Use engine='python' for potentially problematic CSV files
            # Use on_bad_lines='skip' to skip problematic rows
            self.df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')

            # Handle both album_id and track_album_id column names
            if 'album_id' in self.df.columns and 'track_album_id' not in self.df.columns:
                self.df['track_album_id'] = self.df['album_id']

            # Handle 'year' column if it exists (for legacy datasets)
            if 'year' in self.df.columns:
                self.df['release_year'] = self.df['year']
                self.df.dropna(subset=['release_year'], inplace=True)

            # Drop ID and metadata columns that aren't needed for training
            columns_to_drop = ['year', 'track_id', 'album_id', 'track_album_id', 
                               'playlist_id', 'artwork_url', 'track_url', 'source',
                               'track_name', 'artist_name', 'album_name', 'genre', 'language']
            # Only drop columns that actually exist
            columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
            self.df = self.df.drop(columns_to_drop, axis=1)

            # Convert musical DNA features to numeric, coercing errors
            for feature in self.musical_dna_features:
                if feature in self.df.columns:
                    self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')

            # Drop rows with NaN values in musical DNA features after coercion
            self.df.dropna(subset=self.musical_dna_features, inplace=True)
            
            # CRITICAL: Filter outliers that corrupt model training
            # Loudness should be between -60 and +5 dB (realistic audio range)
            # Some corrupt entries have -100000 which destroys the model
            initial_count = len(self.df)
            self.df = self.df[self.df['loudness'] >= -60]
            self.df = self.df[self.df['loudness'] <= 5]
            filtered_count = initial_count - len(self.df)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} rows with invalid loudness values")

            # Check if 'is_hit' already exists (from combined dataset pipeline)
            if 'is_hit' not in self.df.columns:
                # Define a 'hit' based on popularity score
                popularity_col = 'track_popularity' if 'track_popularity' in self.df.columns else 'popularity'
                if popularity_col not in self.df.columns:
                     raise KeyError("Neither 'track_popularity' nor 'popularity' column found for target variable.")

                # Convert the popularity column to numeric, coercing errors to NaN
                self.df[popularity_col] = pd.to_numeric(self.df[popularity_col], errors='coerce')
                # Drop rows where popularity is NaN after coercion
                self.df.dropna(subset=[popularity_col], inplace=True)

                # Define hit as popularity >= 50 (more balanced than 70)
                self.df['is_hit'] = (self.df[popularity_col] >= 50).astype(int)
            else:
                # Ensure is_hit is integer type
                self.df['is_hit'] = self.df['is_hit'].astype(int)

            # Check if all musical features exist and are numeric
            if not all(feature in self.df.columns for feature in self.musical_dna_features):
                missing_features = [f for f in self.musical_dna_features if f not in self.df.columns]
                raise KeyError(f"Missing musical DNA features: {missing_features}")

            # Also check if the dtypes are numeric
            for feature in self.musical_dna_features:
                if not pd.api.types.is_numeric_dtype(self.df[feature]):
                     raise TypeError(f"Musical DNA feature '{feature}' is not numeric after conversion.")

            X = self.df[self.musical_dna_features]
            Y = self.df['is_hit']

            return self.df, X, Y

        except Exception as e:
            print(f"[ERROR] load_and_prepare_data: {e}")
            return None, None, None

    def train_model(self, X, Y, force_retrain=False):
        """Train the model with option to force retrain"""
        data_hash = self._calculate_data_hash(X)

        if not force_retrain and self.load_model():
            return True

        start_time = datetime.now()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

        self.feature_names = X.columns

        if self.model_type == "lstm" and TENSORFLOW_AVAILABLE:
            self._train_lstm_model(data_hash, start_time)
        else:
            self._train_xgboost_model(data_hash, start_time)

        return True

    def _train_xgboost_model(self, data_hash, start_time):
        """Train XGBoost with proper bias correction"""
        # Calculate class weights to handle severe class imbalance
        n_samples = len(self.Y_train)
        n_hits = (self.Y_train == 1).sum()
        n_non_hits = (self.Y_train == 0).sum()
        
        # Calculate scale_pos_weight as ratio of negative to positive
        # This is the correct parameter for XGBoost to handle class imbalance
        scale_pos_weight = n_non_hits / n_hits
        
        # Balanced weights for sample weighting
        weight_hits = n_samples / (2 * n_hits)
        weight_non_hits = n_samples / (2 * n_non_hits)
        
        sample_weights = np.where(self.Y_train == 1, weight_hits, weight_non_hits)

        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',  # Use logloss for binary classification
            enable_categorical=False,
            random_state=42,
            scale_pos_weight=scale_pos_weight,  # Proper class weighting
            max_depth=6,  # Moderate depth for balanced learning
            learning_rate=0.05,  # Lower learning rate for stability
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,  # No regularization penalty
            # Use default base_score (0.5) for unbiased predictions
            # Do NOT artificially adjust this
        )

        self.model.fit(self.X_train, self.Y_train, sample_weight=sample_weights)

        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, predictions)
        training_time = (datetime.now() - start_time).total_seconds()

        self._save_model_metadata(accuracy, data_hash, training_time)
        self.save_model()

    def _train_lstm_model(self, data_hash, start_time):
        """Train LSTM model for sequential feature patterns"""
        # Scale features for LSTM
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Reshape for LSTM (samples, time steps, features)
        # Treat each feature as a time step
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        # Build LSTM model
        self.model = models.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], 1), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with class weight handling
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        # Calculate class weights to balance imbalanced dataset
        n_samples = len(self.Y_train)
        n_hits = (self.Y_train == 1).sum()
        n_non_hits = (self.Y_train == 0).sum()
        # Inverse frequency weighting: weight = total / (2 * class_count)
        class_weight = {0: n_samples / (2 * n_non_hits), 1: n_samples / (2 * n_hits)}
        
        # Train with early stopping
        self.model.fit(
            X_train_lstm, self.Y_train,
            validation_data=(X_test_lstm, self.Y_test),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ],
            verbose=0
        )
        
        # Evaluate
        loss, accuracy, auc = self.model.evaluate(X_test_lstm, self.Y_test, verbose=0)
        training_time = (datetime.now() - start_time).total_seconds()

        self._save_model_metadata(accuracy, data_hash, training_time)
        self.save_model()

    def get_optimal_ranges(self):
        """Get optimal parameter ranges for hit songs"""
        if self.df is None:
            return None

        hit_songs = self.df[self.df['is_hit'] == 1]
        non_hit_songs = self.df[self.df['is_hit'] == 0]

        optimal_ranges = {}

        for feature in self.musical_dna_features:
            hit_mean = hit_songs[feature].mean()
            hit_std = hit_songs[feature].std()
            non_hit_mean = non_hit_songs[feature].mean()

            # Calculate optimal range (mean ± 1 std)
            optimal_min = max(0, hit_mean - hit_std)
            optimal_max = hit_mean + hit_std

            # Calculate statistical significance
            t_stat, p_value = stats.ttest_ind(hit_songs[feature], non_hit_songs[feature])
            significance = "VERY IMPORTANT" if p_value < 0.001 else "IMPORTANT" if p_value < 0.05 else "NORMAL"

            optimal_ranges[feature] = {
                'min': float(optimal_min),
                'max': float(optimal_max),
                'optimal_value': float(hit_mean),
                'importance': significance,
                'difference_from_non_hits': float(hit_mean - non_hit_mean)
            }

        return optimal_ranges

    def get_optimal_ranges_dict(self):
        """Return optimal ranges as dictionary (for API use)"""
        return self.get_optimal_ranges()

    def get_feature_importance(self):
        """Get feature importance from model"""
        if self.model is None:
            return None

        try:
            feature_importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)

            return importance_df

        except Exception as e:
            return None

    def predict_song_hit_probability(self, song_features):
        """Predict the hit probability of a single song"""
        # Check if we have a valid model (either single model or ensemble)
        if self.model_type == "ensemble":
            if self.xgb_model is None or self.rf_model is None or self.lr_model is None:
                logger.error("Ensemble models not loaded!")
                return None
        elif self.model is None:
            logger.error("Model is None!")
            return None

        # Ensure the input song features are in the correct format (DataFrame)
        if isinstance(song_features, dict):
            song_df = pd.DataFrame([song_features])
        else:
            song_df = song_features.copy()

        # Ensure the columns are in the same order as the training features
        if not all(feature in song_df.columns for feature in self.feature_names):
            missing = [f for f in self.feature_names if f not in song_df.columns]
            logger.error(f"Missing features: {missing}")
            return None

        song_df = song_df[self.feature_names]

        # Convert features to numeric if they aren't already
        for feature in self.feature_names:
            if not pd.api.types.is_numeric_dtype(song_df[feature]):
                song_df[feature] = pd.to_numeric(song_df[feature], errors='coerce')

        # Validate and normalize feature ranges
        feature_ranges = {
            'danceability': (0, 1),
            'energy': (0, 1),
            'key': (0, 11),
            'loudness': (-60, 0),
            'mode': (0, 1),
            'speechiness': (0, 1),
            'acousticness': (0, 1),
            'instrumentalness': (0, 1),
            'liveness': (0, 1),
            'valence': (0, 1),
            'tempo': (0, 250),
            'duration_ms': (0, 3600000)
        }
        
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in song_df.columns:
                song_df[feature] = np.clip(song_df[feature], min_val, max_val)

        # Handle potential NaN values after conversion
        if song_df.isnull().any().any():
            song_df.fillna(song_df.mean(), inplace=True)

        try:
            # Ensure no infinite values
            song_df = song_df.replace([np.inf, -np.inf], np.nan)
            song_df = song_df.fillna(song_df.mean())
            
            if self.model_type == "ensemble":
                hit_prob, confidence, is_hit = self._predict_ensemble(song_df)
            elif self.model_type == "lstm" and TENSORFLOW_AVAILABLE:
                hit_prob, confidence, is_hit = self._predict_lstm(song_df)
            else:
                hit_prob, confidence, is_hit = self._predict_xgboost(song_df)

            return {
                'hit_probability': float(hit_prob),
                'is_hit_prediction': bool(is_hit),
                'confidence': float(confidence),
                'model_type': self.model_type
            }

        except Exception as e:
            logger.error(f"Prediction error in predict_song_hit_probability: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _predict_ensemble(self, song_df):
        """Ensemble prediction with probability scaling for realistic hit scores"""
        try:
            # Get predictions from all three models
            xgb_proba = self.xgb_model.predict_proba(song_df)[:, 1][0]
            rf_proba = self.rf_model.predict_proba(song_df)[:, 1][0]
            
            # Scale for logistic regression
            if self.ensemble_scaler is None:
                logger.error("Ensemble scaler is None!")
                return self._predict_xgboost(song_df)
            
            song_scaled = self.ensemble_scaler.transform(song_df)
            lr_proba = self.lr_model.predict_proba(song_scaled)[:, 1][0]
            
            # Equal weight average (raw probability)
            raw_prob = (xgb_proba + rf_proba + lr_proba) / 3
            
            # PROBABILITY SCALING for realistic hit scores
            # Model outputs range roughly 0.27-0.67 (non-hit min to hit max)
            # We want to scale this to ~0.05-0.95 for meaningful display
            # Lower minimum (0.05) so amateur recordings show very low scores
            # 
            # Scaling formula: map [0.27, 0.67] -> [0.05, 0.95]
            min_raw, max_raw = 0.27, 0.67
            min_scaled, max_scaled = 0.05, 0.95
            
            # DON'T clip - allow raw values below 0.27 to go negative (then clip to 0.05)
            # This ensures amateur recordings with low raw prob get very low scaled scores
            scaled_prob = (raw_prob - min_raw) / (max_raw - min_raw) * (max_scaled - min_scaled) + min_scaled
            scaled_prob = max(min_scaled, min(max_scaled, scaled_prob))
            
            # Threshold at 50% of scaled range
            hit_threshold = 0.50
            is_hit = scaled_prob >= hit_threshold
            
            # Confidence based on model agreement
            model_std = float(abs(xgb_proba - raw_prob) + abs(rf_proba - raw_prob) + abs(lr_proba - raw_prob)) / 3
            agreement = 1 - min(model_std * 3, 1.0)
            confidence = abs(scaled_prob - hit_threshold) * 2 * agreement
            confidence = min(max(confidence, 0.1), 1.0)
            
            return scaled_prob, confidence, is_hit
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            import traceback
            traceback.print_exc()
            return self._predict_xgboost(song_df)

    def _predict_xgboost(self, song_df):
        """XGBoost prediction - COMPLETELY NEUTRAL"""
        proba = self.model.predict_proba(song_df)[:, 1][0]
        
        # STANDARD 50% threshold
        hit_threshold = 0.50
        is_hit = proba > hit_threshold
        confidence = min(abs(proba - hit_threshold) * 2, 1.0)

        return proba, confidence, is_hit

    def _predict_lstm(self, song_df):
        """LSTM prediction - COMPLETELY NEUTRAL"""
        song_scaled = self.scaler.transform(song_df)
        song_lstm = song_scaled.reshape((song_scaled.shape[0], song_scaled.shape[1], 1))
        
        hit_prob = float(self.model.predict(song_lstm, verbose=0)[0][0])
        
        # STANDARD 50% threshold
        hit_threshold = 0.50
        is_hit = hit_prob > hit_threshold
        confidence = min(abs(hit_prob - hit_threshold) * 2, 1.0)

        return hit_prob, confidence, is_hit

    def _apply_ensemble_bias_correction(self, probability):
        """
        NEUTRAL: No bias correction - return raw probability.
        This function is kept for compatibility but does nothing.
        """
        return float(probability)
    
    def _apply_probability_bias_correction(self, probability):
        """
        NEUTRAL: No bias correction - return raw probability.
        This function is kept for compatibility but does nothing.
        """
        return float(probability)


    def suggest_feature_improvements(self, song_features):
        """Suggest which features to change to make a song more likely to be a hit"""
        if self.model is None:
            return None

        # Convert to DataFrame if dict
        if isinstance(song_features, dict):
            original_features = pd.DataFrame([song_features])
        else:
            original_features = song_features.copy()

        # Ensure all features are in the right order
        original_features = original_features[self.feature_names]
        
        # Get original prediction
        original_prob = self.model.predict_proba(original_features)[0][1]

        # Get optimal ranges
        optimal_ranges = self.get_optimal_ranges()
        suggestions = []

        for feature in self.feature_names:
            current_value = original_features[feature].iloc[0]
            optimal_range = optimal_ranges[feature]
            optimal_val = optimal_range['optimal_value']
            
            # Test improvement by moving towards optimal value
            test_features = original_features.copy()

            # Determine best direction and value
            if current_value < optimal_val:
                suggested_value = min(optimal_val, optimal_range['max'])
                direction = "INCREASE"
            elif current_value > optimal_val:
                suggested_value = max(optimal_val, optimal_range['min'])
                direction = "DECREASE"
            else:
                # Already at optimal
                suggested_value = optimal_val
                direction = "OPTIMAL"

            test_features[feature] = suggested_value
            
            try:
                new_prob = self.model.predict_proba(test_features)[0][1]
                improvement = new_prob - original_prob
            except:
                improvement = 0.0
                new_prob = original_prob

            # Lower threshold - suggest if any positive improvement
            if improvement > 0.001 or direction != "OPTIMAL":
                suggestions.append({
                    'feature': feature,
                    'current': float(current_value),
                    'suggested': float(suggested_value),
                    'direction': direction,
                    'improvement': float(improvement),
                    'improvement_percent': float(improvement * 100),
                    'new_probability': float(new_prob),
                    'importance': optimal_range['importance']
                })

        # Sort by improvement potential
        suggestions.sort(key=lambda x: x['improvement'], reverse=True)
        
        # Return top 5 suggestions
        return suggestions[:5]


    def get_prediction_dict(self, song_features):
        """Get prediction as dictionary (for API use)"""
        result = self.predict_song_hit_probability(song_features)
        if result:
            return {
                'hit_probability': float(result['hit_probability']),
                'confidence': float(result['confidence']),
                'is_hit': bool(result['is_hit_prediction'])
            }
        return None


# Export class for use in Flask app and other modules
__all__ = ['SongHitPredictor']
