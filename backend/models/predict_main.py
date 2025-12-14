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
from sklearn.metrics import accuracy_score, classification_report
import json
import hashlib
from scipy import stats
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SongHitPredictor:
    def __init__(self, model_dir="models", data_dir="data"):
        """
        Initialize the Song Hit Predictor with model persistence capabilities
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.feature_names = None
        self.model_metadata = {}
        self.df = None

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
        self.model_metadata = {
            'model_type': 'XGBClassifier',
            'accuracy': accuracy,
            'training_time': training_time,
            'data_hash': data_hash,
            'feature_names': self.feature_names.tolist(),
            'created_at': datetime.now().isoformat(),
            'data_size': len(self.X_train) + len(self.X_test)
        }

        metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

    def save_model(self, model_name="song_hit_model"):
        """Save the trained model, scaler, and metadata"""
        if self.model is None:
            return False

        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            joblib.dump(self.model, model_path)

            feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
            joblib.dump(self.feature_names, feature_path)

            return True

        except Exception as e:
            return False

    def load_model(self, model_name="song_hit_model"):
        """Load a previously trained model"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
            metadata_path = os.path.join(self.model_dir, 'model_metadata.json')

            if not os.path.exists(model_path):
                return False

            self.model = joblib.load(model_path)
            self.feature_names = joblib.load(feature_path)

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)

            return True

        except Exception as e:
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

            # Use the 'year' column directly for release year
            if 'year' not in self.df.columns:
                 raise KeyError("The 'year' column was not found in the dataset.")

            self.df['release_year'] = self.df['year']

            # Drop rows where 'release_year' is NaT and other irrelevant columns
            self.df.dropna(subset=['release_year'], inplace=True)

            # Drop ID columns and the original 'year' column (if we want to use 'release_year')
            columns_to_drop = ['year']
            if 'track_id' in self.df.columns:
                columns_to_drop.append('track_id')
            if 'album_id' in self.df.columns:
                columns_to_drop.append('album_id')
            if 'track_album_id' in self.df.columns and 'album_id' in self.df.columns:
                columns_to_drop.append('track_album_id')  # Keep only one
            if 'playlist_id' in self.df.columns:
                columns_to_drop.append('playlist_id')
            if 'artwork_url' in self.df.columns:
                columns_to_drop.append('artwork_url')
            if 'track_url' in self.df.columns:
                columns_to_drop.append('track_url')

            # Only drop columns that actually exist
            columns_to_drop = [col for col in columns_to_drop if col in self.df.columns]
            self.df = self.df.drop(columns_to_drop, axis=1)

            # Define a 'hit' as any song with a popularity score of 70 or higher
            # Check for 'popularity' as an alternative name for 'track_popularity'
            popularity_col = 'track_popularity' if 'track_popularity' in self.df.columns else 'popularity'
            if popularity_col not in self.df.columns:
                 raise KeyError("Neither 'track_popularity' nor 'popularity' column found for target variable.")

            # Convert the popularity column to numeric, coercing errors to NaN
            self.df[popularity_col] = pd.to_numeric(self.df[popularity_col], errors='coerce')
            # Drop rows where popularity is NaN after coercion
            self.df.dropna(subset=[popularity_col], inplace=True)

            # Convert musical DNA features to numeric, coercing errors
            for feature in self.musical_dna_features:
                if feature in self.df.columns:
                    self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')

            # Drop rows with NaN values in musical DNA features after coercion
            self.df.dropna(subset=self.musical_dna_features, inplace=True)

            # Define hit as popularity >= 50 (more balanced than 70)
            # This creates ~15-20% hit ratio which is more realistic and reduces bias
            self.df['is_hit'] = (self.df[popularity_col] >= 50).astype(int)

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

        # Calculate class weights to handle severe class imbalance (1% hits vs 99% non-hits)
        # This penalizes wrong predictions on minority class (hits)
        n_samples = len(self.Y_train)
        n_hits = (self.Y_train == 1).sum()
        n_non_hits = (self.Y_train == 0).sum()
        
        # Weight for hits: penalize missing them more heavily
        weight_hits = n_samples / (2 * n_hits)
        weight_non_hits = n_samples / (2 * n_non_hits)
        
        # Create sample weights
        sample_weights = np.where(self.Y_train == 1, weight_hits, weight_non_hits)

        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            enable_categorical=False,
            random_state=42,
            scale_pos_weight=weight_hits / weight_non_hits  # Also use scale_pos_weight parameter
        )

        self.model.fit(self.X_train, self.Y_train, sample_weight=sample_weights)

        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, predictions)
        training_time = (datetime.now() - start_time).total_seconds()

        self._save_model_metadata(accuracy, data_hash, training_time)
        self.save_model()

        return True

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
        if self.model is None:
            return None

        # Ensure the input song features are in the correct format (DataFrame)
        if isinstance(song_features, dict):
            song_df = pd.DataFrame([song_features])
        else:
            song_df = song_features.copy()

        # Ensure the columns are in the same order as the training features
        if not all(feature in song_df.columns for feature in self.feature_names):
            return None

        song_df = song_df[self.feature_names]

        # Convert features to numeric if they aren't already
        for feature in self.feature_names:
            if not pd.api.types.is_numeric_dtype(song_df[feature]):
                song_df[feature] = pd.to_numeric(song_df[feature], errors='coerce')

        # Validate and normalize feature ranges
        # Clamp values to realistic ranges to prevent model errors
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
            
            # Get prediction probability
            hit_prob = self.model.predict_proba(song_df)[:, 1][0]

            # Get the predicted class (0 or 1)
            is_hit_prediction = self.model.predict(song_df)[0]

            # Get confidence score (higher probability of the predicted class)
            confidence = self.model.predict_proba(song_df).max(axis=1)[0]

            return {
                'hit_probability': float(hit_prob),
                'is_hit_prediction': bool(is_hit_prediction),
                'confidence': float(confidence)
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None


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
