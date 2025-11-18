import pandas as pd
import numpy as np
import pickle
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import hashlib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

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
            print("❌ No model to save. Train a model first.")
            return False

        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            joblib.dump(self.model, model_path)

            feature_path = os.path.join(self.model_dir, f"{model_name}_features.pkl")
            joblib.dump(self.feature_names, feature_path)

            print(f"✅ Model saved successfully to {model_path}")
            return True

        except Exception as e:
            print(f"❌ Error saving model: {e}")
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
                print(f"✅ Model loaded! Accuracy: {self.model_metadata.get('accuracy', 0):.2%}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def load_and_prepare_data(self, file_path):
        """Load and prepare the data for training/prediction"""
        print("Loading and cleaning data...")
        try:
            # Use engine='python' for potentially problematic CSV files
            # Use on_bad_lines='skip' to skip problematic rows
            self.df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')

            # Print columns to help identify the correct date column
            print("\nColumns in the loaded DataFrame:")
            print(self.df.columns.tolist())

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
                else:
                    print(f"⚠️ Warning: Musical DNA feature '{feature}' not found in DataFrame.")

            # Drop rows with NaN values in musical DNA features after coercion
            self.df.dropna(subset=self.musical_dna_features, inplace=True)


            self.df['is_hit'] = (self.df[popularity_col] >= 70).astype(int)

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

            print(f"📊 Data loaded: {len(self.df)} songs, {Y.sum()} hits ({Y.mean():.1%} hit rate)")

            return self.df, X, Y

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None, None

    def train_model(self, X, Y, force_retrain=False):
        """Train the model with option to force retrain"""
        data_hash = self._calculate_data_hash(X)

        if not force_retrain and self.load_model():
            print("🚀 Using existing trained model!")
            return True

        print("🔄 Training new model...")
        start_time = datetime.now()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

        self.feature_names = X.columns

        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            enable_categorical=False,
            random_state=42
        )

        self.model.fit(self.X_train, self.Y_train)

        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, predictions)
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"✅ Model training complete. Accuracy: {accuracy*100:.2f}%")

        self._save_model_metadata(accuracy, data_hash, training_time)
        self.save_model()

        return True

    def get_optimal_ranges(self):
        """Get optimal parameter ranges for hit songs"""
        if self.df is None:
            print("❌ No data loaded.")
            return None

        hit_songs = self.df[self.df['is_hit'] == 1]
        non_hit_songs = self.df[self.df['is_hit'] == 0]

        optimal_ranges = {}

        print("\n🎯 === OPTIMAL PARAMETER RANGES FOR HIT SONGS ===")
        print("="*60)

        for feature in self.musical_dna_features:
            hit_mean = hit_songs[feature].mean()
            hit_std = hit_songs[feature].std()
            non_hit_mean = non_hit_songs[feature].mean()

            # Calculate optimal range (mean ± 1 std)
            optimal_min = max(0, hit_mean - hit_std)
            optimal_max = hit_mean + hit_std

            # Calculate statistical significance
            t_stat, p_value = stats.ttest_ind(hit_songs[feature], non_hit_songs[feature])
            significance = "🔥 VERY IMPORTANT" if p_value < 0.001 else "⚡ IMPORTANT" if p_value < 0.05 else "📊 NORMAL"

            optimal_ranges[feature] = {
                'min': optimal_min,
                'max': optimal_max,
                'optimal_value': hit_mean,
                'importance': significance,
                'difference_from_non_hits': hit_mean - non_hit_mean
            }

            print(f"\n🎵 {feature.upper()}")
            print(f"   Optimal Range: {optimal_min:.3f} to {optimal_max:.3f}")
            print(f"   Best Value: {hit_mean:.3f}")
            print(f"   Vs Non-Hits: {hit_mean - non_hit_mean:+.3f} difference")
            print(f"   Status: {significance}")

        return optimal_ranges

    def analyze_best_songs_by_language(self):
        """Analyze best songs from each language with visualizations"""
        if self.df is None:
            print("❌ No data loaded.")
            return None

        print("\n🏆 === BEST SONGS FROM EACH LANGUAGE ===")
        print("="*50)

        # Get hit songs by language
        hit_songs = self.df[self.df['is_hit'] == 1]

        if 'language' not in self.df.columns:
            print("❌ Language column not found in dataset.")
            return None

        languages = hit_songs['language'].value_counts().head(6)  # Top 6 languages

        # Create subplots for each language
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        language_data = {}

        for idx, (lang, count) in enumerate(languages.items()):
            if idx >= 6:
                break

            lang_hits = hit_songs[hit_songs['language'] == lang]
            top_songs = lang_hits.nlargest(5, 'popularity')

            language_data[lang] = {
                'top_songs': top_songs,
                'avg_params': lang_hits[self.musical_dna_features].mean()
            }

            print(f"\n🎼 {lang.upper()} (Top 5 Hits)")
            print("-" * 40)
            for i, (_, song) in enumerate(top_songs.iterrows(), 1):
                print(f"{i}. {song['track_name']} by {song['artist_name']}")
                print(f"   Popularity: {song['popularity']}")
                print(f"   Dance: {song['danceability']:.2f} | Energy: {song['energy']:.2f} | Valence: {song['valence']:.2f}")

            # Plot language characteristics
            ax = axes[idx]
            key_features = ['danceability', 'energy', 'valence', 'acousticness']
            values = [language_data[lang]['avg_params'][f] for f in key_features]

            bars = ax.bar(key_features, values, color=plt.cm.Set3(idx))
            ax.set_title(f'{lang.upper()} - Key Features', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('Musical DNA Analysis by Language', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

        return language_data

    def analyze_consistent_hit_artists(self):
        """Find artists with back-to-back hits and analyze their patterns"""
        if self.df is None:
            print("❌ No data loaded.")
            return None

        print("\n⭐ === ARTISTS WITH CONSISTENT HITS ===")
        print("="*50)

        hit_songs = self.df[self.df['is_hit'] == 1]
        hit_artist_counts = hit_songs['artist_name'].value_counts()
        consistent_artists = hit_artist_counts[hit_artist_counts >= 3].head(8)

        artist_analysis = {}

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, (artist, hit_count) in enumerate(consistent_artists.items()):
            artist_hits = hit_songs[hit_songs['artist_name'] == artist].nlargest(5, 'popularity')

            print(f"\n🎤 {artist.upper()} ({hit_count} hits)")
            print("-" * 40)

            artist_params = {}
            for feature in self.musical_dna_features[:6]:  # Top 6 features for display
                avg_value = artist_hits[feature].mean()
                artist_params[feature] = avg_value

            artist_analysis[artist] = {
                'hit_count': hit_count,
                'top_songs': artist_hits,
                'avg_parameters': artist_params
            }

            # Display top songs
            for i, (_, song) in enumerate(artist_hits.iterrows(), 1):
                print(f"{i}. {song['track_name']} (Pop: {song['popularity']})")

            print(f"\n📊 Average Musical DNA:")
            for param, value in artist_params.items():
                print(f"   {param}: {value:.3f}")

            # Plot artist's musical signature
            ax = axes[idx]
            features = list(artist_params.keys())
            values = list(artist_params.values())

            bars = ax.bar(features, values, color=plt.cm.tab10(idx))
            ax.set_title(f'{artist}\n({hit_count} hits)', fontweight='bold', fontsize=10)
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45, labelsize=8)

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('Musical Signatures of Consistent Hit Artists', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

        return artist_analysis


    def predict_song_hit_probability(self, song_features):
        """Predict the hit probability of a single song"""
        if self.model is None:
            print("❌ No model loaded. Train a model first.")
            return None

        # Ensure the input song features are in the correct format (DataFrame)
        if isinstance(song_features, dict):
            song_df = pd.DataFrame([song_features])
        else:
            song_df = song_features.copy()

        # Ensure the columns are in the same order as the training features
        if not all(feature in song_df.columns for feature in self.feature_names):
            missing_features = [f for f in self.feature_names if f not in song_df.columns]
            print(f"❌ Missing required features for prediction: {missing_features}")
            return None

        song_df = song_df[self.feature_names]

        # Convert features to numeric if they aren't already
        for feature in self.feature_names:
            if not pd.api.types.is_numeric_dtype(song_df[feature]):
                song_df[feature] = pd.to_numeric(song_df[feature], errors='coerce')

        # Handle potential NaN values after conversion (e.g., if input was non-numeric)
        if song_df.isnull().any().any():
            print("❌ Input song features contain non-numeric or missing values after conversion.")
            # Option: Impute missing values or raise an error
            song_df.fillna(song_df.mean(), inplace=True) # Example: Impute with mean (use with caution)
            print("⚠️ Warning: Imputed missing values in input features.")


        try:
            # Get prediction probability
            hit_prob = self.model.predict_proba(song_df)[:, 1][0]

            # Get the predicted class (0 or 1)
            is_hit_prediction = self.model.predict(song_df)[0]

            # Get confidence score (higher probability of the predicted class)
            confidence = self.model.predict_proba(song_df).max(axis=1)[0]


            return {
                'hit_probability': hit_prob,
                'is_hit_prediction': is_hit_prediction,
                'confidence': confidence
            }

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None


    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of the musical data"""
        if self.df is None:
            print("❌ No data loaded.")
            return

        print("\n📊 === COMPREHENSIVE MUSICAL ANALYSIS ===")
        print("="*50)

        # 1. Danceability vs Energy scatter plot
        plt.figure(figsize=(15, 12))

        # Plot 1: Danceability vs Energy
        plt.subplot(2, 3, 1)
        hit_songs = self.df[self.df['is_hit'] == 1]
        non_hit_songs = self.df[self.df['is_hit'] == 0]

        plt.scatter(non_hit_songs['danceability'], non_hit_songs['energy'],
                   alpha=0.5, c='lightcoral', label='Non-Hits', s=20)
        plt.scatter(hit_songs['danceability'], hit_songs['energy'],
                   alpha=0.7, c='gold', label='Hits', s=30)
        plt.xlabel('Danceability')
        plt.ylabel('Energy')
        plt.title('Danceability vs Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Valence vs Acousticness
        plt.subplot(2, 3, 2)
        plt.scatter(non_hit_songs['valence'], non_hit_songs['acousticness'],
                   alpha=0.5, c='lightblue', label='Non-Hits', s=20)
        plt.scatter(hit_songs['valence'], hit_songs['acousticness'],
                   alpha=0.7, c='orange', label='Hits', s=30)
        plt.xlabel('Valence (Positivity)')
        plt.ylabel('Acousticness')
        plt.title('Valence vs Acousticness')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Tempo vs Loudness
        plt.subplot(2, 3, 3)
        plt.scatter(non_hit_songs['tempo'], non_hit_songs['loudness'],
                   alpha=0.5, c='lightgreen', label='Non-Hits', s=20)
        plt.scatter(hit_songs['tempo'], hit_songs['loudness'],
                   alpha=0.7, c='red', label='Hits', s=30)
        plt.xlabel('Tempo (BPM)')
        plt.ylabel('Loudness (dB)')
        plt.title('Tempo vs Loudness')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Hit rate by feature ranges
        plt.subplot(2, 3, 4)
        danceability_bins = pd.cut(self.df['danceability'], bins=5)
        hit_rate_by_dance = self.df.groupby(danceability_bins)['is_hit'].mean()
        hit_rate_by_dance.plot(kind='bar', color='skyblue')
        plt.title('Hit Rate by Danceability Range')
        plt.ylabel('Hit Rate')
        plt.xticks(rotation=45)

        # Plot 5: Feature correlation with hits
        plt.subplot(2, 3, 5)
        correlations = self.df[self.musical_dna_features + ['is_hit']].corr()['is_hit'][:-1]
        correlations.plot(kind='barh', color='purple', alpha=0.7)
        plt.title('Feature Correlation with Hits')
        plt.xlabel('Correlation with Hit Status')

        # Plot 6: Distribution of key musical features
        plt.subplot(2, 3, 6)
        key_features = ['danceability', 'energy', 'valence']
        for i, feature in enumerate(key_features):
            plt.hist(hit_songs[feature], alpha=0.5, label=f'Hits - {feature}', bins=20)
        plt.title('Distribution of Key Features (Hits Only)')
        plt.legend()
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def suggest_feature_improvements(self, song_features):
        """Suggest which features to change to make a song more likely to be a hit"""
        if self.model is None:
            print("❌ No model loaded.")
            return None

        print("\n🎯 === SONG IMPROVEMENT SUGGESTIONS ===")
        print("="*50)

        # Convert to DataFrame if dict
        if isinstance(song_features, dict):
            original_features = pd.DataFrame([song_features])
        else:
            original_features = song_features.copy()

        # Get original prediction
        original_prob = self.model.predict_proba(original_features[self.feature_names])[0][1]

        print(f"🎵 Original Hit Probability: {original_prob:.1%}")

        if original_prob >= 0.7:
            print("🎉 This song is already predicted to be a HIT! 🎉")
            return original_prob

        # Get optimal ranges
        optimal_ranges = self.get_optimal_ranges()
        suggestions = []

        print(f"\n💡 IMPROVEMENT SUGGESTIONS:")
        print("-" * 30)

        for feature in self.feature_names:
            current_value = original_features[feature].iloc[0]
            optimal_range = optimal_ranges[feature]

            # Test improvement by moving towards optimal value
            test_features = original_features.copy()

            # Move towards optimal value
            if current_value < optimal_range['min']:
                suggested_value = optimal_range['min']
                direction = "INCREASE"
            elif current_value > optimal_range['max']:
                suggested_value = optimal_range['max']
                direction = "DECREASE"
            else:
                suggested_value = optimal_range['optimal_value']
                direction = "FINE-TUNE"

            test_features[feature] = suggested_value
            new_prob = self.model.predict_proba(test_features[self.feature_names])[0][1]
            improvement = new_prob - original_prob

            if improvement > 0.01:  # Only suggest if improvement > 1%
                suggestions.append({
                    'feature': feature,
                    'current': current_value,
                    'suggested': suggested_value,
                    'direction': direction,
                    'improvement': improvement,
                    'new_probability': new_prob
                })

        # Sort by improvement potential
        suggestions.sort(key=lambda x: x['improvement'], reverse=True)

        for i, suggestion in enumerate(suggestions[:5], 1):  # Top 5 suggestions
            print(f"\n{i}. {suggestion['feature'].upper()}")
            print(f"   Current: {suggestion['current']:.3f}")
            print(f"   Suggested: {suggestion['suggested']:.3f}")
            print(f"   Action: {suggestion['direction']} by {abs(suggestion['suggested'] - suggestion['current']):.3f}")
            print(f"   Expected improvement: +{suggestion['improvement']:.1%}")
            print(f"   New hit probability: {suggestion['new_probability']:.1%}")

        # Test applying all top 3 suggestions
        if len(suggestions) >= 3:
            combined_features = original_features.copy()
            for suggestion in suggestions[:3]:
                combined_features[suggestion['feature']] = suggestion['suggested']

            combined_prob = self.model.predict_proba(combined_features[self.feature_names])[0][1]
            total_improvement = combined_prob - original_prob

            print(f"\n🚀 COMBINED EFFECT (Top 3 changes):")
            print(f"   New hit probability: {combined_prob:.1%}")
            print(f"   Total improvement: +{total_improvement:.1%}")

            if combined_prob >= 0.7:
                print("🎉 With these changes, your song could be a HIT! 🎉")

        return suggestions

    def comprehensive_song_analysis(self, song_features=None):
        """Perform comprehensive analysis including all insights"""
        print("\n🎼 === COMPREHENSIVE SONG HIT ANALYSIS ===")
        print("="*60)

        # 1. Get optimal ranges
        optimal_ranges = self.get_optimal_ranges()

        # 2. Analyze best songs by language
        language_analysis = self.analyze_best_songs_by_language()

        # 3. Analyze consistent hit artists
        artist_analysis = self.analyze_consistent_hit_artists()

        # 4. Create visualizations
        self.create_comprehensive_visualizations()

        # 5. Feature importance
        if self.model:
            self.analyze_feature_importance()

        # 6. Song improvement suggestions (if song provided)
        if song_features:
            suggestions = self.suggest_feature_improvements(song_features)
            return suggestions

        return {
            'optimal_ranges': optimal_ranges,
            'language_analysis': language_analysis,
            'artist_analysis': artist_analysis
        }

    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        if self.model is None:
            print("❌ No model loaded.")
            return None

        feature_importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        print("\n🔍 === FEATURE IMPORTANCE ANALYSIS ===")
        print("="*50)
        print("Most Important Features for Hit Prediction:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:15s} - {row['importance']:.3f}")

        # Visualize
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(10)
        bars = plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Feature")
        plt.title("Top 10 Most Important Musical DNA Features for Hit Prediction")
        plt.gca().invert_yaxis()

        # Add value labels
        for bar, importance in zip(bars, top_features['importance']):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

        return importance_df

    def get_user_input_and_predict(self):
        """Interactive function to get user input and predict hit probability"""
        print("\n🎤 === SONG HIT PREDICTION - USER INPUT MODE ===")
        print("="*60)
        print("Enter your song's musical DNA parameters:")
        print("(Press Enter to use suggested values for hit songs)")
        print("-" * 60)

        # Get optimal ranges for suggestions
        if self.df is not None:
            optimal_ranges = self.get_optimal_ranges()
        else:
            optimal_ranges = None

        user_song = {}

        # Define feature descriptions and typical ranges
        feature_info = {
            'danceability': {
                'description': 'How suitable a track is for dancing (0.0 to 1.0)',
                'example': 'Pop songs: ~0.6-0.8, Ballads: ~0.3-0.5',
                'optimal': 0.65
            },
            'energy': {
                'description': 'Perceptual measure of intensity (0.0 to 1.0)',
                'example': 'Rock/EDM: ~0.7-0.9, Acoustic: ~0.2-0.5',
                'optimal': 0.72
            },
            'key': {
                'description': 'The key the track is in (0-11, where 0=C, 1=C#, etc.)',
                'example': 'C=0, D=2, E=4, F=5, G=7, A=9, B=11',
                'optimal': 5
            },
            'loudness': {
                'description': 'Overall loudness in decibels (-60 to 0)',
                'example': 'Loud songs: -3 to -8 dB, Quiet: -15 to -25 dB',
                'optimal': -6.5
            },
            'mode': {
                'description': 'Major (1) or Minor (0) key',
                'example': 'Major=1 (happy), Minor=0 (sad)',
                'optimal': 1
            },
            'speechiness': {
                'description': 'Presence of spoken words (0.0 to 1.0)',
                'example': 'Songs: ~0.03-0.1, Rap: ~0.3-0.9',
                'optimal': 0.08
            },
            'acousticness': {
                'description': 'Whether the track is acoustic (0.0 to 1.0)',
                'example': 'Electric: ~0.0-0.2, Acoustic: ~0.7-1.0',
                'optimal': 0.25
            },
            'instrumentalness': {
                'description': 'Whether track contains no vocals (0.0 to 1.0)',
                'example': 'Vocal songs: ~0.0-0.1, Instrumental: ~0.5-1.0',
                'optimal': 0.05
            },
            'liveness': {
                'description': 'Presence of live audience (0.0 to 1.0)',
                'example': 'Studio: ~0.05-0.2, Live: ~0.8-1.0',
                'optimal': 0.15
            },
            'valence': {
                'description': 'Musical positivity/happiness (0.0 to 1.0)',
                'example': 'Happy: ~0.6-1.0, Sad: ~0.0-0.4',
                'optimal': 0.58
            },
            'tempo': {
                'description': 'Beats per minute (BPM)',
                'example': 'Ballad: ~60-80, Pop: ~120-130, Dance: ~128-140',
                'optimal': 125
            },
            'duration_ms': {
                'description': 'Track duration in milliseconds',
                'example': '3 min = 180000, 4 min = 240000',
                'optimal': 210000
            }
        }

        for feature in self.musical_dna_features:
            info = feature_info[feature]

            print(f"\n🎵 {feature.upper().replace('_', ' ')}")
            print(f"   Description: {info['description']}")
            print(f"   Examples: {info['example']}")

            if optimal_ranges and feature in optimal_ranges:
                optimal_val = optimal_ranges[feature]['optimal_value']
                print(f"   💡 Optimal for hits: {optimal_val:.3f}")
            else:
                print(f"   💡 Suggested: {info['optimal']}")

            while True:
                try:
                    user_input = input(f"   Enter {feature} value (or press Enter for optimal): ").strip()

                    if user_input == "":
                        # Use optimal value
                        if optimal_ranges and feature in optimal_ranges:
                            value = optimal_ranges[feature]['optimal_value']
                        else:
                            value = info['optimal']
                        print(f"   ✅ Using optimal value: {value}")
                        user_song[feature] = value
                        break
                    else:
                        value = float(user_input)

                        # Validate ranges
                        if feature in ['danceability', 'energy', 'speechiness', 'acousticness',
                                     'instrumentalness', 'liveness', 'valence']:
                            if not 0 <= value <= 1:
                                print("   ❌ Value must be between 0.0 and 1.0")
                                continue
                        elif feature == 'key':
                            if not 0 <= value <= 11:
                                print("   ❌ Key must be between 0 and 11")
                                continue
                        elif feature == 'mode':
                            if value not in [0, 1]:
                                print("   ❌ Mode must be 0 (minor) or 1 (major)")
                                continue
                        elif feature == 'loudness':
                            if not -60 <= value <= 0:
                                print("   ❌ Loudness must be between -60 and 0 dB")
                                continue
                        elif feature == 'tempo':
                            if not 30 <= value <= 250:
                                print("   ❌ Tempo must be between 30 and 250 BPM")
                                continue
                        elif feature == 'duration_ms':
                            if not 30000 <= value <= 600000:  # 30 sec to 10 min
                                print("   ❌ Duration must be between 30,000 and 600,000 ms")
                                continue

                        user_song[feature] = value
                        break

                except ValueError:
                    print("   ❌ Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n\n👋 Prediction cancelled by user.")
                    return None

        # Make prediction
        print(f"\n🔍 === ANALYZING YOUR SONG ===")
        print("="*40)

        # Display entered values
        print("Your song's musical DNA:")
        for feature, value in user_song.items():
            print(f"  {feature:15s}: {value}")

        # Get prediction
        prediction_result = self.predict_song_hit_probability(user_song)

        if prediction_result:
            hit_prob = prediction_result['hit_probability']
            is_hit = prediction_result['is_hit_prediction']
            confidence = prediction_result['confidence']

            print(f"\n🎯 === PREDICTION RESULTS ===")
            print("="*35)
            print(f"Hit Probability: {hit_prob:.1%}")
            print(f"Prediction: {'🎉 HIT! 🎉' if is_hit else '❌ Not a Hit'}")
            print(f"Confidence: {confidence}")

            # Add interpretation
            if hit_prob >= 0.8:
                print(f"\n🚀 AMAZING! Your song has excellent hit potential!")
            elif hit_prob >= 0.6:
                print(f"\n✨ GOOD! Your song has solid hit potential!")
            elif hit_prob >= 0.4:
                print(f"\n📊 MODERATE. Your song has some hit potential.")
            else:
                print(f"\n💡 LOW. But don't worry - let's see how to improve it!")

            # Get improvement suggestions
            if hit_prob < 0.7:
                print(f"\n🔧 Want to improve your song's hit potential?")
                improve = input("Get improvement suggestions? (y/n): ").lower().strip()
                if improve in ['y', 'yes']:
                    suggestions = self.suggest_feature_improvements(user_song)

        return user_song, prediction_result

    def interactive_prediction_mode(self):
        """Run interactive prediction mode with options"""
        if self.model is None:
            print("❌ No model loaded. Please train or load a model first.")
            return

        while True:
            print(f"\n🎼 === SONG HIT PREDICTOR - INTERACTIVE MODE ===")
            print("="*55)
            print("1. 🎤 Enter new song data and get prediction")
            print("2. 📊 View optimal parameter ranges")
            print("3. 🏆 View best songs by language")
            print("4. ⭐ View consistent hit artists")
            print("5. 📈 View comprehensive analysis")
            print("6. Exit")
            print("="*55)

            try:
                choice = input("Choose an option (1-6): ").strip()

                if choice == '1':
                    result = self.get_user_input_and_predict()
                    if result:
                        input("\nPress Enter to continue...")

                elif choice == '2':
                    self.get_optimal_ranges()
                    input("\nPress Enter to continue...")

                elif choice == '3':
                    self.analyze_best_songs_by_language()
                    input("\nPress Enter to continue...")

                elif choice == '4':
                    self.analyze_consistent_hit_artists()
                    input("\nPress Enter to continue...")

                elif choice == '5':
                    self.comprehensive_song_analysis()
                    input("\nPress Enter to continue...")

                elif choice == '6':
                    print("👋 Thanks for using Song Hit Predictor!")
                    break

                else:
                    print("❌ Invalid choice. Please enter 1-6.")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}")
                input("Press Enter to continue...")

# Usage Example
def main():
    # Initialize the predictor
    predictor = SongHitPredictor()

    # File path
    file_path = r"demo.csv"  #<-- Enter your file path

    print("🎵 Welcome to the Song Hit Predictor! 🎵")
    print("="*50)

    # Load and prepare data
    print("Loading dataset...")
    df, X, Y = predictor.load_and_prepare_data(file_path)

    if df is not None:
        # Print columns after loading data
        print("\nDataset Columns:")
        print(df.columns.tolist())


        # Train model
        print("Training/Loading model...")
        predictor.train_model(X, Y, force_retrain=False)

        # Ask user what they want to do
        print(f"\n🚀 Model ready! What would you like to do?")
        print("1. 🎤 Predict hit potential for your song (Interactive)")
        print("2. 📊 Run comprehensive analysis of the dataset")
        print("3. Interactive mode (multiple predictions)")

        try:
            choice = input("\nChoose an option (1-3): ").strip()

            if choice == '1':
                # Single prediction
                result = predictor.get_user_input_and_predict()

            elif choice == '2':
                # Comprehensive analysis
                print("\n🚀 Running comprehensive analysis...")
                analysis_results = predictor.comprehensive_song_analysis()
                print("\n✅ Analysis complete! Check the visualizations above.")

            elif choice == '3':
                # Interactive mode
                predictor.interactive_prediction_mode()

            else:
                print("Running default comprehensive analysis...")
                analysis_results = predictor.comprehensive_song_analysis()

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Song Hit Predictor!")
        except Exception as e:
            print(f"❌ An error occurred: {e}")

    else:
        print("❌ Failed to load data. Please check your file path.")

if __name__ == "__main__":
    main()
