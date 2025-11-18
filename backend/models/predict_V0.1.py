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
            self.df = pd.read_csv(file_path)

            # Convert the 'track_album_release_date' column to a numerical year
            self.df['release_year'] = pd.to_datetime(self.df['track_album_release_date'], errors='coerce').dt.year

            # Drop rows where 'release_year' is NaT and other irrelevant columns
            self.df.dropna(subset=['release_year'], inplace=True)
            self.df = self.df.drop(['track_id', 'track_album_id', 'playlist_id', 'track_album_release_date'], axis=1)

            # Define a 'hit' as any song with a popularity score of 70 or higher
            self.df['is_hit'] = (self.df['track_popularity'] >= 70).astype(int)

            # Check if all musical features exist
            if not all(feature in self.df.columns for feature in self.musical_dna_features):
                missing_features = [f for f in self.musical_dna_features if f not in self.df.columns]
                raise KeyError(f"Missing musical DNA features: {missing_features}")

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

    def analyze_best_songs_by_genre(self):
        """Analyze best songs from each genre with visualizations"""
        if self.df is None:
            print("❌ No data loaded.")
            return None

        print("\n🏆 === BEST SONGS FROM EACH GENRE ===")
        print("="*50)

        # Get hit songs by genre
        hit_songs = self.df[self.df['is_hit'] == 1]

        if 'playlist_genre' not in self.df.columns:
            print("❌ Genre column not found in dataset.")
            return None

        genres = hit_songs['playlist_genre'].value_counts().head(6)  # Top 6 genres

        # Create subplots for each genre
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        genre_data = {}

        for idx, (genre, count) in enumerate(genres.items()):
            if idx >= 6:
                break

            genre_hits = hit_songs[hit_songs['playlist_genre'] == genre]
            top_songs = genre_hits.nlargest(5, 'track_popularity')

            genre_data[genre] = {
                'top_songs': top_songs,
                'avg_params': genre_hits[self.musical_dna_features].mean()
            }

            print(f"\n🎼 {genre.upper()} (Top 5 Hits)")
            print("-" * 40)
            for i, (_, song) in enumerate(top_songs.iterrows(), 1):
                print(f"{i}. {song['track_name']} by {song['track_artist']}")
                print(f"   Popularity: {song['track_popularity']}")
                print(f"   Dance: {song['danceability']:.2f} | Energy: {song['energy']:.2f} | Valence: {song['valence']:.2f}")

            # Plot genre characteristics
            ax = axes[idx]
            key_features = ['danceability', 'energy', 'valence', 'acousticness']
            values = [genre_data[genre]['avg_params'][f] for f in key_features]

            bars = ax.bar(key_features, values, color=plt.cm.Set3(idx))
            ax.set_title(f'{genre.title()} - Key Features', fontweight='bold')
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.suptitle('Musical DNA Analysis by Genre', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

        return genre_data

    def analyze_consistent_hit_artists(self):
        """Find artists with back-to-back hits and analyze their patterns"""
        if self.df is None:
            print("❌ No data loaded.")
            return None

        print("\n⭐ === ARTISTS WITH CONSISTENT HITS ===")
        print("="*50)

        hit_songs = self.df[self.df['is_hit'] == 1]
        hit_artist_counts = hit_songs['track_artist'].value_counts()
        consistent_artists = hit_artist_counts[hit_artist_counts >= 3].head(8)

        artist_analysis = {}

        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, (artist, hit_count) in enumerate(consistent_artists.items()):
            artist_hits = hit_songs[hit_songs['track_artist'] == artist].nlargest(5, 'track_popularity')

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
                print(f"{i}. {song['track_name']} (Pop: {song['track_popularity']})")

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
                       f'{value:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.suptitle('Musical Signatures of Consistent Hit Artists', fontsize=16, fontweight='bold', y=1.02)
        plt.show()

        return artist_analysis

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

        # 2. Analyze best songs by genre
        genre_analysis = self.analyze_best_songs_by_genre()

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
            'genre_analysis': genre_analysis,
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

# Usage Example
def main():
    # Initialize the predictor
    predictor = SongHitPredictor()

    # File path
    file_path = r"spotify_songs.csv"

    # Load and prepare data
    df, X, Y = predictor.load_and_prepare_data(file_path)

    if df is not None:
        # Train model
        predictor.train_model(X, Y, force_retrain=False)

        # Perform comprehensive analysis
        print("\n🚀 Starting Comprehensive Analysis...")

        # Example song for improvement suggestions
        example_song = {
            'danceability': 0.5,
            'energy': 0.6,
            'key': 5,
            'loudness': -8.0,
            'mode': 1,
            'speechiness': 0.1,
            'acousticness': 0.4,
            'instrumentalness': 0.01,
            'liveness': 0.15,
            'valence': 0.4,
            'tempo': 110.0,
            'duration_ms': 220000
        }

        # Run comprehensive analysis
        analysis_results = predictor.comprehensive_song_analysis(example_song)

        print("\n✅ Analysis Complete! Check the visualizations and suggestions above.")

if __name__ == "__main__":
    main()
