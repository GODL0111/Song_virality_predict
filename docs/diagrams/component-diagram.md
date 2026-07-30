# Component Diagram

This diagram shows all the components of the Song Virality Prediction System and their relationships.

## Diagram

```mermaid
classDiagram
    %% Frontend Components
    class App {
        +state score
        +state logs
        +useState()
        +useEffect()
        +handleResult()
        +resetGame()
        -loadFromLocalStorage()
        -saveToLocalStorage()
    }
    
    class Layout {
        +state currentPage
        +state menuOpen
        +state isDarkTheme
        +navigate()
        +toggleTheme()
        +toggleMenu()
        -renderPage()
    }
    
    class HomePage {
        +renderForm()
        +handleSubmit()
        +validateInput()
        +displayResult()
    }
    
    class LiveSongTest {
        +state selectedFile
        +handleFileUpload()
        +analyzeAudio()
        +displayExtractedFeatures()
        +displayResult()
    }
    
    class PredictorForm {
        +state features
        +handleInputChange()
        +handleSubmit()
        +validateFeatures()
        +resetForm()
    }
    
    class Creators {
        +displayTeamInfo()
        +renderMemberCards()
    }
    
    class GameDashboard {
        +props score
        +props logs
        +renderScoreCard()
        +renderHistory()
        +calculateStats()
    }
    
    class Recommendations {
        +props suggestions
        +renderSuggestions()
        +formatImprovement()
    }
    
    %% Backend Components
    class FlaskApp {
        +route("/")
        +route("/api/health")
        +route("/api/predict")
        +route("/api/analyze-audio")
        +route("/api/model-info")
        +route("/api/optimal-ranges")
        +route("/api/feature-importance")
        +route("/api/suggest-improvements")
        -load_model_globally()
        -extract_audio_features()
    }
    
    class SongHitPredictor {
        -model XGBClassifier
        -feature_names array
        -model_metadata dict
        -df DataFrame
        +load_and_prepare_data()
        +train_model()
        +predict_song_hit_probability()
        +get_optimal_ranges()
        +get_feature_importance()
        +suggest_feature_improvements()
        +save_model()
        +load_model()
        -_calculate_data_hash()
        -_save_model_metadata()
    }
    
    class LibrosaExtractor {
        <<service>>
        +load_audio()
        +extract_tempo()
        +extract_energy()
        +extract_spectral_features()
        +extract_chroma_features()
        +extract_zcr_features()
    }
    
    %% ML Components
    class XGBoostClassifier {
        <<model>>
        +fit(X, y, sample_weight)
        +predict(X)
        +predict_proba(X)
        +feature_importances_
        -_check_params()
    }
    
    class ModelPersistence {
        <<service>>
        +save_model()
        +load_model()
        +save_metadata()
        +load_metadata()
        -joblib_dump()
        -joblib_load()
    }
    
    class FeatureEngineering {
        <<service>>
        +normalize_features()
        +clamp_values()
        +validate_ranges()
        +convert_to_dataframe()
    }
    
    %% Data Components
    class ModelFiles {
        <<storage>>
        +song_hit_model.pkl
        +song_hit_model_features.pkl
        +model_metadata.json
    }
    
    class DatasetFiles {
        <<storage>>
        +spotify_tracks.csv
        +spotify_songs.csv
    }
    
    class TempStorage {
        <<storage>>
        +uploaded_audio_files
        +temp_processing_files
    }
    
    %% Relationships - Frontend
    App --> Layout : contains
    Layout --> HomePage : routes to
    Layout --> LiveSongTest : routes to
    Layout --> Creators : routes to
    Layout --> GameDashboard : contains
    HomePage --> PredictorForm : contains
    HomePage --> Recommendations : contains
    LiveSongTest --> Recommendations : contains
    
    %% Relationships - Frontend to Backend
    PredictorForm ..> FlaskApp : POST /api/predict
    LiveSongTest ..> FlaskApp : POST /api/analyze-audio
    HomePage ..> FlaskApp : GET /api/optimal-ranges
    HomePage ..> FlaskApp : POST /api/suggest-improvements
    HomePage ..> FlaskApp : GET /api/feature-importance
    
    %% Relationships - Backend
    FlaskApp --> SongHitPredictor : uses
    FlaskApp --> LibrosaExtractor : uses for audio
    SongHitPredictor --> XGBoostClassifier : contains
    SongHitPredictor --> FeatureEngineering : uses
    SongHitPredictor --> ModelPersistence : uses
    
    %% Relationships - ML to Storage
    ModelPersistence ..> ModelFiles : reads/writes
    SongHitPredictor ..> DatasetFiles : reads training data
    LibrosaExtractor ..> TempStorage : reads audio files
    FlaskApp ..> TempStorage : writes uploads
    
    %% Styling
    class App,Layout,HomePage,LiveSongTest,PredictorForm,Creators,GameDashboard,Recommendations frontend
    class FlaskApp,SongHitPredictor,LibrosaExtractor backend
    class XGBoostClassifier,ModelPersistence,FeatureEngineering ml
    class ModelFiles,DatasetFiles,TempStorage storage
    
    classDef frontend fill:#61dafb,stroke:#333,stroke-width:2px,color:#000
    classDef backend fill:#3c873a,stroke:#333,stroke-width:2px,color:#fff
    classDef ml fill:#ff6b35,stroke:#333,stroke-width:2px,color:#fff
    classDef storage fill:#ffd700,stroke:#333,stroke-width:2px,color:#000
```

## Component Categories

### 🎨 Frontend Components (React)

#### App.jsx
**Responsibility**: Root component and state manager  
**State**:
- `score`: User's total gamification score
- `logs`: Array of prediction history
**Methods**:
- `handleResult()`: Process prediction results, update score
- `resetGame()`: Clear score and logs
- Storage management with localStorage

#### Layout.jsx
**Responsibility**: Navigation, routing, and theme management  
**Features**:
- Page routing (home, live test, creators)
- Dark/light theme toggle
- Responsive mobile menu
- Background animations

#### HomePage
**Responsibility**: Main prediction interface with manual input  
**Contains**: PredictorForm, Recommendations  
**Features**: Form validation, result display, improvement suggestions

#### LiveSongTest
**Responsibility**: Audio file upload and analysis  
**Features**:
- File selection and upload
- Audio feature extraction display
- Automatic prediction
- Result visualization

#### PredictorForm
**Responsibility**: 12-feature input form  
**State**: All 12 musical DNA features  
**Validation**: Range checking, type validation

#### Creators
**Responsibility**: Team information page  
**Content**: Team member profiles, project info

#### GameDashboard
**Responsibility**: Score and prediction history display  
**Props**: score, logs  
**Features**: Statistics calculation, history timeline

#### Recommendations
**Responsibility**: Display improvement suggestions  
**Props**: suggestions array  
**Features**: Feature change visualization, improvement percentages

### 🔧 Backend Components (Flask/Python)

#### FlaskApp (app.py)
**Responsibility**: REST API server  
**Endpoints**:
- `/api/predict`: Manual prediction
- `/api/analyze-audio`: Audio file analysis
- `/api/health`: Health check
- `/api/model-info`: Model metadata
- `/api/optimal-ranges`: Feature ranges
- `/api/feature-importance`: Importance scores
- `/api/suggest-improvements`: Improvement suggestions

**Dependencies**: Flask, Flask-CORS, SongHitPredictor, LibrosaExtractor

#### SongHitPredictor (predict_main.py)
**Responsibility**: Core ML model wrapper  
**Key Methods**:
- `load_and_prepare_data()`: CSV loading and preprocessing
- `train_model()`: XGBoost training with class weights
- `predict_song_hit_probability()`: Single song prediction
- `get_optimal_ranges()`: Statistical analysis of hit songs
- `get_feature_importance()`: Feature importance from model
- `suggest_feature_improvements()`: Optimization suggestions

**State**:
- `model`: XGBoost classifier instance
- `feature_names`: List of 12 features
- `model_metadata`: Training info
- `df`: Loaded dataset

#### LibrosaExtractor
**Responsibility**: Audio feature extraction  
**Methods**:
- `load_audio()`: Load audio file with librosa
- `extract_tempo()`: Beat tracking for tempo
- `extract_energy()`: RMS energy calculation
- `extract_spectral_features()`: Danceability, valence, loudness
- `extract_chroma_features()`: Key detection
- `extract_zcr_features()`: Speechiness, liveness

### 🤖 ML Components

#### XGBoostClassifier
**Responsibility**: Binary classification model  
**Configuration**:
- `use_label_encoder=False`
- `eval_metric='logloss'`
- `scale_pos_weight`: For class imbalance
- `random_state=42`: Reproducibility

**Key Methods**:
- `fit()`: Train with sample weights
- `predict()`: Binary classification (0/1)
- `predict_proba()`: Probability scores (0-1)

#### ModelPersistence
**Responsibility**: Save/load models and metadata  
**Files**:
- Model: joblib format (*.pkl)
- Features: joblib format (*.pkl)
- Metadata: JSON format (*.json)

#### FeatureEngineering
**Responsibility**: Feature preprocessing and validation  
**Operations**:
- Normalize features to valid ranges
- Clamp extreme values
- Convert dict to DataFrame
- Validate data types

### 💾 Storage Components

#### ModelFiles
**Location**: `backend/models/`  
**Contents**:
- `song_hit_model.pkl`: Trained XGBoost model
- `song_hit_model_features.pkl`: Feature names array
- `model_metadata.json`: Training metadata

#### DatasetFiles
**Location**: `datasets/`  
**Contents**:
- `spotify_tracks.csv`: Main dataset (~114K songs)
- `spotify_songs.csv`: Alternative dataset

#### TempStorage
**Location**: `data/` or system temp  
**Contents**: Uploaded audio files (deleted after processing)

## Communication Patterns

### Frontend → Backend
- **Protocol**: HTTP/REST
- **Format**: JSON (predictions), multipart/form-data (audio)
- **CORS**: Enabled for cross-origin requests

### Backend → ML
- **Method**: Direct Python function calls
- **Data**: Pandas DataFrame or dict
- **Returns**: Dict with prediction results

### ML → Storage
- **Read**: Training data from CSV
- **Write**: Model files with joblib
- **Format**: Pickle (models), JSON (metadata)

## Deployment Units

1. **Frontend**: Deployed to Vercel from `frontend/` directory
2. **Backend**: Deployed to Vercel from `backend/` directory
3. **Models**: Included in backend deployment
4. **Datasets**: Included in backend deployment (for training)

## Technology Stack by Layer

**Frontend**:
- React 18
- Vite (build tool)
- CSS3 (styling)

**Backend**:
- Flask 2.x
- Flask-CORS
- Python 3.9+

**ML**:
- XGBoost 1.7+
- scikit-learn 1.0+
- pandas, numpy
- Librosa 0.10+

**Storage**:
- joblib (model persistence)
- JSON (metadata)
- CSV (datasets)
