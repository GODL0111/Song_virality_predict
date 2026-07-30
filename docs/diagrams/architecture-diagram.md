# System Architecture Diagram

This diagram illustrates the complete system architecture of the Song Virality Prediction System, showing all layers and their interactions.

## Diagram

```mermaid
graph TB
    subgraph "Client Layer - Browser"
        A[React Frontend<br/>Port 5173]
        A1[App.jsx<br/>State Manager]
        A2[Layout.jsx<br/>Navigation & Theme]
        A3[HomePage<br/>Manual Input]
        A4[LiveSongTest<br/>Audio Upload]
        A5[Creators Page<br/>Team Info]
        A6[GameDashboard<br/>Score & Logs]
        A --> A1
        A1 --> A2
        A2 --> A3
        A2 --> A4
        A2 --> A5
        A2 --> A6
    end
    
    subgraph "Backend Layer - Flask API Server"
        B[Flask App<br/>Port 5001]
        B1["/api/predict<br/>Manual Prediction"]
        B2["/api/analyze-audio<br/>Audio Analysis"]
        B3["/api/health<br/>Health Check"]
        B4["/api/model-info<br/>Model Metadata"]
        B5["/api/optimal-ranges<br/>Feature Ranges"]
        B6["/api/feature-importance<br/>Feature Scores"]
        B7["/api/suggest-improvements<br/>Recommendations"]
        B --> B1
        B --> B2
        B --> B3
        B --> B4
        B --> B5
        B --> B6
        B --> B7
    end
    
    subgraph "ML Layer - Prediction Engine"
        C[SongHitPredictor<br/>Class]
        C1[XGBoost Classifier<br/>Binary Classification]
        C2[Feature Extractor<br/>Librosa Integration]
        C3[Model Persistence<br/>joblib]
        C4[Feature Analysis<br/>Importance & Ranges]
        C --> C1
        C --> C2
        C --> C3
        C --> C4
    end
    
    subgraph "Data Layer - Storage"
        D1[(models/<br/>*.pkl files)]
        D2[(datasets/<br/>*.csv files)]
        D3[(data/<br/>temp uploads)]
        D4[model_metadata.json]
    end
    
    %% Client to Backend connections
    A3 -->|HTTP POST<br/>JSON features| B1
    A4 -->|HTTP POST<br/>multipart/form-data| B2
    A -->|HTTP GET| B3
    A -->|HTTP GET| B4
    A -->|HTTP GET| B5
    A -->|HTTP GET| B6
    A3 -->|HTTP POST| B7
    
    %% Backend to ML connections
    B1 --> C
    B2 --> C
    B4 --> C
    B5 --> C
    B6 --> C
    B7 --> C
    
    %% ML to Data connections
    C1 -.->|Load| D1
    C3 -.->|Save/Load| D1
    C -.->|Read training data| D2
    C2 -.->|Process temp files| D3
    C -.->|Read/Write| D4
    
    %% Styling
    classDef frontend fill:#61dafb,stroke:#333,stroke-width:2px,color:#000
    classDef backend fill:#3c873a,stroke:#333,stroke-width:2px,color:#fff
    classDef ml fill:#ff6b35,stroke:#333,stroke-width:2px,color:#fff
    classDef data fill:#ffd700,stroke:#333,stroke-width:2px,color:#000
    
    class A,A1,A2,A3,A4,A5,A6 frontend
    class B,B1,B2,B3,B4,B5,B6,B7 backend
    class C,C1,C2,C3,C4 ml
    class D1,D2,D3,D4 data
```

## Architecture Layers

### 1. Client Layer (Frontend - Port 5173)
Built with **React + Vite**, provides the user interface:
- **App.jsx**: Main application state manager (score, logs, localStorage)
- **Layout.jsx**: Navigation, routing, and theme management
- **HomePage/PredictorForm**: Manual input for 12 musical features
- **LiveSongTest**: Audio file upload and analysis
- **Creators**: Team information page
- **GameDashboard**: Score tracking and prediction history

### 2. Backend Layer (Flask API - Port 5001)
RESTful API built with **Flask + CORS**:
- **POST /api/predict**: Accepts 12 features as JSON, returns prediction
- **POST /api/analyze-audio**: Accepts audio file, extracts features, returns prediction
- **GET /api/health**: Server health check
- **GET /api/model-info**: Model metadata and feature list
- **GET /api/optimal-ranges**: Optimal parameter ranges for hits
- **GET /api/feature-importance**: Feature importance scores
- **POST /api/suggest-improvements**: Improvement suggestions for low-scoring songs

### 3. ML Layer (Prediction Engine)
Core machine learning components:
- **SongHitPredictor Class**: Main predictor with training and prediction methods
- **XGBoost Classifier**: Binary classification model (~87% accuracy)
- **Librosa Integration**: Audio feature extraction from uploaded files
- **Feature Analysis**: Optimal ranges, importance scores, improvement suggestions

### 4. Data Layer (Storage)
Persistent storage:
- **models/**: Trained model files (*.pkl)
- **datasets/**: Training data (spotify_tracks.csv, spotify_songs.csv)
- **data/**: Temporary uploaded audio files
- **model_metadata.json**: Model version, accuracy, features

## Data Flow

1. **Manual Input Path**:
   - User enters 12 features in HomePage
   - POST to `/api/predict`
   - XGBoost model predicts hit probability
   - Results displayed with confidence score

2. **Audio Upload Path**:
   - User uploads audio file in LiveSongTest
   - POST to `/api/analyze-audio`
   - Librosa extracts 12 musical features
   - XGBoost model predicts hit probability
   - Results displayed with extracted features

## Technology Stack

- **Frontend**: React, Vite, CSS3
- **Backend**: Python, Flask, Flask-CORS
- **ML**: XGBoost, scikit-learn, pandas, numpy
- **Audio**: Librosa
- **Persistence**: joblib, JSON
- **Deployment**: Vercel (both frontend and backend)

## Communication Protocol

- **Protocol**: HTTP/REST
- **Data Format**: JSON (application/json) for predictions, multipart/form-data for audio uploads
- **CORS**: Enabled for frontend-backend communication
- **Ports**: Frontend (5173), Backend (5001)
