# Backend Structure

Professional organization of Song Virality Prediction backend.

## Directory Layout

```
backend/
├── models/              # ML models and predictors
│   ├── predict_V0.1.py          # Initial XGBoost predictor
│   ├── predict_main.py          # Enhanced predictor (V0.2)
│   ├── song_hit_model.pkl       # Trained model weights
│   ├── song_hit_model_features.pkl  # Feature mapping
│   ├── model_metadata.json      # Model configuration
│   └── __init__.py
├── api/                 # API and blockchain integration
│   ├── __init__.py
├── scripts/             # Deployment and utility scripts
│   ├── __init__.py
├── utils/               # Helper functions and utilities
│   ├── __init__.py
├── requirements.txt     # Python dependencies
└── __init__.py
```

## Installation

Install backend dependencies:
```bash
pip install -r backend/requirements.txt
```

## Usage

### ML Models
- **predict_V0.1.py**: Initial prediction model with feature engineering
- **predict_main.py**: Enhanced V0.2 model with improved accuracy

### Trained Models
- Pickle files in `models/` directory for quick inference without retraining
- Model metadata in `model_metadata.json`

## Dependencies

All Python dependencies documented in `requirements.txt`:
- pandas, numpy for data processing
- scikit-learn, xgboost for ML
- matplotlib, seaborn for visualization
- web3 for blockchain integration
- flask for API endpoints

