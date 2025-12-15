# Entity Relationship Diagram (ERD)

This diagram shows the data model for the Song Virality Prediction System, including all entities and their relationships.

## Diagram

```mermaid
erDiagram
    SONG ||--o{ PREDICTION : "has"
    ARTIST ||--o{ SONG : "creates"
    LANGUAGE ||--o{ SONG : "sung_in"
    AUDIO_FILE ||--o| SONG : "represents"
    MODEL ||--o{ PREDICTION : "generates"
    
    SONG {
        string track_id PK
        string track_name
        string artist_id FK
        int year
        int popularity
        string album_name
        string artwork_url
        string track_url
        string language_id FK
        float danceability "0-1: Rhythm suitability"
        float energy "0-1: Intensity/activity"
        int key "0-11: Musical key"
        float loudness "-60-0 dB: Overall loudness"
        int mode "0=Minor, 1=Major"
        float speechiness "0-1: Spoken words presence"
        float acousticness "0-1: Acoustic likelihood"
        float instrumentalness "0-1: Vocal absence"
        float liveness "0-1: Audience presence"
        float valence "0-1: Musical positivity"
        float tempo "BPM: Track tempo"
        int duration_ms "Track duration"
    }
    
    ARTIST {
        string artist_id PK
        string artist_name
        string genre
        int total_songs
        float avg_popularity
    }
    
    LANGUAGE {
        string language_id PK
        string language_name
        string language_code
    }
    
    AUDIO_FILE {
        string file_id PK
        string track_id FK
        string file_path
        string file_format
        int file_size_bytes
        datetime upload_timestamp
    }
    
    PREDICTION {
        string prediction_id PK
        string track_id FK
        string model_id FK
        float hit_probability "0-1: Predicted hit score"
        float confidence "0-1: Model confidence"
        string classification "hit/miss"
        datetime prediction_timestamp
        json input_features
    }
    
    MODEL {
        string model_id PK
        string model_type "XGBClassifier"
        string version
        float accuracy
        datetime training_timestamp
        string data_hash
        int training_samples
        json feature_names
        json model_metadata
    }
```

## Entity Descriptions

### SONG
The core entity representing a musical track with all 12 musical DNA features used for prediction:
- **Rhythm Features**: danceability, tempo, duration_ms
- **Energy Features**: energy, loudness, valence
- **Texture Features**: acousticness, instrumentalness, speechiness, liveness
- **Tonal Features**: key, mode

### ARTIST
Represents the creator of songs, linked to one or more tracks.

### LANGUAGE
Represents the language of the song lyrics (e.g., English, Tamil, Spanish).

### AUDIO_FILE
Stores uploaded audio files that can be analyzed to extract musical features.

### PREDICTION
Records each prediction made by the system, including:
- Predicted hit probability (0-1 scale)
- Confidence score
- Classification result (hit/miss)
- Timestamp and input features

### MODEL
Stores metadata about the trained ML model:
- Model type (XGBoost Classifier)
- Training accuracy (~87%)
- Training date and data hash
- Feature names and importance

## Key Relationships

1. **SONG ↔ PREDICTION**: One song can have multiple predictions over time
2. **ARTIST ↔ SONG**: One artist creates multiple songs
3. **LANGUAGE ↔ SONG**: Songs are sung in one language
4. **AUDIO_FILE ↔ SONG**: Each audio file represents one song
5. **MODEL ↔ PREDICTION**: One model generates multiple predictions

## Notes

- The 12 musical DNA features are the primary predictors for hit probability
- Songs with popularity ≥ 50 are classified as "hits" (target threshold)
- The system uses binary classification: hit (1) or miss (0)
