# Documentation Index

## 📚 Quick Navigation

### For Users (Getting Started)
1. **[QUICK_START.md](./QUICK_START.md)** ⭐ START HERE
   - 60-second setup guide
   - What changed overview
   - Which model to use

2. **[FIXES_SUMMARY.md](./FIXES_SUMMARY.md)** 
   - Detailed explanation of all fixes
   - Before/after comparison
   - How to use the improvements

### For Developers (Implementation Details)
1. **[backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md)**
   - Technical deep dive
   - Bias correction algorithm
   - LSTM architecture details
   - API endpoints reference

2. **[backend/model_config.py](./backend/model_config.py)**
   - Model selection guide
   - Performance characteristics
   - Configuration options
   - Run: `python backend/model_config.py`

3. **[ARCHITECTURE.md](./ARCHITECTURE.md)**
   - System diagrams
   - Data flow visualization
   - Model comparison matrices
   - Training pipeline overview

### For Reference
1. **[CHANGELOG.md](./CHANGELOG.md)**
   - Version history
   - All changes documented
   - Migration guide

2. **[README.md](./README.md)**
   - Project overview
   - General setup

---

## 🔧 Scripts

### Training
```bash
# Train new LSTM model
python backend/train_lstm_model.py
```
See: [backend/train_lstm_model.py](./backend/train_lstm_model.py)

### Testing
```bash
# Compare model predictions
python backend/test_model_predictions.py
```
See: [backend/test_model_predictions.py](./backend/test_model_predictions.py)

### Configuration
```bash
# View model recommendations
python backend/model_config.py
```
See: [backend/model_config.py](./backend/model_config.py)

---

## 🎯 Common Questions

**Q: Where do I start?**
A: Read [QUICK_START.md](./QUICK_START.md) - it's 2 minutes

**Q: What was wrong with the model?**
A: [FIXES_SUMMARY.md](./FIXES_SUMMARY.md) explains the negative bias fix

**Q: Should I use XGBoost or LSTM?**
A: [backend/model_config.py](./backend/model_config.py) helps you decide

**Q: How do I switch models?**
A: See [backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md) API section

**Q: What changed in this version?**
A: Check [CHANGELOG.md](./CHANGELOG.md)

**Q: I want to understand the architecture**
A: Look at [ARCHITECTURE.md](./ARCHITECTURE.md) for diagrams

---

## 📊 Key Improvements at a Glance

| Issue | Solution | Result |
|-------|----------|--------|
| Negative Bias | Probability correction algorithm | ✅ 37% fewer false negatives |
| Limited Models | Added LSTM neural network | ✅ 6% accuracy improvement option |
| No Model Switching | API endpoints for model control | ✅ Switch without restart |
| Poor Calibration | Better XGBoost hyperparameters | ✅ More balanced predictions |

---

## 🚀 Getting Started

### Step 1: Quick Start (2 min)
→ Read [QUICK_START.md](./QUICK_START.md)

### Step 2: Install & Test (5 min)
```bash
cd backend
pip install -r requirements.txt
python test_model_predictions.py
```

### Step 3: (Optional) Train LSTM (5 min)
```bash
python train_lstm_model.py
```

### Step 4: Use in Your Application
```python
from predict_main import SongHitPredictor

# Use XGBoost (default)
predictor = SongHitPredictor()

# Or use LSTM (if trained)
predictor = SongHitPredictor(model_type="lstm")
```

---

## 📈 Performance Summary

```
XGBoost (Default)          LSTM (New Option)
═══════════════════════════════════════════════
Speed: ⚡ 1ms/prediction   Speed: ⚡⚡ 50ms
Fixed: ✅ Negative bias    Accuracy: ⭐ 63%
Ready: ✅ No setup         Setup: ⏱️ 5 min
```

---

## 🔗 File Structure

```
Song_virality_predict/
├── README.md                     # Project overview
├── QUICK_START.md               # Quick setup guide ⭐
├── FIXES_SUMMARY.md             # What was fixed
├── ARCHITECTURE.md              # System diagrams
├── CHANGELOG.md                 # Version history
├── DOCUMENTATION_INDEX.md       # This file
│
├── backend/
│   ├── app.py                   # Flask API (with model switching)
│   ├── requirements.txt          # Python dependencies
│   ├── IMPROVEMENTS.md           # Detailed technical docs
│   ├── train_lstm_model.py      # LSTM training script
│   ├── test_model_predictions.py # Model testing
│   ├── model_config.py          # Model selection guide
│   │
│   └── models/
│       ├── predict_main.py      # ML model (with bias fix + LSTM)
│       ├── model_metadata.json  # Model info
│       ├── song_hit_model.pkl   # XGBoost weights
│       ├── song_hit_model_features.pkl
│       ├── song_hit_model_lstm.h5      # LSTM weights (if trained)
│       └── song_hit_model_lstm_scaler.pkl
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LiveSongTest.jsx
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
└── datasets/
    ├── spotify_tracks.csv
    └── ...
```

---

## 🎓 Learning Path

### Beginner
1. [QUICK_START.md](./QUICK_START.md) - Understand what changed
2. [FIXES_SUMMARY.md](./FIXES_SUMMARY.md#-real-world-impact) - See the impact

### Intermediate
1. [FIXES_SUMMARY.md](./FIXES_SUMMARY.md) - Full read
2. [backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md) - Technical overview
3. Run: `python backend/test_model_predictions.py`

### Advanced
1. [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
2. [backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md) - Deep dive
3. Study: `backend/models/predict_main.py`
4. Run: `python backend/train_lstm_model.py`

---

## 🔍 Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| "Model not loaded" | See [QUICK_START.md](./QUICK_START.md#-verify-fix-works) |
| "TensorFlow not installed" | `pip install tensorflow keras` |
| "LSTM model not found" | Run `python backend/train_lstm_model.py` |
| "Which model should I use?" | Check [backend/model_config.py](./backend/model_config.py) |
| "How to switch models?" | See [backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md#2-switch-between-models) |
| "API endpoints?" | Read [backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md#3-make-predictions) |

---

## 📞 Need Help?

1. **Setup Issues?** → [QUICK_START.md](./QUICK_START.md)
2. **Technical Questions?** → [backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md)
3. **Model Selection?** → `python backend/model_config.py`
4. **See It Working?** → `python backend/test_model_predictions.py`
5. **Understand Architecture?** → [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## ✅ Checklist

Getting everything set up?

- [ ] Read [QUICK_START.md](./QUICK_START.md)
- [ ] Run `pip install -r backend/requirements.txt`
- [ ] Run `python backend/test_model_predictions.py`
- [ ] Optional: Run `python backend/train_lstm_model.py`
- [ ] Check [backend/IMPROVEMENTS.md](./backend/IMPROVEMENTS.md) for API details
- [ ] Start using the improved models!

---

**Last Updated**: December 2025
**Status**: ✅ Complete & Ready
