# Backend-Frontend Integration Guide

## Architecture Overview

The Song Virality Prediction system now consists of:

### **Backend (Python/Flask)**
- **Flask REST API** running on `http://localhost:5000`
- **XGBoost ML Model** for song hit probability predictions
- **Lazy model loading** for fast startup
- **CORS enabled** for frontend requests

### **Frontend (React/Vite)**
- **React application** running on `http://localhost:5173`
- **Vite dev server proxy** to backend API
- **Fallback client-side predictor** when API unavailable
- **Real-time prediction** with slider controls

---

## Running Both Servers Together

### **Option 1: Start Both Servers Manually (Recommended for Development)**

**Terminal 1 - Start Flask API:**
```bash
cd d:\project\Song_virality_predict
.\.venv\Scripts\Activate.ps1
$env:FLASK_ENV='production'
cd backend
python -m api.server
```

**Terminal 2 - Start Vite Frontend:**
```bash
cd d:\project\Song_virality_predict\frontend
npm run dev
```

Then navigate to: **http://localhost:5173**

### **Option 2: Use Development Server Script**
```bash
cd d:\project\Song_virality_predict
python dev_server.py
```

This starts both servers concurrently and provides a clean shutdown.

---

## API Endpoints

### **Health Check**
```
GET /api/health
```
Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### **Make Prediction**
```
POST /api/predict
Content-Type: application/json

{
  "danceability": 0.65,
  "energy": 0.72,
  "key": 5,
  "loudness": -6.5,
  "mode": 1,
  "speechiness": 0.08,
  "acousticness": 0.25,
  "instrumentalness": 0.05,
  "liveness": 0.15,
  "valence": 0.58,
  "tempo": 125,
  "duration_ms": 210000
}
```

Response:
```json
{
  "hit_probability": 0.732,
  "confidence": 0.85,
  "prediction": "hit",
  "model_version": "1.0.0",
  "timestamp": "2025-11-18T12:00:00"
}
```

### **Model Information**
```
GET /api/model-info
```

---

## How Frontend-Backend Communication Works

1. **User submits song features** via React form sliders
2. **PredictorForm.jsx sends POST request** to `/api/predict`
3. **Vite proxy forwards request** to Flask API (localhost:5000)
4. **Flask API loads XGBoost model** (lazy loading on first request)
5. **Model predicts hit probability** based on features
6. **API returns prediction JSON** to frontend
7. **React component displays result** with confidence score

### **Fallback Mechanism**
If the API is unavailable:
- Frontend automatically uses **client-side fallback predictor**
- Weighted formula based on feature importance
- No user-facing error - seamless experience

---

## Project Structure

```
Song_virality_predict/
├── frontend/                           # React/Vite application
│   ├── src/
│   │   ├── components/
│   │   │   └── PredictorForm.jsx      # Form with API integration
│   │   ├── App.jsx                    # Main app component
│   │   └── styles.css                 # Styling
│   ├── vite.config.js                 # Vite config with API proxy
│   └── package.json                   # Frontend dependencies
│
├── backend/                            # Python/Flask API
│   ├── api/
│   │   ├── server.py                  # Flask API server
│   │   └── __init__.py
│   ├── models/                        # ML models & weights
│   │   ├── predict_main.py            # Model training code
│   │   ├── song_hit_model.pkl         # Trained XGBoost model
│   │   ├── song_hit_model_features.pkl # Feature list
│   │   └── model_metadata.json        # Model info
│   ├── scripts/                       # Utility scripts
│   ├── utils/                         # Helper functions
│   ├── requirements.txt               # Backend dependencies
│   └── README.md                      # Backend documentation
│
├── dev_server.py                      # Start both servers
├── package.json                       # Root scripts
└── vercel.json                        # Deployment config
```

---

## Dependencies

### **Backend (Python)**
```
flask>=2.3.0
flask-cors>=4.0.0
xgboost>=2.0.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
```

Install with:
```bash
cd backend
pip install -r requirements.txt
```

### **Frontend (Node.js)**
```
react@^18.2.0
react-dom@^18.2.0
vite@^5.0.0
@vitejs/plugin-react@^4.2.0
terser@^5.24.0
```

Install with:
```bash
cd frontend
npm install
```

---

## Development Workflow

1. **Start Backend API:**
   ```bash
   cd backend
   python -m api.server
   ```

2. **Start Frontend Dev Server:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open browser:** `http://localhost:5173`

4. **Make changes:** Edit React components or Flask endpoints
   - React changes hot-reload automatically
   - Flask changes require server restart

---

## Testing the Integration

### **Test 1: Health Check**
```bash
curl http://localhost:5000/api/health
```

### **Test 2: Prediction (PowerShell)**
```powershell
$body = @{
  danceability=0.65; energy=0.72; key=5; loudness=-6.5;
  mode=1; speechiness=0.08; acousticness=0.25;
  instrumentalness=0.05; liveness=0.15; valence=0.58;
  tempo=125; duration_ms=210000
} | ConvertTo-Json

Invoke-WebRequest http://localhost:5000/api/predict `
  -Method POST -ContentType "application/json" -Body $body
```

### **Test 3: Frontend UI**
- Navigate to `http://localhost:5173`
- Adjust sliders to change song features
- Click "Predict Virality" button
- Should see prediction result (from API or fallback)

---

## Deployment

### **Production Build**
```bash
cd frontend
npm run build
```
Output: `frontend/dist/` folder ready for Vercel

### **Deploy to Vercel**
```bash
git push origin feature/gamified-frontend
```
Vercel automatically:
1. Builds frontend (React → static files)
2. Copies to `dist/` folder
3. Deploys as static site

### **Backend Deployment (Optional)**
For production backend, use:
- **Heroku** with Procfile
- **AWS Lambda** with Zappa
- **DigitalOcean** with Gunicorn
- **Railway/Render** for easy Python deployment

---

## Troubleshooting

### **API Not Responding**
- Check Flask server is running on port 5000
- Verify `http://localhost:5000/api/health` returns 200 OK
- Check firewall isn't blocking port 5000

### **Model Loading Fails**
- Verify `backend/models/song_hit_model.pkl` exists
- Check file permissions
- Model loads on first request (lazy loading)

### **Frontend Can't Reach API**
- Ensure Vite proxy is configured in `vite.config.js`
- Check `/api/*` routes proxy to `http://localhost:5000`
- Clear browser cache if seeing old behavior

### **Prediction Returns Error**
- Backend returns 503 if model fails to load
- Check `Flask server logs` for error details
- Frontend falls back to client-side predictor

---

## Next Steps

1. **Deploy Backend API**
   - Set up Heroku/Railway account
   - Configure environment variables
   - Push backend to production

2. **Update Frontend API URL**
   - Change `/api/predict` to production URL
   - Update Vite proxy configuration

3. **Add Database**
   - Store prediction history
   - Track user scores
   - Analytics

4. **Blockchain Integration** (Optional)
   - Connect Avalanche oracle contract
   - Record predictions on-chain
   - Reward system

