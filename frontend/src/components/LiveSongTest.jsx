import React, { useState } from 'react'
import './LiveSongTest.css'

const BACKEND_URL = typeof window !== 'undefined' && window.BACKEND_URL ? window.BACKEND_URL : 'http://localhost:5000'

export default function LiveSongTest() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [fileName, setFileName] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.type.startsWith('audio/')) {
        setFile(droppedFile)
        setFileName(droppedFile.name)
      }
    }
  }

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setFileName(selectedFile.name)
    }
  }

  const handleUploadAndAnalyze = async () => {
    if (!file) return

    setUploading(true)
    setError(null)
    
    try {
      // Create FormData for file upload
      const formData = new FormData()
      formData.append('file', file)

      // Upload and analyze with backend
      const response = await fetch(`${BACKEND_URL}/api/analyze-audio`, {
        method: 'POST',
        body: formData
      })

      setUploading(false)
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Backend analysis failed')
      }

      setAnalyzing(true)
      const prediction = await response.json()
      
      // Simulate analysis animation duration
      await new Promise(resolve => setTimeout(resolve, 2000))

      setResult({
        fileName: prediction.file_name || fileName,
        viralScore: (prediction.hit_probability * 100).toFixed(1),
        isViral: prediction.hit_probability > 0.6,
        confidence: (prediction.confidence * 100).toFixed(0),
        prediction: prediction.prediction,
        features: prediction.extracted_features
      })

      setAnalyzing(false)
    } catch (err) {
      setError(err.message || 'Failed to analyze song. Make sure backend is running.')
      console.error('Analysis error:', err)
      setAnalyzing(false)
      setUploading(false)
    }
  }

  const resetForm = () => {
    setFile(null)
    setFileName('')
    setResult(null)
    setAnalyzing(false)
    setUploading(false)
  }

  return (
    <div className="live-song-test">
      <div className="page-header">
        <h2>🎧 Live Song Test</h2>
        <p>Upload a song and get instant viral analysis</p>
      </div>

      <div className="test-container">
        {error && (
          <div className="result error" style={{marginBottom: '24px'}}>
            <p>Error: {error}</p>
          </div>
        )}
        {!result ? (
          <div className="upload-section">
            {/* Upload Area */}
            <div 
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="upload-content">
                <div className="upload-icon">🎵</div>
                <h3>Upload Your Song</h3>
                <p>Drag and drop your audio file here or</p>
                <label className="file-input-label">
                  <input 
                    type="file" 
                    accept="audio/*" 
                    onChange={handleFileSelect}
                    disabled={analyzing || uploading}
                  />
                  <span className="file-input-button">Choose File</span>
                </label>
                <p className="file-info">MP3, WAV, OGG, M4A up to 50MB</p>
              </div>
            </div>

            {/* Selected File Display */}
            {file && (
              <div className="file-selected">
                <div className="file-details">
                  <span className="file-icon">🎵</span>
                  <div className="file-info-box">
                    <p className="file-name">{fileName}</p>
                    <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
                {!uploading && !analyzing && (
                  <button 
                    className="btn primary large"
                    onClick={handleUploadAndAnalyze}
                  >
                    Analyze Song
                  </button>
                )}
              </div>
            )}

            {/* Upload Progress */}
            {uploading && (
              <div className="progress-section">
                <div className="progress-text">Uploading song...</div>
                <div className="progress-bar">
                  <div className="progress-fill upload-progress"></div>
                </div>
                <div className="upload-steps">
                  <div className="step active">
                    <div className="step-icon">✓</div>
                    <span>File Received</span>
                  </div>
                  <div className="step">
                    <div className="step-icon">⏳</div>
                    <span>Processing</span>
                  </div>
                  <div className="step">
                    <div className="step-icon">🔍</div>
                    <span>Analyzing</span>
                  </div>
                </div>
              </div>
            )}

            {/* Analysis Animation */}
            {analyzing && (
              <div className="analysis-section">
                <div className="analysis-header">
                  <h3>Analyzing Your Song</h3>
                  <p>Training model and extracting audio features...</p>
                </div>

                <div className="analysis-visual">
                  <div className="waveform">
                    {[...Array(50)].map((_, i) => (
                      <div 
                        key={i} 
                        className="wave-bar"
                        style={{
                          '--height': `${30 + Math.random() * 70}%`,
                          '--delay': `${i * 0.05}s`
                        }}
                      ></div>
                    ))}
                  </div>

                  <div className="analyzing-spinner">
                    <div className="spinner"></div>
                    <p>Processing audio...</p>
                  </div>

                  <div className="analysis-steps">
                    <div className="analysis-step">
                      <div className="step-dot active"></div>
                      <span>Loading Audio</span>
                    </div>
                    <div className="analysis-step">
                      <div className="step-dot"></div>
                      <span>Feature Extraction</span>
                    </div>
                    <div className="analysis-step">
                      <div className="step-dot"></div>
                      <span>Model Training</span>
                    </div>
                    <div className="analysis-step">
                      <div className="step-dot"></div>
                      <span>Results</span>
                    </div>
                  </div>
                </div>

                <div className="progress-bar">
                  <div className="progress-fill analyze-progress"></div>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Results Display */
          <div className="results-section">
            <div className={`result-card ${result.isViral ? 'viral' : 'not-viral'}`}>
              <div className="result-header">
                <h3>{result.isViral ? '🚀 Viral Hit!' : '📊 Below Average'}</h3>
                <p className="song-title">{result.fileName}</p>
              </div>

              <div className="viral-score-section">
                <div className="viral-meter">
                  <div className="viral-bar">
                    <div 
                      className="viral-fill"
                      style={{ width: `${result.viralScore}%` }}
                    ></div>
                  </div>
                  <div className="score-display">
                    <span className="score-value">{result.viralScore}%</span>
                    <span className="score-label">Viral Score</span>
                  </div>
                </div>

                <div className="confidence-badge">
                  <span className="confidence-icon">✓</span>
                  <span className="confidence-text">{result.confidence}% confidence</span>
                </div>
              </div>

              <div className="result-features">
                <h4>🎵 Core Audio Features (12)</h4>
                <div className="features-grid">
                  <div className="feature">
                    <span className="feature-label">Danceability</span>
                    <div className="feature-bar">
                      <div className="feature-fill" style={{ width: `${result.features.danceability * 100}%` }}></div>
                    </div>
                    <span className="feature-value">{(result.features.danceability * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Energy</span>
                    <div className="feature-bar">
                      <div className="feature-fill" style={{ width: `${result.features.energy * 100}%` }}></div>
                    </div>
                    <span className="feature-value">{(result.features.energy * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Valence (Positivity)</span>
                    <div className="feature-bar">
                      <div className="feature-fill" style={{ width: `${result.features.valence * 100}%` }}></div>
                    </div>
                    <span className="feature-value">{(result.features.valence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Acousticness</span>
                    <div className="feature-bar">
                      <div className="feature-fill" style={{ width: `${result.features.acousticness * 100}%` }}></div>
                    </div>
                    <span className="feature-value">{(result.features.acousticness * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Speechiness</span>
                    <div className="feature-bar">
                      <div className="feature-fill" style={{ width: `${result.features.speechiness * 100}%` }}></div>
                    </div>
                    <span className="feature-value">{(result.features.speechiness * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Instrumentalness</span>
                    <div className="feature-bar">
                      <div className="feature-fill" style={{ width: `${result.features.instrumentalness * 100}%` }}></div>
                    </div>
                    <span className="feature-value">{(result.features.instrumentalness * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Liveness</span>
                    <div className="feature-bar">
                      <div className="feature-fill" style={{ width: `${result.features.liveness * 100}%` }}></div>
                    </div>
                    <span className="feature-value">{(result.features.liveness * 100).toFixed(0)}%</span>
                  </div>
                </div>

                <div className="feature-stats">
                  <div className="stat-item">
                    <span className="stat-label">Tempo</span>
                    <span className="stat-value">{result.features.tempo?.toFixed(0) || 'N/A'} BPM</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Loudness</span>
                    <span className="stat-value">{result.features.loudness?.toFixed(1) || 'N/A'} dB</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Key</span>
                    <span className="stat-value">
                      {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][result.features.key] || 'N/A'}
                      {' '}{result.features.mode === 1 ? 'Major' : 'Minor'}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Duration</span>
                    <span className="stat-value">
                      {result.features.duration_ms 
                        ? `${Math.floor(result.features.duration_ms / 60000)}:${String(Math.floor((result.features.duration_ms % 60000) / 1000)).padStart(2, '0')}`
                        : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Extended Features Display */}
              {result.features._all_features && (
                <div className="extended-features">
                  <h4>🔬 Extended Analysis ({result.features._feature_count} features extracted)</h4>
                  <div className="extended-features-grid">
                    {/* Spectral Features */}
                    <div className="feature-group">
                      <h5>📊 Spectral Analysis</h5>
                      <div className="feature-list">
                        <div className="ext-feature"><span>Centroid</span><span>{result.features._all_features.spectral_centroid_hz?.toFixed(0)} Hz</span></div>
                        <div className="ext-feature"><span>Bandwidth</span><span>{result.features._all_features.spectral_bandwidth_hz?.toFixed(0)} Hz</span></div>
                        <div className="ext-feature"><span>Rolloff</span><span>{result.features._all_features.spectral_rolloff_hz?.toFixed(0)} Hz</span></div>
                        <div className="ext-feature"><span>Flatness</span><span>{(result.features._all_features.spectral_flatness_mean * 100)?.toFixed(2)}%</span></div>
                      </div>
                    </div>

                    {/* Rhythm Analysis */}
                    <div className="feature-group">
                      <h5>🥁 Rhythm Analysis</h5>
                      <div className="feature-list">
                        <div className="ext-feature"><span>Beat Count</span><span>{result.features._all_features.beat_count}</span></div>
                        <div className="ext-feature"><span>Beat Regularity</span><span>{(result.features._all_features.beat_regularity * 100)?.toFixed(1)}%</span></div>
                        <div className="ext-feature"><span>Rhythm Strength</span><span>{(result.features._all_features.rhythm_strength * 100)?.toFixed(1)}%</span></div>
                        <div className="ext-feature"><span>Onset Strength</span><span>{result.features._all_features.onset_strength_mean?.toFixed(2)}</span></div>
                      </div>
                    </div>

                    {/* Harmonic Analysis */}
                    <div className="feature-group">
                      <h5>🎼 Harmonic Analysis</h5>
                      <div className="feature-list">
                        <div className="ext-feature"><span>Key</span><span>{result.features._all_features.key_name} ({result.features._all_features.mode_name})</span></div>
                        <div className="ext-feature"><span>Key Strength</span><span>{(result.features._all_features.key_strength * 100)?.toFixed(1)}%</span></div>
                        <div className="ext-feature"><span>Harmonic Ratio</span><span>{(result.features._all_features.harmonic_ratio * 100)?.toFixed(1)}%</span></div>
                        <div className="ext-feature"><span>Tonnetz Mean</span><span>{result.features._all_features.tonnetz_mean?.toFixed(4)}</span></div>
                      </div>
                    </div>

                    {/* Energy Analysis */}
                    <div className="feature-group">
                      <h5>⚡ Energy Analysis</h5>
                      <div className="feature-list">
                        <div className="ext-feature"><span>RMS Mean</span><span>{result.features._all_features.rms_mean?.toFixed(4)}</span></div>
                        <div className="ext-feature"><span>Dynamic Range</span><span>{result.features._all_features.dynamic_range?.toFixed(2)}</span></div>
                        <div className="ext-feature"><span>Loudness Raw</span><span>{result.features._all_features.loudness_raw_db?.toFixed(1)} dB</span></div>
                        <div className="ext-feature"><span>ZCR</span><span>{result.features._all_features.zero_crossing_rate?.toFixed(4)}</span></div>
                      </div>
                    </div>
                  </div>

                  {/* MFCC Visualization */}
                  <div className="mfcc-section">
                    <h5>🎤 MFCC Coefficients (Timbral Texture)</h5>
                    <div className="mfcc-bars">
                      {[...Array(20)].map((_, i) => {
                        const mfccVal = result.features._all_features[`mfcc_${i+1}`] || 0
                        const normalized = Math.min(100, Math.max(0, (mfccVal + 50) * 1.5))
                        return (
                          <div key={i} className="mfcc-bar-container">
                            <div 
                              className="mfcc-bar" 
                              style={{ height: `${normalized}%`, background: `hsl(${280 - i * 12}, 70%, 50%)` }}
                            ></div>
                            <span className="mfcc-label">{i+1}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  <p className="feature-note">
                    ⚠️ Note: Librosa extracts raw audio features. Spotify uses proprietary ML algorithms trained on millions of songs, 
                    so values may differ from Spotify's API. Our calibration attempts to approximate Spotify's definitions.
                  </p>
                </div>
              )}
            </div>

            <button className="btn secondary" onClick={resetForm}>
              Test Another Song
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
