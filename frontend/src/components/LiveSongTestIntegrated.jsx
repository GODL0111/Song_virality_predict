import React, { useState } from 'react'
import './LiveSongTest.css'

export default function LiveSongTestIntegrated() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [fileName, setFileName] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(0)

  // Use relative URL for Vercel (same domain), fallback to localhost for development
  const BACKEND_URL = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? '' : 'http://localhost:5000')

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
    setError(null)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.type.startsWith('audio/')) {
        setFile(droppedFile)
        setFileName(droppedFile.name)
      } else {
        setError('Please select an audio file')
      }
    }
  }

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setFileName(selectedFile.name)
      setError(null)
    }
  }

  const handleUploadAndAnalyze = async () => {
    if (!file) {
      setError('No file selected')
      return
    }

    setError(null)
    setUploading(true)
    setAnalyzing(false)
    setProgress(0)

    try {
      // Create FormData for file upload
      const formData = new FormData()
      formData.append('file', file)

      // Upload and analyze
      const response = await fetch(`${BACKEND_URL}/api/analyze-audio`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type, let browser set it with boundary
      })

      setProgress(50)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Upload failed')
      }

      const data = await response.json()
      setProgress(100)

      // Simulate brief analysis animation
      setUploading(false)
      setAnalyzing(true)

      // Brief delay for animation
      setTimeout(() => {
        setAnalyzing(false)
        setResult(data)
      }, 2000)

    } catch (err) {
      setError(err.message)
      setUploading(false)
      setAnalyzing(false)
    }
  }

  const resetForm = () => {
    setFile(null)
    setFileName('')
    setResult(null)
    setAnalyzing(false)
    setUploading(false)
    setError(null)
    setProgress(0)
  }

  const handleDownloadReport = () => {
    if (!result) return
    
    const report = {
      fileName: result.fileName,
      viralPrediction: result.isViral ? 'Viral Hit' : 'Not Viral',
      probability: (result.probability * 100).toFixed(2) + '%',
      confidence: (result.confidence * 100).toFixed(0) + '%',
      features: result.features,
      timestamp: new Date().toISOString()
    }

    const dataStr = JSON.stringify(report, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `analysis_${Date.now()}.json`
    link.click()
  }

  return (
    <div className="live-song-test">
      <div className="page-header">
        <h2>Song Analysis</h2>
        <p>Test your track and see how it performs</p>
      </div>

      <div className="test-container">
        {!result ? (
          <div className="upload-section">
            {/* Error Display */}
            {error && (
              <div className="error-message">
                <span>{error}</span>
                <button onClick={() => setError(null)}>×</button>
              </div>
            )}

            {/* Upload Area */}
            <div 
              className={`upload-area ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="upload-content">
                <div className="upload-icon">↓</div>
                <h3>Drop your file here</h3>
                <p>or browse to select</p>
                <label className="file-input-label">
                  <input 
                    type="file" 
                    accept="audio/*" 
                    onChange={handleFileSelect}
                    disabled={analyzing || uploading}
                  />
                  <span className="file-input-button">Choose File</span>
                </label>
                <p className="file-info">MP3, WAV, OGG, M4A, FLAC (max 50MB)</p>
              </div>
            </div>

            {/* Selected File Display */}
            {file && (
              <div className="file-selected">
                <div className="file-details">
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
                <div className="progress-text">Processing...</div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill upload-progress"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* Analysis Animation */}
            {analyzing && (
              <div className="analysis-section">
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
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Results Display */
          <div className="results-section">
            <div className={`result-card ${result.isViral ? 'viral' : 'not-viral'}`}>
              <div className="result-header">
                <h3>{result.isViral ? 'Good Match' : 'Needs Work'}</h3>
                <p className="song-title">{result.fileName}</p>
              </div>

              <div className="viral-score-section">
                <div className="viral-meter">
                  <div className="viral-bar">
                    <div 
                      className="viral-fill"
                      style={{ width: `${result.probability * 100}%` }}
                    ></div>
                  </div>
                  <div className="score-display">
                    <span className="score-value">{(result.probability * 100).toFixed(1)}%</span>
                    <span className="score-label">Hit Probability</span>
                  </div>
                </div>

                <div className="confidence-badge">
                  <span className="confidence-text">{(result.confidence * 100).toFixed(0)}% confidence</span>
                </div>
              </div>

              <div className="result-features">
                <h4>Track Analysis</h4>
                <div className="features-grid">
                  <div className="feature">
                    <span className="feature-label">Danceability</span>
                    <div className="feature-bar">
                      <div 
                        className="feature-fill"
                        style={{ width: `${result.features.danceability * 100}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{(result.features.danceability * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Energy</span>
                    <div className="feature-bar">
                      <div 
                        className="feature-fill"
                        style={{ width: `${result.features.energy * 100}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{(result.features.energy * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Valence</span>
                    <div className="feature-bar">
                      <div 
                        className="feature-fill"
                        style={{ width: `${result.features.valence * 100}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{(result.features.valence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Acousticness</span>
                    <div className="feature-bar">
                      <div 
                        className="feature-fill"
                        style={{ width: `${result.features.acousticness * 100}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{(result.features.acousticness * 100).toFixed(0)}%</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Tempo</span>
                    <div className="feature-bar">
                      <div 
                        className="feature-fill"
                        style={{ width: `${Math.min(100, (result.features.tempo / 200) * 100)}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{result.features.tempo.toFixed(0)} BPM</span>
                  </div>
                  <div className="feature">
                    <span className="feature-label">Duration</span>
                    <div className="feature-bar">
                      <div 
                        className="feature-fill"
                        style={{ width: `${Math.min(100, (result.features.duration_ms / 600000) * 100)}%` }}
                      ></div>
                    </div>
                    <span className="feature-value">{(result.features.duration_ms / 1000).toFixed(0)}s</span>
                  </div>
                </div>
              </div>

              <div className="result-actions">
                <button className="btn secondary" onClick={handleDownloadReport}>
                  Download
                </button>
                <button className="btn primary" onClick={resetForm}>
                  Try Another
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
