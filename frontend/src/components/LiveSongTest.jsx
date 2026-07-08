import React, { useState, useRef } from 'react'
import html2pdf from 'html2pdf.js'
import ReportTemplate from './ReportTemplate'
import './LiveSongTest.css'

// Use relative URL for Vercel (same domain), fallback to localhost for development
const BACKEND_URL = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? '' : 'http://localhost:5000')

export default function LiveSongTest() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStep, setUploadStep] = useState(1)
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzingHooks, setAnalyzingHooks] = useState(false)
  const [hookLoadingText, setHookLoadingText] = useState('Mapping beat boundaries...')
  const [result, setResult] = useState(null)
  const [fileName, setFileName] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState(null)
  
  // Playground state
  const [showPlayground, setShowPlayground] = useState(false)
  const [playgroundFeatures, setPlaygroundFeatures] = useState(null)
  const [playgroundScore, setPlaygroundScore] = useState(null)
  const [playgroundLoading, setPlaygroundLoading] = useState(false)
  const [mutatingAudio, setMutatingAudio] = useState(false)
  const [mutatedAudioUrl, setMutatedAudioUrl] = useState(null)
  const debounceTimerRef = useRef(null)

  const reportRef = useRef(null)
  const audioRef = useRef(null)

  const audioUrl = React.useMemo(() => {
    return file ? URL.createObjectURL(file) : null;
  }, [file]);


  const jumpToHook = (startTime) => {
    if (audioRef.current) {
      audioRef.current.currentTime = startTime;
      audioRef.current.play().catch(e => console.log('Playback prevented:', e));
    }
  }

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
    setUploadStep(1)
    setError(null)
    
    // Animate steps while uploading
    const step2Timer = setTimeout(() => setUploadStep(2), 800)
    const step3Timer = setTimeout(() => setUploadStep(3), 1600)

    try {
      // Create FormData for file upload
      const formData = new FormData()
      formData.append('file', file)

      // Add 300-second timeout (5 minutes) to account for Render cold starts and slow upload speeds
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(new Error(`Request timed out trying to reach ${BACKEND_URL || 'Vercel'}. Please check if VITE_API_URL is set correctly in Vercel.`)), 300000)

      // Upload and analyze with backend
      const response = await fetch(`${BACKEND_URL}/api/analyze-audio`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      })
      
      clearTimeout(timeoutId)

      clearTimeout(step2Timer)
      clearTimeout(step3Timer)
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
        features: prediction.features || prediction.extracted_features,
        temporalSegments: [], // To be fetched in Phase 2
        topHooks: [], // To be fetched in Phase 2
        totalDurationSec: prediction.total_duration_sec || 180,
        analysisId: prediction.analysisId,
        prescriptions: prediction.prescriptions || []
      })

      setAnalyzing(false)
    } catch (err) {
      setError(err.message || 'Failed to analyze song. Make sure backend is running.')
      console.error('Analysis error:', err)
      setAnalyzing(false)
      setUploading(false)
    }
  }

  const handleAnalyzeHooks = async () => {
    if (!result || !result.analysisId) return;
    
    setAnalyzingHooks(true);
    setError(null);
    
    // Start text rotation
    const texts = [
      'Mapping beat boundaries...',
      'Extracting local energy deviations...',
      'Calculating composite hook scores...',
      'Isolating top candidates...'
    ];
    let textIdx = 0;
    const interval = setInterval(() => {
      textIdx = (textIdx + 1) % texts.length;
      setHookLoadingText(texts[textIdx]);
    }, 2500);

    try {
      const response = await fetch(`${BACKEND_URL}/api/analyze-hooks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ analysisId: result.analysisId })
      });
      
      clearInterval(interval);
      setAnalyzingHooks(false);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Hook analysis failed');
      }
      
      const hookData = await response.json();
      
      setResult(prev => ({
        ...prev,
        temporalSegments: hookData.temporal_segments || [],
        topHooks: hookData.top_hooks || []
      }));
      
    } catch (err) {
      clearInterval(interval);
      setAnalyzingHooks(false);
      setError(err.message || 'Failed to analyze hooks.');
      console.error('Hook analysis error:', err);
    }
  }

  const resetForm = () => {
    setFile(null)
    setFileName('')
    setResult(null)
    setAnalyzing(false)
    setUploading(false)
    setShowPlayground(false)
  }

  const enterPlayground = () => {
    setPlaygroundFeatures({ ...result.features })
    setPlaygroundScore(result.hit_probability)
    setShowPlayground(true)
  }

  const handlePlaygroundChange = (feature, value) => {
    const newFeatures = { ...playgroundFeatures, [feature]: parseFloat(value) }
    setPlaygroundFeatures(newFeatures)
    
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    setPlaygroundLoading(true)
    
    debounceTimerRef.current = setTimeout(async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newFeatures)
        })
        if (response.ok) {
          const data = await response.json()
          setPlaygroundScore(data.hit_probability)
        }
      } catch (err) {
        console.error("Playground predict error:", err)
      } finally {
        setPlaygroundLoading(false)
      }
    }, 400)
  }

  const applyConsultantTweaks = () => {
    if (!result || !result.prescriptions) return
    let newFeatures = { ...playgroundFeatures }
    result.prescriptions.forEach(rec => {
      newFeatures[rec.feature] = rec.suggested
    })
    setPlaygroundFeatures(newFeatures)
    
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    setPlaygroundLoading(true)
    fetch(`${BACKEND_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newFeatures)
    }).then(res => res.json()).then(data => {
      setPlaygroundScore(data.hit_probability)
      setPlaygroundLoading(false)
    }).catch(() => setPlaygroundLoading(false))
  }

  const generateMutatedAudio = async () => {
    if (!result || !result.analysisId) return;
    
    setMutatingAudio(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/mutate-audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysisId: result.analysisId,
          target_tempo: playgroundFeatures.tempo,
          target_key: playgroundFeatures.key,
          target_loudness: playgroundFeatures.loudness,
          original_tempo: result.features.tempo,
          original_key: result.features.key,
          original_loudness: result.features.loudness
        })
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Failed to mutate audio');
      }
      
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setMutatedAudioUrl(url);
      
    } catch (err) {
      setError(err.message || 'Error mutating audio');
      console.error(err);
    } finally {
      setMutatingAudio(false);
    }
  }

  const handleDownloadReport = () => {
    if (!result || !reportRef.current) return;
    
    const now = new Date();
    const pad = (n) => n.toString().padStart(2, '0');
    const timestamp = `${now.getFullYear()}${pad(now.getMonth()+1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    
    const opt = {
      margin:       [15, 0, 15, 0],
      filename:     `virality_report_${timestamp}.pdf`,
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2, useCORS: true },
      jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' },
      pagebreak:    { mode: ['css', 'legacy'], avoid: '.avoid-break' }
    };
    
    html2pdf().set(opt).from(reportRef.current).save();
  };

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
                  <div className={`step ${uploadStep >= 1 ? 'active' : ''}`}>
                    <div className="step-icon">✓</div>
                    <span>File Received</span>
                  </div>
                  <div className={`step ${uploadStep >= 2 ? 'active' : ''}`}>
                    <div className="step-icon">⏳</div>
                    <span>Processing</span>
                  </div>
                  <div className={`step ${uploadStep >= 3 ? 'active' : ''}`}>
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
              
              {/* VIRAL HOOK DETECTION TRIGGER */}
              {result.temporalSegments && result.temporalSegments.length === 0 && !analyzingHooks && (
                <div style={{ margin: '30px 0 40px', textAlign: 'center' }}>
                  <button 
                    onClick={handleAnalyzeHooks}
                    className="primary-button hook-button"
                    style={{
                      background: 'linear-gradient(135deg, #1DB954 0%, #1ed760 100%)',
                      fontSize: '1.1rem',
                      padding: '15px 30px',
                      boxShadow: '0 8px 20px rgba(29, 185, 84, 0.4)'
                    }}
                  >
                    🎯 Analyze Viral Hooks
                  </button>
                  <p style={{ marginTop: '10px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Deep DSP analysis to find the best 15s TikTok segments (~10 seconds)
                  </p>
                </div>
              )}
              
              {/* HOOK LOADING STATE */}
              {analyzingHooks && (
                <div className="hook-loading" style={{ margin: '40px 0', textAlign: 'center', padding: '30px', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                  <div className="hook-scanner-container" style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.1)', position: 'relative', overflow: 'hidden', borderRadius: '2px', marginBottom: '20px' }}>
                    <div className="hook-scanner" style={{
                      position: 'absolute',
                      top: 0, left: 0, height: '100%', width: '30%',
                      background: 'var(--accent)',
                      boxShadow: '0 0 10px var(--accent)',
                      animation: 'scan 2s infinite ease-in-out alternate'
                    }}></div>
                  </div>
                  <style>{`
                    @keyframes scan {
                      0% { left: -10%; }
                      100% { left: 80%; }
                    }
                  `}</style>
                  <h4 style={{ color: 'var(--accent)', marginBottom: '10px' }}>Analyzing Acoustic Structure</h4>
                  <p style={{ color: 'var(--text-secondary)', animation: 'pulse 1.5s infinite' }}>{hookLoadingText}</p>
                </div>
              )}

              {/* VIRAL HOOK DETECTION (HEATMAP) */}
              {result.temporalSegments && result.temporalSegments.length > 0 && (
                <div className="viral-hook-section" style={{ margin: '30px 0 40px', padding: '20px', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                  <h4 style={{ marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '1.2rem' }}>🔥</span> Viral Hook Detection (15s Segments)
                  </h4>
                  
                  {(() => {
                    const bestSegment = [...result.temporalSegments].sort((a, b) => b.hit_probability - a.hit_probability)[0];
                    const formatTime = (secs) => {
                      const m = Math.floor(secs / 60);
                      const s = Math.floor(secs % 60);
                      return `${m}:${s.toString().padStart(2, '0')}`;
                    };
                    
                    return (
                      <>
                        <div className="top-hooks-list" style={{ marginBottom: '20px' }}>
                          
                          {/* Embedded Audio Player */}
                          {audioUrl && (
                            <div style={{ marginBottom: '20px', padding: '15px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
                              <h5 style={{ margin: '0 0 10px 0', color: 'var(--text-primary)' }}>Playback</h5>
                              <audio 
                                ref={audioRef} 
                                src={audioUrl} 
                                controls 
                                style={{ width: '100%', height: '40px', outline: 'none' }}
                              />
                            </div>
                          )}

                          {result.topHooks && result.topHooks.map((hook, idx) => (
                            <div 
                              key={idx} 
                              className="hook-callout" 
                              onClick={() => jumpToHook(hook.start_time)}
                              style={{ 
                                background: idx === 0 ? 'linear-gradient(135deg, rgba(255,107,107,0.15) 0%, rgba(255,107,107,0.05) 100%)' : 'rgba(255,255,255,0.03)',
                                border: `1px solid ${idx === 0 ? 'rgba(255,107,107,0.4)' : 'rgba(255,255,255,0.1)'}`,
                                padding: '12px 15px',
                                borderRadius: '8px',
                                marginBottom: '10px',
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease'
                              }}
                              onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)'; }}
                              onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
                            >
                              <div>
                                <strong style={{ color: idx === 0 ? 'var(--accent)' : 'inherit', display: 'block', marginBottom: '4px' }}>
                                  {idx === 0 ? '🥇 ' : idx === 1 ? '🥈 ' : '🥉 '} {hook.type}
                                </strong>
                                <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                  {formatTime(hook.start_time)} - {formatTime(hook.end_time)} | {hook.description}
                                </p>
                              </div>
                              <div style={{ textAlign: 'right' }}>
                                <span style={{ display: 'block', fontWeight: 'bold', fontSize: '1.1rem', color: 'var(--accent)' }}>
                                  {(hook.hook_score * 100).toFixed(1)}
                                </span>
                                <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                  Hook Score
                                </span>
                              </div>
                            </div>
                          ))}
                          
                          {/* Fallback for older API versions */}
                          {(!result.topHooks || result.topHooks.length === 0) && (
                            <div className="golden-hook-callout" style={{ 
                              background: 'linear-gradient(135deg, rgba(255,107,107,0.1) 0%, rgba(255,107,107,0.05) 100%)',
                              border: '1px solid rgba(255,107,107,0.3)',
                              padding: '15px',
                              borderRadius: '8px',
                              marginBottom: '20px'
                            }}>
                              <strong style={{ color: 'var(--accent)', display: 'block', marginBottom: '5px' }}>Golden Hook Identified!</strong>
                              <p style={{ margin: 0, fontSize: '0.9rem' }}>
                                The most viral snippet is from <strong>{formatTime(bestSegment.start_time)} - {formatTime(bestSegment.end_time)}</strong> 
                                &nbsp;({(bestSegment.hit_probability * 100).toFixed(1)}% Viral Potential). 
                              </p>
                            </div>
                          )}
                        </div>
                        
                        <div className="heatmap-container" style={{ position: 'relative', height: '40px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                          {result.temporalSegments.map((seg, i) => {
                            const widthPct = ((seg.end_time - seg.start_time) / result.totalDurationSec) * 100;
                            const leftPct = (seg.start_time / result.totalDurationSec) * 100;
                            // Map probability to color: Greenish for high, Reddish for low
                            const hue = seg.hit_probability > 0.5 ? 140 : 0;
                            const saturation = Math.abs(seg.hit_probability - 0.5) * 200; // 0 to 100%
                            const color = `hsl(${hue}, ${saturation}%, 50%)`;
                            
                            return (
                              <div 
                                key={i}
                                className="heatmap-segment"
                                title={`Time: ${formatTime(seg.start_time)}-${formatTime(seg.end_time)} | Hook Score: ${((seg.hook_score || seg.golden_candidate_score || seg.hit_probability || 0)*100).toFixed(1)}`}
                                style={{
                                  position: 'absolute',
                                  left: `${leftPct}%`,
                                  width: `${widthPct}%`,
                                  height: '100%',
                                  background: color,
                                  opacity: 0.8,
                                  borderRight: '1px solid rgba(255,255,255,0.1)'
                                }}
                              />
                            );
                          })}
                        </div>
                        <div className="heatmap-labels" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '5px' }}>
                          <span>0:00</span>
                          <span>{formatTime(result.totalDurationSec)}</span>
                        </div>
                      </>
                    );
                  })()}
                </div>
              )}

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

              {/* Hit Consultant Panel */}
              {result.prescriptions && result.prescriptions.length > 0 && (
                <div className="hit-consultant-panel">
                  <div className="consultant-header">
                    <h3>🤖 Prescriptive Hit Consultant</h3>
                    <p>Mathematical production tweaks to maximize viral probability.</p>
                  </div>
                  <div className="prescriptions-list">
                    {result.prescriptions.map((rec, index) => (
                      <div key={index} className="prescription-card">
                        <div className="prescription-icon">
                          {rec.feature === 'tempo' || rec.feature === 'danceability' ? '🥁' : 
                           rec.feature === 'energy' || rec.feature === 'loudness' ? '⚡' : 
                           rec.feature === 'valence' ? '😊' : '🎛️'}
                        </div>
                        <div className="prescription-details">
                          <h4>
                            {rec.direction === 'INCREASE' ? 'Increase ' : 'Decrease '}
                            {rec.feature.charAt(0).toUpperCase() + rec.feature.slice(1)}
                          </h4>
                          <p>
                            From <strong>{rec.current.toFixed(2)}</strong> to <strong>{rec.suggested.toFixed(2)}</strong>
                          </p>
                        </div>
                        <div className="prescription-gain">
                          <span className="gain-value">+{rec.improvement_percent.toFixed(1)}%</span>
                          <span className="gain-label">Hit Prob</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Playground Mode Toggle */}
              {!showPlayground ? (
                <div className="playground-toggle" style={{textAlign: 'center', marginTop: '20px'}}>
                  <button className="btn secondary" onClick={enterPlayground}>
                    🎛️ Enter Interactive Playground
                  </button>
                </div>
              ) : (
                <div className="playground-panel">
                  <div className="playground-header">
                    <h3>🎛️ Interactive Hit Playground</h3>
                    <p>Tweak the song's DNA and see the predicted hit probability update live.</p>
                  </div>
                  
                  <div className="playground-score-container">
                    <div className="live-score">
                      <span className="live-score-label">Live Hit Probability</span>
                      <div className="live-score-value" style={{ color: playgroundScore > result.hit_probability ? '#00ff88' : playgroundScore < result.hit_probability ? '#ff3385' : 'var(--text-primary)' }}>
                        {playgroundLoading ? <span className="loading-dots">...</span> : `${(playgroundScore * 100).toFixed(1)}%`}
                      </div>
                    </div>
                    {result.prescriptions && result.prescriptions.length > 0 && (
                      <button className="btn primary snap-btn" onClick={applyConsultantTweaks}>
                        ✨ Apply Consultant Tweaks
                      </button>
                    )}
                  </div>

                  <div className="playground-sliders">
                    {playgroundFeatures && Object.keys(playgroundFeatures).filter(k => !k.startsWith('_') && k !== 'duration_ms' && k !== 'key' && k !== 'mode' && k !== 'target_year').map(feature => {
                      let min = 0; let max = 1; let step = 0.01;
                      if (feature === 'tempo') { min = 60; max = 200; step = 1; }
                      if (feature === 'loudness') { min = -30; max = 0; step = 0.1; }
                      
                      return (
                        <div key={feature} className="slider-group">
                          <div className="slider-label-row">
                            <span className="slider-name">{feature.charAt(0).toUpperCase() + feature.slice(1)}</span>
                            <span className="slider-value">{playgroundFeatures[feature].toFixed(2)}</span>
                          </div>
                          <input 
                            type="range" 
                            min={min} 
                            max={max} 
                            step={step} 
                            value={playgroundFeatures[feature]} 
                            onChange={(e) => handlePlaygroundChange(feature, e.target.value)}
                            className="feature-slider"
                          />
                        </div>
                      )
                    })}
                  </div>
                  
                  <div className="audio-mutator-section" style={{marginTop: '30px', textAlign: 'center'}}>
                    {!mutatedAudioUrl ? (
                      <button 
                        className="btn primary" 
                        onClick={generateMutatedAudio} 
                        disabled={mutatingAudio}
                        style={{width: '100%', maxWidth: '400px'}}
                      >
                        {mutatingAudio ? '⏳ Mutating Audio Physics...' : '🎵 Generate Mutated Audio'}
                      </button>
                    ) : (
                      <div className="mutated-audio-player" style={{background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px'}}>
                        <h4 style={{color: 'var(--primary)', marginBottom: '15px', marginTop: 0}}>🎧 Your Mutated Hit Song</h4>
                        <audio controls src={mutatedAudioUrl} style={{width: '100%'}} />
                        <button 
                          className="btn secondary" 
                          onClick={generateMutatedAudio} 
                          disabled={mutatingAudio}
                          style={{marginTop: '15px'}}
                        >
                          {mutatingAudio ? '⏳ Re-Mutating...' : '🔄 Re-Generate Audio'}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="result-actions" style={{display: 'flex', gap: '15px', justifyContent: 'center', marginTop: '30px', borderTop: '1px solid #eee', paddingTop: '20px'}}>
                <button className="btn secondary" onClick={handleDownloadReport}>Download PDF Report</button>
                <button className="btn primary" onClick={resetForm}>Test Another Song</button>
              </div>
            </div>
        )}
      </div>
      
      {/* Hidden PDF Template */}
      <div className="pdf-hidden-wrapper">
        <ReportTemplate result={result} ref={reportRef} />
      </div>
    </div>
  )
}
