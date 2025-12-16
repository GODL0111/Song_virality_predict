import React, {useState, useEffect} from 'react'

const BACKEND_URL = typeof window !== 'undefined' && window.BACKEND_URL ? window.BACKEND_URL : 'http://localhost:5001'

const DEFAULTS = {
  danceability: 0.65,
  energy: 0.72,
  key: 5,
  loudness: -6.5,
  mode: 1,
  speechiness: 0.08,
  acousticness: 0.25,
  instrumentalness: 0.05,
  liveness: 0.15,
  valence: 0.58,
  tempo: 125,
  duration_ms: 210000
}

export default function PredictorForm({onResult}){
  const [songName, setSongName] = useState('')
  const [form, setForm] = useState(DEFAULTS)
  const [last, setLast] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendStatus, setBackendStatus] = useState(null) // null=checking, true=available, false=unavailable

  // Check backend availability on component mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const resp = await fetch(`${BACKEND_URL}/api/health`, { method: 'GET' })
        setBackendStatus(resp.ok)
      } catch (err) {
        setBackendStatus(false)
      }
    }
    checkBackend()
  }, [])

  function update(k,v){
    setForm(f => ({...f, [k]: v}))
  }

  async function handleSubmit(e){
    e.preventDefault()
    
    // Validate song name is not empty
    if(!songName.trim()){
      setError('Song name is required')
      return
    }
    
    setLoading(true)
    setError(null)
    try{
      // Backend is required for predictions
      const resp = await fetch(`${BACKEND_URL}/api/predict`,{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(form)
      })
      
      if(!resp.ok){
        throw new Error('Backend service is not available. Please start the backend server to make predictions.')
      }
      
      const result = await resp.json()
      
      // Get improvement suggestions
      let suggestions = []
      try {
        const sugResp = await fetch(`${BACKEND_URL}/api/suggest-improvements`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(form)
        })
        if (sugResp.ok) {
          const sugData = await sugResp.json()
          suggestions = sugData.suggestions || []
        }
      } catch (err) {
        console.error('Error fetching suggestions:', err)
      }
      
      const payload = {...result, features: form, songName: songName.trim(), suggestions}
      setLast(payload)
      onResult && onResult(payload)

      // celebration when high probability
      if(payload.hit_probability > 0.75) {
        fireConfetti()
      }

    }catch(err){
      setError(err.message || 'Failed to connect to prediction service. Make sure the backend is running on port 5001.')
      console.error('Prediction error:', err)
    }finally{
      setLoading(false)
    }
  }

  function fireConfetti(){
    try {
      const c = document.createElement('div')
      c.className = 'confetti'
      c.innerText = '✨'
      c.style.position = 'fixed'
      c.style.top = '50%'
      c.style.left = '50%'
      c.style.fontSize = '48px'
      c.style.zIndex = '9999'
      c.style.pointerEvents = 'none'
      c.style.animation = 'fadeOut 1s ease-out'
      document.body.appendChild(c)
      setTimeout(() => {
        if (document.body.contains(c)) {
          document.body.removeChild(c)
        }
      }, 1200)
    } catch (err) {
      console.error('Confetti error:', err)
    }
  }

  return (
    <div className="card">
      <h2>Design your track</h2>
      
      {backendStatus === false && (
        <div className="result error" style={{marginBottom: '24px', marginTop: '-20px'}}>
          <p style={{margin: 0}}>⚠️ <strong>Backend service not running</strong> - Using local prediction model. For accurate ML-based predictions, start the backend server.</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="form-grid">
        <div className="form-group full-width">
          <label>Song Name <span className="required">*</span></label>
          <input 
            type="text" 
            placeholder="Enter song name"
            value={songName}
            onChange={e => setSongName(e.target.value)}
            maxLength="100"
            className="song-name-input"
            required
          />
        </div>

        <label>Danceability <span className="value">{form.danceability.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={form.danceability} onChange={e=>update('danceability',Number(e.target.value))} /></label>
        <label>Energy <span className="value">{form.energy.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={form.energy} onChange={e=>update('energy',Number(e.target.value))} /></label>
        <label>Valence <span className="value">{form.valence.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={form.valence} onChange={e=>update('valence',Number(e.target.value))} /></label>
        <label>Acousticness <span className="value">{form.acousticness.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={form.acousticness} onChange={e=>update('acousticness',Number(e.target.value))} /></label>
        <label>Speechiness <span className="value">{form.speechiness.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={form.speechiness} onChange={e=>update('speechiness',Number(e.target.value))} /></label>
        <label>Instrumentalness <span className="value">{form.instrumentalness.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={form.instrumentalness} onChange={e=>update('instrumentalness',Number(e.target.value))} /></label>
        <label>Liveness <span className="value">{form.liveness.toFixed(2)}</span><input type="range" min="0" max="1" step="0.01" value={form.liveness} onChange={e=>update('liveness',Number(e.target.value))} /></label>
        <label>Loudness <span className="value">{form.loudness.toFixed(1)}</span><input type="range" min="-60" max="0" step="0.1" value={form.loudness} onChange={e=>update('loudness',Number(e.target.value))} /></label>
        <label>Tempo <span className="value">{form.tempo}</span><input type="number" min="30" max="250" value={form.tempo} onChange={e=>update('tempo',Number(e.target.value))} /></label>
        <label>Duration (ms) <span className="value">{form.duration_ms}</span><input type="number" min="30000" max="600000" value={form.duration_ms} onChange={e=>update('duration_ms',Number(e.target.value))} /></label>

        <div className="actions">
          <button type="submit" className="btn primary" disabled={loading || !songName.trim()}>{loading? 'Analyzing...':'Predict Virality'}</button>
          <button type="button" className="btn" onClick={()=>{setForm(DEFAULTS); setSongName('')}}>Reset</button>
        </div>
      </form>

      {error && (
        <div className="result error">
          <p>Error: {error}</p>
        </div>
      )}

      {last && (
        <div className="result">
          <h3>Prediction for "{last.songName}"</h3>
          <p>Hit Probability: <strong>{(last.hit_probability*100).toFixed(1)}%</strong></p>
          <p>Confidence: {(last.confidence*100).toFixed(0)}%</p>
          <p>Status: <strong>{last.hit_probability > 0.5 ? '🎵 HIT' : '📊 MISS'}</strong></p>
          
          {last.suggestions && last.suggestions.length > 0 && (
            <div style={{marginTop: '20px', paddingTop: '20px', borderTop: '1px solid rgba(255,255,255,0.2)'}}>
              <h4>💡 Suggestions to Improve</h4>
              <div style={{display: 'grid', gap: '12px'}}>
                {last.suggestions.slice(0, 3).map((sug, idx) => (
                  <div key={idx} style={{
                    background: 'rgba(255,255,255,0.05)',
                    padding: '12px',
                    borderRadius: '8px',
                    borderLeft: '3px solid #4CAF50'
                  }}>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px'}}>
                      <strong>{sug.feature}</strong>
                      <span style={{fontSize: '12px', color: '#4CAF50', fontWeight: 'bold'}}>
                        {sug.direction === 'INCREASE' ? '⬆' : sug.direction === 'DECREASE' ? '⬇' : '→'} {(sug.improvement_percent || 0).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{fontSize: '13px', color: '#aaa', marginBottom: '4px'}}>
                      Change from {sug.current.toFixed(2)} → {sug.suggested.toFixed(2)}
                    </div>
                    <div style={{fontSize: '12px', color: '#888'}}>
                      New probability: {(sug.new_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}      
    </div>
  )
}
