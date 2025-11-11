import React, {useState} from 'react'

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

// Simple frontend fallback predictor (weights inspired by feature importance)
function fallbackPredict(features){
  // normalize some ranges
  const f = {...features}
  f.loudness = (f.loudness + 60)/60 // map -60..0 to 0..1
  f.tempo = Math.min(200, Math.max(30, f.tempo))/200
  f.duration_ms = Math.min(300000, Math.max(30000, f.duration_ms))/300000

  // weights (toy model)
  const weights = {
    danceability: 1.8,
    energy: 1.6,
    valence: 1.2,
    acousticness: -0.8,
    speechiness: -0.4,
    instrumentalness: -0.6,
    liveness: -0.2,
    loudness: 0.9,
    tempo: 0.3,
    duration_ms: 0.2
  }

  let score = 0
  score += (f.danceability || 0) * weights.danceability
  score += (f.energy || 0) * weights.energy
  score += (f.valence || 0) * weights.valence
  score += (f.acousticness || 0) * weights.acousticness
  score += (f.speechiness || 0) * weights.speechiness
  score += (f.instrumentalness || 0) * weights.instrumentalness
  score += (f.liveness || 0) * weights.liveness
  score += (f.loudness || 0) * weights.loudness
  score += (f.tempo || 0) * weights.tempo
  score += (f.duration_ms || 0) * weights.duration_ms

  // squash to probability 0..1
  const prob = 1/(1+Math.exp(- (score - 1.5)))
  const confidence = Math.min(0.99, Math.max(0.4, Math.abs(score)/4))
  return {hit_probability: prob, confidence}
}

export default function PredictorForm({onResult}){
  const [form, setForm] = useState(DEFAULTS)
  const [last, setLast] = useState(null)
  const [loading, setLoading] = useState(false)

  function update(k,v){
    setForm(f => ({...f, [k]: v}))
  }

  async function handleSubmit(e){
    e.preventDefault()
    setLoading(true)
    try{
      // If API exists, try calling /api/predict (not present by default)
      let result = null
      try{
        const resp = await fetch('/api/predict',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(form)})
        if(resp.ok){
          result = await resp.json()
        }
      }catch(err){/* ignore - fallback used */}

      if(!result){
        result = fallbackPredict(form)
      }

      const payload = {...result, features: form}
      setLast(payload)
      onResult && onResult(payload)

      // small celebration when high probability
      if(payload.hit_probability > 0.75) {
        fireConfetti()
      }

    }finally{
      setLoading(false)
    }
  }

  function fireConfetti(){
    // simple DOM confetti using emojis
    const c = document.createElement('div')
    c.className = 'confetti'
    c.innerText = '✨🎉'
    document.body.appendChild(c)
    setTimeout(()=>document.body.removeChild(c),1200)
  }

  return (
    <div className="card">
      <h2>Design your track</h2>
      <form onSubmit={handleSubmit} className="form-grid">
        <label>Danceability<input type="range" min="0" max="1" step="0.01" value={form.danceability} onChange={e=>update('danceability',Number(e.target.value))} /></label>
        <label>Energy<input type="range" min="0" max="1" step="0.01" value={form.energy} onChange={e=>update('energy',Number(e.target.value))} /></label>
        <label>Valence<input type="range" min="0" max="1" step="0.01" value={form.valence} onChange={e=>update('valence',Number(e.target.value))} /></label>
        <label>Acousticness<input type="range" min="0" max="1" step="0.01" value={form.acousticness} onChange={e=>update('acousticness',Number(e.target.value))} /></label>
        <label>Speechiness<input type="range" min="0" max="1" step="0.01" value={form.speechiness} onChange={e=>update('speechiness',Number(e.target.value))} /></label>
        <label>Instrumentalness<input type="range" min="0" max="1" step="0.01" value={form.instrumentalness} onChange={e=>update('instrumentalness',Number(e.target.value))} /></label>
        <label>Liveness<input type="range" min="0" max="1" step="0.01" value={form.liveness} onChange={e=>update('liveness',Number(e.target.value))} /></label>
        <label>Loudness<input type="range" min="-60" max="0" step="0.1" value={form.loudness} onChange={e=>update('loudness',Number(e.target.value))} /></label>
        <label>Tempo<input type="number" min="30" max="250" value={form.tempo} onChange={e=>update('tempo',Number(e.target.value))} /></label>
        <label>Duration (ms)<input type="number" min="30000" max="600000" value={form.duration_ms} onChange={e=>update('duration_ms',Number(e.target.value))} /></label>

        <div className="actions">
          <button type="submit" className="btn primary" disabled={loading}>{loading? 'Analyzing...':'Predict Virality'}</button>
          <button type="button" className="btn" onClick={()=>setForm(DEFAULTS)}>Reset</button>
        </div>
      </form>

      {last && (
        <div className="result">
          <h3>Prediction</h3>
          <p>Hit Probability: <strong>{(last.hit_probability*100).toFixed(1)}%</strong></p>
          <p>Confidence: {(last.confidence*100).toFixed(0)}%</p>
        </div>
      )}      
    </div>
  )
}