import React, { useState, useEffect } from 'react'

export default function App(){
  const [mounted, setMounted] = useState(false)
  const [Layout, setLayout] = useState(null)
  const [error, setError] = useState(null)
  const [score, setScore] = useState(() => {
    try {
      return Number(localStorage.getItem('sv_score') || 0)
    } catch {
      return 0
    }
  })
  const [logs, setLogs] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('sv_logs') || '[]')
    } catch {
      return []
    }
  })

  // Lazy load Layout component
  useEffect(() => {
    import('./components/Layout').then(mod => {
      setLayout(() => mod.default)
      setMounted(true)
    }).catch(err => {
      console.error('Failed to load Layout:', err)
      setError(err.message)
      setMounted(true)
    })
  }, [])

  useEffect(()=>{
    try {
      localStorage.setItem('sv_score', score)
    } catch (e) {
      console.error('Failed to save score:', e)
    }
  },[score])

  useEffect(()=>{
    try {
      localStorage.setItem('sv_logs', JSON.stringify(logs))
    } catch (e) {
      console.error('Failed to save logs:', e)
    }
  },[logs])

  function handleResult({hit_probability, confidence, features, songName}){
    const points = Math.round(hit_probability * 100)
    const bonus = confidence > 0.75 ? 25 : confidence > 0.5 ? 10 : 0
    const total = points + bonus
    setScore(s => s + total)
    const entry = {time: new Date().toISOString(), probability: hit_probability, confidence, points: total, features, songName}
    setLogs(l => [entry, ...l].slice(0,50))
  }

  if (!mounted) {
    return <div style={{color: '#fff', padding: '40px', textAlign: 'center', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f1724'}}>
      <div>
        <div style={{fontSize: '24px', marginBottom: '20px'}}>🎵 Loading SoundViral...</div>
        <div style={{width: '50px', height: '50px', border: '4px solid rgba(255,122,182,0.2)', borderTopColor: '#ff7ab6', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto'}}></div>
      </div>
    </div>
  }

  if (error) {
    return <div style={{color: '#ff6b6b', padding: '40px', textAlign: 'center', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f1724'}}>
      <div>
        <div style={{fontSize: '24px', marginBottom: '20px'}}>⚠️ Error Loading App</div>
        <div style={{fontFamily: 'monospace', background: 'rgba(255,0,0,0.1)', padding: '20px', borderRadius: '8px'}}>{error}</div>
      </div>
    </div>
  }

  if (!Layout) {
    return <div style={{color: '#fff', padding: '40px', textAlign: 'center', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0f1724'}}>
      <div>
        <div style={{fontSize: '24px', marginBottom: '20px'}}>🎵 Loading Layout...</div>
      </div>
    </div>
  }

  return (
    <Layout score={score} logs={logs} onResult={handleResult} />
  )
}