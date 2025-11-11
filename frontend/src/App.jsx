import React, { useState, useEffect } from 'react'
import PredictorForm from './components/PredictorForm'
import GameDashboard from './components/GameDashboard'

export default function App(){
  const [score, setScore] = useState(() => Number(localStorage.getItem('sv_score') || 0))
  const [logs, setLogs] = useState(() => JSON.parse(localStorage.getItem('sv_logs') || '[]'))

  useEffect(()=>{
    localStorage.setItem('sv_score', score)
  },[score])

  useEffect(()=>{
    localStorage.setItem('sv_logs', JSON.stringify(logs))
  },[logs])

  function handleResult({hit_probability, confidence, features}){
    const points = Math.round(hit_probability * 100)
    const bonus = confidence > 0.75 ? 25 : confidence > 0.5 ? 10 : 0
    const total = points + bonus
    setScore(s => s + total)
    const entry = {time: new Date().toISOString(), probability: hit_probability, confidence, points: total, features}
    setLogs(l => [entry, ...l].slice(0,50))
  }

  return (
    <div className="app-root">
      <header className="hero">
        <h1>Song Virality Arena</h1>
        <p>Design a track, predict its virality, earn points & unlock badges.</p>
      </header>

      <main className="container">
        <section className="left">
          <PredictorForm onResult={handleResult} />
        </section>
        <aside className="right">
          <GameDashboard score={score} logs={logs} />
        </aside>
      </main>

      <footer className="footer">Built for local & Vercel deployment — frontend only fallback included.</footer>
    </div>
  )
}