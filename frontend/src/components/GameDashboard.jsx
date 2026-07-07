import React, { useState } from 'react'

function Badge({name, unlocked}){
  return (
    <div className={"badge " + (unlocked? 'unlocked':'locked')}>
      <div className="badge-emoji">{unlocked? '🏆':'🔒'}</div>
      <div className="badge-name">{name}</div>
    </div>
  )
}

export default function GameDashboard({score, logs}){
  const [showAllAttempts, setShowAllAttempts] = useState(false)

  const badges = [
    {name:'First Hit', min:100},
    {name:'Rising Star', min:500},
    {name:'Chart Topper', min:1500},
    {name:'Legend', min:5000}    
  ]

  // Show last 5 or all based on toggle
  const displayLogs = showAllAttempts ? logs : logs.slice(0, 5)

  return (
    <aside className="card compact-hub">
      <h2>🎮 Player Hub</h2>
      <div className="score">Score: <strong>{score}</strong></div>

      <div className="badges compact">
        {badges.map(b=> <Badge key={b.name} name={b.name} unlocked={score>=b.min} />)}
      </div>

      <div className="log compact">
        <div className="log-header">
          <h3>Recent Attempts</h3>
          {logs.length > 5 && (
            <button 
              className="toggle-all-btn"
              onClick={() => setShowAllAttempts(!showAllAttempts)}
            >
              {showAllAttempts ? 'Show Less ▲' : 'Show All ▼'}
            </button>
          )}
        </div>
        {displayLogs.length===0? <p className="empty-state">No attempts yet</p> : (
          <ul>
            {displayLogs.map((l,idx)=>(
              <li key={idx} className="attempt-log">
                <div className="song-info">
                  <div className="song-name">♪ {l.songName || 'Untitled'}</div>
                  <div className="song-score"><strong>{(l.probability*100).toFixed(0)}%</strong></div>
                </div>
                <div className="attempt-points">
                  <span className="points">+{l.points}</span>
                </div>
              </li>
            ))}
          </ul>
        )}     
      </div>

      <div className="tips card-quiet small">
        ⚡ High danceability + energy = viral potential!
      </div>
      
      <div className="tips card-quiet small" style={{marginTop: '8px', fontSize: '11px', opacity: 0.7}}>
        🤖 Model trained on 176K+ songs
      </div>
    </aside>
  )
}