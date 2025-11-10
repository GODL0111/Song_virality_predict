import React from 'react'

function Badge({name, unlocked}){
  return (
    <div className={"badge " + (unlocked? 'unlocked':'locked')}>
      <div className="badge-emoji">{unlocked? '🏆':'🔒'}</div>
      <div className="badge-name">{name}</div>
    </div>
  )
}

export default function GameDashboard({score, logs}){
  const badges = [
    {name:'First Hit', min:100},
    {name:'Rising Star', min:500},
    {name:'Chart Topper', min:1500},
    {name:'Legend', min:5000}    
  ]

  return (
    <aside className="card">
      <h2>Player Hub</h2>
      <div className="score">Score: <strong>{score}</strong></div>

      <div className="badges">
        {badges.map(b=> <Badge key={b.name} name={b.name} unlocked={score>=b.min} />)}
      </div>

      <div className="log">
        <h3>Recent Attempts</h3>
        {logs.length===0? <p>No attempts yet — predict a song to start earning points.</p> : (
          <ul>
            {logs.map((l,idx)=>(<li key={idx}><strong>{(l.probability*100).toFixed(1)}%</strong> — +{l.points}pts <span className="muted">{new Date(l.time).toLocaleString()}</span></li>))}
          </ul>
        )}     
      </div>

      <div className="tips card-quiet">
        Pro Tip: Increase danceability & energy together — it tends to boost virality!
      </div>
    </aside>
  )
}