import React, { useState, useEffect } from 'react'
import './Layout.css'

// Lazy load components to avoid import errors
const PredictorForm = React.lazy(() => import('./PredictorForm'))
const LiveSongTest = React.lazy(() => import('./LiveSongTest'))
const Recommendations = React.lazy(() => import('./Recommendations'))
const Creators = React.lazy(() => import('./Creators'))
const GameDashboard = React.lazy(() => import('./GameDashboard'))
const LiveRecording = React.lazy(() => import('./LiveRecording'))

export default function Layout({ score, logs, onResult, user, onLogout }) {
  const [currentPage, setCurrentPage] = useState('home')
  const [menuOpen, setMenuOpen] = useState(false)
  const [isDarkTheme, setIsDarkTheme] = useState(true)

  useEffect(() => {
    const savedTheme = localStorage.getItem('theme')
    if (savedTheme) {
      setIsDarkTheme(savedTheme === 'dark')
      document.documentElement.setAttribute('data-theme', savedTheme)
    } else {
      document.documentElement.setAttribute('data-theme', 'dark')
    }
  }, [])

  const toggleTheme = () => {
    const newTheme = !isDarkTheme
    setIsDarkTheme(newTheme)
    const themeValue = newTheme ? 'dark' : 'light'
    localStorage.setItem('theme', themeValue)
    document.documentElement.setAttribute('data-theme', themeValue)
  }

  const toggleMenu = () => setMenuOpen(!menuOpen)
  const closeMenu = () => setMenuOpen(false)

  const navigate = (page) => {
    setCurrentPage(page)
    closeMenu()
  }

  return (
    <div className="layout">
      {/* Animated 3D Background */}
      <div className="background">
        <div className="music-symbol">♪</div>
        <div className="gradient-orbs">
          <div className="orb orb-1"></div>
          <div className="orb orb-2"></div>
          <div className="orb orb-3"></div>
        </div>
      </div>

      {/* Hamburger Menu */}
      <button 
        className={`hamburger ${menuOpen ? 'active' : ''}`}
        onClick={toggleMenu}
        aria-label="Toggle menu"
      >
        <span></span>
        <span></span>
        <span></span>
      </button>

      {/* Theme Toggle Button */}
      <button 
        className="theme-toggle"
        onClick={toggleTheme}
        aria-label="Toggle theme"
      >
        <span className="theme-icon">{isDarkTheme ? '☀️' : '🌙'}</span>
      </button>

      {/* Sidebar Menu */}
      <nav className={`sidebar ${menuOpen ? 'open' : ''}`}>
        <div className="menu-header">
          <h1>SoundViral</h1>
          {user && (
            <div className="user-info">
              <div className="user-avatar">
                {user.username?.charAt(user.username.length - 1) || 'U'}
              </div>
              <div className="user-details">
                <span className="user-name">{user.name || user.username}</span>
                <span className="user-id">@{user.username}</span>
              </div>
            </div>
          )}
        </div>
        <ul className="menu-items">
          <li>
            <button 
              className={`menu-link ${currentPage === 'home' ? 'active' : ''}`}
              onClick={() => navigate('home')}
            >
              Home
            </button>
          </li>
          <li>
            <button 
              className={`menu-link ${currentPage === 'static' ? 'active' : ''}`}
              onClick={() => navigate('static')}
            >
              Static Viral Check
            </button>
          </li>
          <li>
            <button 
              className={`menu-link ${currentPage === 'live' ? 'active' : ''}`}
              onClick={() => navigate('live')}
            >
              Live Song Test
            </button>
          </li>
          <li>
            <button 
              className={`menu-link ${currentPage === 'record' ? 'active' : ''}`}
              onClick={() => navigate('record')}
            >
              Live Recording
            </button>
          </li>
          <li>
            <button 
              className={`menu-link ${currentPage === 'recommend' ? 'active' : ''}`}
              onClick={() => navigate('recommend')}
            >
              Recommendations
            </button>
          </li>
          <li>
            <button 
              className={`menu-link ${currentPage === 'creators' ? 'active' : ''}`}
              onClick={() => navigate('creators')}
            >
              Creators
            </button>
          </li>
        </ul>
        <div className="menu-footer">
          {user && (
            <button className="logout-btn" onClick={onLogout}>
              <span>🚪</span>
              Logout
            </button>
          )}
          <p>Made with music passion</p>
        </div>
      </nav>

      {/* Overlay for menu */}
      {menuOpen && (
        <div className="menu-overlay" onClick={closeMenu}></div>
      )}

      {/* Main Content */}
      <main className="main-content">
        {currentPage === 'home' && (
          <div className="home-page">
            <div className="welcome-section">
              <h1 className="title-3d">SoundViral</h1>
              <p className="subtitle">Predict Your Song's Potential</p>
              <p className="description">
                Use AI-powered machine learning to analyze your song's potential to go viral.
                Our ensemble model is trained on 176,000+ tracks from multiple Spotify datasets.
              </p>
              <div className="quick-buttons">
                <button className="btn primary large" onClick={() => navigate('static')}>
                  Start Analyzing →
                </button>
              </div>
            </div>
            <aside className="game-dashboard-container">
              <React.Suspense fallback={<div>Loading dashboard...</div>}>
                <GameDashboard score={score} logs={logs} />
              </React.Suspense>
            </aside>
          </div>
        )}

        {currentPage === 'static' && (
          <div className="static-page">
            <div className="page-header">
              <h2>Static Viral Check</h2>
              <p>Analyze song features for viral potential</p>
            </div>
            <div className="static-content">
              <div className="static-intro">
                <p>Fine-tune your song's audio features to maximize its viral potential.</p>
              </div>
              <div className="form-container">
                <React.Suspense fallback={<div>Loading form...</div>}>
                  <PredictorForm onResult={onResult} />
                </React.Suspense>
              </div>
            </div>
          </div>
        )}

        {currentPage === 'live' && (
          <div className="live-page">
            <React.Suspense fallback={<div>Loading...</div>}>
              <LiveSongTest />
            </React.Suspense>
          </div>
        )}

        {currentPage === 'record' && (
          <div className="record-page">
            <React.Suspense fallback={<div>Loading...</div>}>
              <LiveRecording />
            </React.Suspense>
          </div>
        )}

        {currentPage === 'recommend' && (
          <div className="recommend-page">
            <React.Suspense fallback={<div>Loading...</div>}>
              <Recommendations />
            </React.Suspense>
          </div>
        )}

        {currentPage === 'creators' && (
          <React.Suspense fallback={<div>Loading...</div>}>
            <Creators />
          </React.Suspense>
        )}
      </main>

    </div>
  )
}