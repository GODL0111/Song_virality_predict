import React, { useState } from 'react'
import { GoogleLogin } from '@react-oauth/google'
import { jwtDecode } from 'jwt-decode'
import './Auth.css'

const BACKEND_URL = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? '' : 'http://localhost:5000');

export default function Login({ onLogin, onSwitchToSignup, isDarkMode, onToggleTheme }) {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    if (error) setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${BACKEND_URL}/api/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Login failed');
      }
      
      if (onLogin) onLogin(data.user);
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleGoogleSuccess = async (credentialResponse) => {
    setLoading(true);
    setError('');
    try {
      const decoded = jwtDecode(credentialResponse.credential);
      const response = await fetch(`${BACKEND_URL}/api/google-login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: decoded.email,
          name: decoded.name,
          google_id: decoded.sub
        })
      });
      
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Google login failed');
      
      if (onLogin) onLogin(data.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`auth-container ${isDarkMode ? 'dark-theme' : 'light-theme'}`}>
      <div className="auth-background">
        <div className="gradient-orbs">
          <div className="orb orb-1"></div>
          <div className="orb orb-2"></div>
          <div className="orb orb-3"></div>
        </div>
        <div className="glass-shapes">
           <div className="shape shape-circle"></div>
           <div className="shape shape-square"></div>
        </div>
      </div>

      <div className="auth-content">
        <div className="auth-card glassmorphism">
          <div className="auth-header">
            <div className="header-top">
              <div className="logo">
                <div className="logo-icon">🎵</div>
                <h1>SoundViral</h1>
              </div>
              <button className="theme-toggle" onClick={onToggleTheme}>
                <div className={`toggle-slider ${isDarkMode ? 'dark' : 'light'}`}>
                  <div className="toggle-icon">{isDarkMode ? '🌙' : '☀️'}</div>
                </div>
              </button>
            </div>
            <h2>Welcome Back</h2>
            <p>Enter your credentials to access your dashboard</p>
          </div>
          
          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}
          
          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="form-group floating-label">
              <input
                id="username"
                name="username"
                type="text"
                placeholder=" "
                value={formData.username}
                onChange={handleChange}
                required
                disabled={loading}
              />
              <label htmlFor="username">Email or Username</label>
            </div>
            
            <div className="form-group floating-label">
              <input
                id="password"
                name="password"
                type="password"
                placeholder=" "
                value={formData.password}
                onChange={handleChange}
                required
                disabled={loading}
              />
              <label htmlFor="password">Password</label>
            </div>
            
            <button
              type="submit"
              className="auth-button primary shine-effect"
              disabled={loading || !formData.username || !formData.password}
            >
              {loading ? <div className="loading-spinner"></div> : 'Sign In'}
            </button>
          </form>
          
          <div className="divider">
             <span>or continue with</span>
          </div>

          <div className="google-auth-wrapper">
             <GoogleLogin
               onSuccess={handleGoogleSuccess}
               onError={() => setError('Google authentication failed')}
               theme={isDarkMode ? 'filled_black' : 'outline'}
               shape="pill"
               width="100%"
               size="large"
               text="signin_with"
             />
          </div>
          
          <div className="auth-footer">
            <p>
              Don't have an account?{' '}
              <button className="switch-auth" onClick={onSwitchToSignup} disabled={loading}>
                Create an account
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
