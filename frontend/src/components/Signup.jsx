import React, { useState } from 'react'
import { GoogleLogin } from '@react-oauth/google'
import { jwtDecode } from 'jwt-decode'
import './Auth.css'

const BACKEND_URL = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? '' : 'http://localhost:5000');

export default function Signup({ onSignup, onSwitchToLogin, isDarkMode, onToggleTheme }) {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
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
    if (formData.password !== formData.confirmPassword) {
       setError("Passwords do not match");
       return;
    }
    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${BACKEND_URL}/api/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
           username: formData.username,
           email: formData.email,
           password: formData.password
        })
      });
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Signup failed');
      }
      
      if (onSignup) onSignup(data.user);
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
      
      if (onSignup) onSignup(data.user);
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
            <h2>Create Account</h2>
            <p>Join to start predicting viral hits</p>
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
              <label htmlFor="username">Username</label>
            </div>

            <div className="form-group floating-label">
              <input
                id="email"
                name="email"
                type="email"
                placeholder=" "
                value={formData.email}
                onChange={handleChange}
                required
                disabled={loading}
              />
              <label htmlFor="email">Email Address</label>
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

            <div className="form-group floating-label">
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                placeholder=" "
                value={formData.confirmPassword}
                onChange={handleChange}
                required
                disabled={loading}
              />
              <label htmlFor="confirmPassword">Confirm Password</label>
            </div>
            
            <button
              type="submit"
              className="auth-button primary shine-effect"
              disabled={loading || !formData.username || !formData.password || !formData.email || !formData.confirmPassword}
            >
              {loading ? <div className="loading-spinner"></div> : 'Sign Up'}
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
               text="signup_with"
             />
          </div>
          
          <div className="auth-footer">
            <p>
              Already have an account?{' '}
              <button className="switch-auth" onClick={onSwitchToLogin} disabled={loading}>
                Log in here
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
