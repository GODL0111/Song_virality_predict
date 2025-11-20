import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

const container = document.getElementById('root')
if (!container) {
  console.error('Root container not found!')
} else {
  try {
    const root = createRoot(container)
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>
    )
  } catch (error) {
    console.error('Failed to render app:', error)
    container.innerHTML = `<div style="color: red; padding: 20px; font-family: monospace;">Error: ${error.message}</div>`
  }
}