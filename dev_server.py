"""
Development server that runs both backend API and frontend together
Usage: python dev_server.py
"""
import subprocess
import os
import sys
import time
import signal
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / 'backend'
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

def run_backend():
    """Start the Flask API backend"""
    print("🚀 Starting Backend API (Flask)...")
    os.chdir(BACKEND_DIR)
    env = os.environ.copy()
    env['FLASK_ENV'] = 'development'
    env['FLASK_PORT'] = '5000'
    
    cmd = [sys.executable, '-m', 'api.server']
    return subprocess.Popen(cmd, env=env)

def run_frontend():
    """Start the Vite dev server"""
    print("⚡ Starting Frontend (Vite)...")
    os.chdir(FRONTEND_DIR)
    cmd = ['npm', 'run', 'dev']
    return subprocess.Popen(cmd)

def main():
    """Run both servers concurrently"""
    print("🎵 Song Virality Predict - Development Server")
    print("=" * 50)
    
    # Check if npm is available
    try:
        subprocess.run(['npm', '--version'], capture_output=True, check=True)
    except:
        print("✗ npm not found. Please install Node.js and npm.")
        sys.exit(1)
    
    # Start both servers
    backend_proc = None
    frontend_proc = None
    
    try:
        backend_proc = run_backend()
        time.sleep(2)  # Give backend time to start
        
        frontend_proc = run_frontend()
        
        print("\n" + "=" * 50)
        print("✓ Both servers started!")
        print("  Backend API: http://localhost:5000")
        print("  Frontend: http://localhost:5173")
        print("=" * 50)
        print("\nPress Ctrl+C to stop both servers...")
        
        # Wait for both processes
        if backend_proc:
            backend_proc.wait()
        if frontend_proc:
            frontend_proc.wait()
    
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        
        # Terminate processes
        if frontend_proc:
            frontend_proc.terminate()
            try:
                frontend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_proc.kill()
        
        if backend_proc:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()
        
        print("✓ Servers stopped")
        sys.exit(0)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
