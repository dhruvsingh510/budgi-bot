#!/usr/bin/env python3
"""
Personal Finance Bot System Launcher

This script helps you start both the backend API server and frontend Streamlit app.
"""

import subprocess
import sys
import time
import os
import signal
import requests
from pathlib import Path


def check_port_available(port):
    """Check if a port is available."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("localhost", port))
            return result != 0
    except Exception:
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import fastapi
        import uvicorn

        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install streamlit fastapi uvicorn")
        return False


def start_backend():
    """Start the backend API server."""
    backend_dir = Path(__file__).parent / "backend"
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None

    print("🚀 Starting backend API server...")

    # Check if port 8080 is available
    if not check_port_available(8080):
        print("❌ Port 8080 is already in use!")
        print("💡 Stop the service using port 8080 or use a different port")
        return None

    try:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give it time to start
        time.sleep(3)

        # Check if it's running
        try:
            response = requests.get("http://localhost:8080/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend API server started successfully!")
                print("🌐 API available at: http://localhost:8080")
                print("📚 Documentation at: http://localhost:8080/docs")
                return process
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException:
            print("❌ Backend failed to start or is not responding")
            return None

    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None


def start_frontend():
    """Start the frontend Streamlit app."""
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None

    print("🎨 Starting frontend Streamlit app...")

    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give it time to start
        time.sleep(5)

        print("✅ Frontend Streamlit app started!")
        print("🎯 Frontend available at: http://localhost:8501")
        return process

    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None


def main():
    """Main launcher function."""
    print("🤖 Personal Finance Bot System Launcher")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    backend_process = None
    frontend_process = None

    try:
        # Start backend
        backend_process = start_backend()
        if not backend_process:
            print("❌ Failed to start backend. Exiting.")
            sys.exit(1)

        # Start frontend
        frontend_process = start_frontend()
        if not frontend_process:
            print("❌ Failed to start frontend. Stopping backend.")
            if backend_process:
                backend_process.terminate()
            sys.exit(1)

        print("\n🎉 System started successfully!")
        print("📱 Frontend: http://localhost:8501")
        print("🔗 Backend API: http://localhost:8080")
        print("📖 API Docs: http://localhost:8080/docs")
        print("\n💡 Press Ctrl+C to stop both services")

        # Wait for user to stop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping services...")

    finally:
        # Clean shutdown
        if frontend_process:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
                print("✅ Frontend stopped")
            except subprocess.TimeoutExpired:
                frontend_process.kill()
                print("🔪 Frontend force stopped")

        if backend_process:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
                print("✅ Backend stopped")
            except subprocess.TimeoutExpired:
                backend_process.kill()
                print("🔪 Backend force stopped")

        print("👋 System shutdown complete!")


if __name__ == "__main__":
    main()
