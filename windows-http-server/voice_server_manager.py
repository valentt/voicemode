#!/usr/bin/env python3
"""
Voice Server Manager - Start/stop/status for voice server

Usage:
    python scripts/voice_server_manager.py start [--port 8765]
    python scripts/voice_server_manager.py stop
    python scripts/voice_server_manager.py status
    python scripts/voice_server_manager.py restart
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

DEFAULT_PORT = 8765
PID_FILE = Path(__file__).parent.parent / "voice_server.pid"
LOG_FILE = Path(__file__).parent.parent / "voice_server.log"


def get_server_url(port: int = DEFAULT_PORT) -> str:
    return f"http://127.0.0.1:{port}"


def is_server_running(port: int = DEFAULT_PORT) -> bool:
    """Check if server is responding"""
    try:
        response = requests.get(f"{get_server_url(port)}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_pid() -> int | None:
    """Get PID from file"""
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except:
            pass
    return None


def save_pid(pid: int):
    """Save PID to file"""
    PID_FILE.write_text(str(pid))


def remove_pid():
    """Remove PID file"""
    if PID_FILE.exists():
        PID_FILE.unlink()


def start_server(port: int = DEFAULT_PORT):
    """Start the voice server"""
    if is_server_running(port):
        print(f"Voice server is already running on port {port}")
        return True

    # Find voice_server.py
    server_script = Path(__file__).parent / "voice_server.py"
    if not server_script.exists():
        print(f"ERROR: {server_script} not found")
        return False

    # Start server in background
    print(f"Starting voice server on port {port}...")

    # Use pythonw on Windows for background process
    python_exe = sys.executable

    # Open log file
    log_file = open(LOG_FILE, "w")

    # Start process
    process = subprocess.Popen(
        [python_exe, str(server_script), "--port", str(port)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    )

    save_pid(process.pid)

    # Wait for server to start
    for i in range(10):
        time.sleep(0.5)
        if is_server_running(port):
            print(f"Voice server started (PID: {process.pid})")
            print(f"Log file: {LOG_FILE}")
            return True

    print("ERROR: Server failed to start. Check log file.")
    return False


def stop_server():
    """Stop the voice server"""
    pid = get_pid()

    if pid is None:
        print("No PID file found")
        # Try to find by port anyway
        if is_server_running():
            print("But server is running. Try killing manually.")
        return False

    print(f"Stopping voice server (PID: {pid})...")

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)

        remove_pid()
        print("Voice server stopped")
        return True

    except ProcessLookupError:
        print("Process not found (already stopped?)")
        remove_pid()
        return True
    except Exception as e:
        print(f"Error stopping server: {e}")
        return False


def server_status(port: int = DEFAULT_PORT):
    """Check server status"""
    pid = get_pid()
    running = is_server_running(port)

    print("Voice Server Status")
    print("=" * 40)
    print(f"PID file: {pid if pid else 'Not found'}")
    print(f"Server responding: {'Yes' if running else 'No'}")
    print(f"URL: {get_server_url(port)}")

    if running:
        try:
            response = requests.get(f"{get_server_url(port)}/health", timeout=2)
            data = response.json()
            print(f"OpenAI configured: {data.get('openai_configured', False)}")
            print(f"Sample rate: {data.get('sample_rate', 'Unknown')}")
        except:
            pass

    print("=" * 40)
    return running


def main():
    parser = argparse.ArgumentParser(description="Voice Server Manager")
    parser.add_argument("action", choices=["start", "stop", "status", "restart"])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if args.action == "start":
        start_server(args.port)
    elif args.action == "stop":
        stop_server()
    elif args.action == "status":
        server_status(args.port)
    elif args.action == "restart":
        stop_server()
        time.sleep(1)
        start_server(args.port)


if __name__ == "__main__":
    main()
