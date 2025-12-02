# Windows HTTP Voice Server

Alternative implementation for Windows users experiencing stdio/audio conflicts with the MCP transport.

## Problem

On Windows, the MCP stdio transport interferes with audio stream callbacks, causing silence detection to fail. Recording continues indefinitely instead of stopping when the user stops speaking.

## Solution

This implementation runs as a standalone HTTP server, completely avoiding the stdio transport layer.

```
┌─────────────────┐     HTTP (not stdio!)    ┌──────────────────┐
│  Claude Code    │ ◄──────────────────────► │  Voice Server    │
│                 │                          │  localhost:8765  │
└─────────────────┘                          └────────┬─────────┘
                                                      │
                                             ┌────────▼─────────┐
                                             │  Audio Pipeline  │
                                             │  (works correctly)│
                                             └──────────────────┘
```

## Features

- ✅ Reliable silence detection using WebRTC VAD
- ✅ OpenAI TTS and STT (Whisper) integration
- ✅ Pleasant audio feedback tones (start/end of recording)
- ✅ Configurable VAD aggressiveness, min/max duration

## Installation

```bash
pip install fastapi uvicorn sounddevice webrtcvad scipy openai numpy
```

## Usage

### Start the server

```bash
python voice_server_manager.py start
python voice_server_manager.py status
python voice_server_manager.py stop
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/devices` | GET | List audio devices |
| `/speak` | POST | TTS only |
| `/converse` | POST | TTS + listen + STT |

### Example

```python
import requests

response = requests.post("http://127.0.0.1:8765/converse", json={
    "message": "Hello! What can I help you with?",
    "wait_for_response": True,
    "min_duration": 2.0,
    "max_duration": 30.0,
    "vad_aggressiveness": 2,
    "voice": "nova",
    "speed": 1.0
})

result = response.json()
# {"success": true, "transcript": "User's response", "timing": {...}}
```

## Configuration

Set `OPENAI_API_KEY` in your environment or create a `.tajne` file:

```
OPENAI_API_KEY=sk-...
```

## Credits

This is a fork of [mbailey/voicemode](https://github.com/mbailey/voicemode) with Windows-specific fixes.

Original voicemode MCP is excellent for Mac/Linux. This HTTP server alternative is specifically for Windows users experiencing the stdio/audio conflict.
