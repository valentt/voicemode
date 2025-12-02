#!/usr/bin/env python3
"""
Voice Server - Local HTTP server for voice interactions

Replaces voicemode MCP to avoid stdio/audio conflicts on Windows.
Runs as a separate process, communicates via HTTP.

Usage:
    python scripts/voice_server.py [--port 8765]

Endpoints:
    GET  /health        - Health check
    GET  /devices       - List audio devices
    POST /speak         - TTS only (no microphone)
    POST /converse      - TTS + listen + STT
"""

import asyncio
import json
import logging
import os
import queue
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Load credentials from .tajne or ~/.claude.json
def load_openai_key():
    # Try .tajne first
    tajne_path = Path(__file__).parent.parent / ".tajne"
    if tajne_path.exists():
        with open(tajne_path) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.strip().split("=", 1)[1]

    # Try ~/.claude.json
    claude_json = Path.home() / ".claude.json"
    if claude_json.exists():
        try:
            with open(claude_json) as f:
                config = json.load(f)
            key = config.get("mcpServers", {}).get("voicemode", {}).get("env", {}).get("OPENAI_API_KEY")
            if key:
                return key
        except:
            pass

    # Try environment
    return os.environ.get("OPENAI_API_KEY")


OPENAI_API_KEY = load_openai_key()

# Configuration
SAMPLE_RATE = 24000
VAD_SAMPLE_RATE = 16000
CHANNELS = 1
VAD_CHUNK_DURATION_MS = 30
DEFAULT_SILENCE_THRESHOLD_MS = 1000
DEFAULT_MIN_DURATION = 2.0
DEFAULT_MAX_DURATION = 30.0
DEFAULT_VAD_AGGRESSIVENESS = 2
DEFAULT_VOICE = "nova"
DEFAULT_SPEED = 1.0

# Beep configuration
BEEP_START_FREQ = 800   # Hz - higher pitch for "start"
BEEP_END_FREQ = 600     # Hz - lower pitch for "end"
BEEP_DURATION = 0.15    # seconds


def play_beep(frequency: int = 800, duration: float = 0.2, is_start: bool = True):
    """
    Play an elegant, pleasant notification sound.
    Inspired by Apple's design philosophy - simple yet refined.
    """
    from scipy.io.wavfile import write
    import subprocess

    # Generate a soft, pleasant tone with harmonics
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)

    if is_start:
        # Start sound: gentle rising chime (C5 -> E5 quick arpeggio feel)
        # Main frequency with soft harmonics
        fundamental = np.sin(2 * np.pi * frequency * t)
        harmonic2 = np.sin(2 * np.pi * frequency * 2 * t) * 0.3  # Octave
        harmonic3 = np.sin(2 * np.pi * frequency * 1.5 * t) * 0.2  # Fifth
        tone = fundamental + harmonic2 + harmonic3
    else:
        # End sound: soft descending tone (more mellow)
        fundamental = np.sin(2 * np.pi * frequency * t)
        harmonic2 = np.sin(2 * np.pi * frequency * 0.5 * t) * 0.2  # Sub-octave for warmth
        tone = fundamental + harmonic2

    # Normalize
    tone = tone / np.max(np.abs(tone))

    # Elegant envelope: quick attack, sustained, gentle release
    envelope = np.ones_like(tone)
    attack_samples = int(SAMPLE_RATE * 0.01)   # 10ms attack
    release_samples = int(SAMPLE_RATE * 0.08)  # 80ms release

    # Smooth attack (exponential)
    envelope[:attack_samples] = 1 - np.exp(-5 * np.linspace(0, 1, attack_samples))
    # Smooth release (exponential decay)
    envelope[-release_samples:] = np.exp(-3 * np.linspace(0, 1, release_samples))

    tone = tone * envelope * 0.4  # Gentle volume

    # Convert to int16
    audio = (tone * 32767).astype(np.int16)

    # Save to temp file and play
    temp_path = tempfile.mktemp(suffix='.wav')
    write(temp_path, SAMPLE_RATE, audio)

    try:
        # Use PowerShell for reliable playback
        subprocess.run(
            ['powershell', '-c', f'(New-Object Media.SoundPlayer "{temp_path}").PlaySync()'],
            capture_output=True,
            timeout=2
        )
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice-server")

# FastAPI app
app = FastAPI(title="Voice Server", version="1.0.0")


# Request models
class SpeakRequest(BaseModel):
    message: str
    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED


class ConverseRequest(BaseModel):
    message: str
    wait_for_response: bool = True
    min_duration: float = DEFAULT_MIN_DURATION
    max_duration: float = DEFAULT_MAX_DURATION
    vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS
    silence_threshold_ms: int = DEFAULT_SILENCE_THRESHOLD_MS
    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED


# Audio functions
def text_to_speech(text: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> tuple[bool, dict]:
    """Convert text to speech using OpenAI API and play it"""
    if not OPENAI_API_KEY:
        return False, {"error": "OPENAI_API_KEY not configured"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        start_time = time.perf_counter()

        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed,
            response_format="pcm"  # Raw PCM for direct playback
        )

        gen_time = time.perf_counter() - start_time

        # Convert to numpy array
        audio_data = np.frombuffer(response.content, dtype=np.int16)

        # Play audio
        play_start = time.perf_counter()
        sd.play(audio_data, samplerate=24000)
        sd.wait()
        play_time = time.perf_counter() - play_start

        return True, {
            "generation_time": round(gen_time, 2),
            "playback_time": round(play_time, 2),
            "total_time": round(gen_time + play_time, 2)
        }

    except Exception as e:
        logger.error(f"TTS error: {e}")
        return False, {"error": str(e)}


def record_with_vad(
    max_duration: float,
    min_duration: float,
    vad_aggressiveness: int,
    silence_threshold_ms: int
) -> tuple[np.ndarray, bool, float]:
    """
    Record audio with Voice Activity Detection.
    Returns: (audio_data, speech_detected, recording_duration)
    """
    try:
        import webrtcvad
        from scipy import signal
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return np.array([]), False, 0

    # Initialize VAD
    vad = webrtcvad.Vad(vad_aggressiveness)

    # Chunk sizes
    chunk_samples = int(SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
    chunk_duration_s = VAD_CHUNK_DURATION_MS / 1000
    vad_chunk_samples = int(VAD_SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)

    # State
    chunks = []
    silence_duration_ms = 0
    recording_duration = 0.0
    speech_detected = False
    stop_recording = False

    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        audio_queue.put(indata.copy())

    logger.info(f"Recording with VAD (max={max_duration}s, min={min_duration}s, vad={vad_aggressiveness})")

    # Beep to indicate recording started
    play_beep(BEEP_START_FREQ, BEEP_DURATION, is_start=True)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
            callback=audio_callback,
            blocksize=chunk_samples
        ):
            while recording_duration < max_duration and not stop_recording:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    chunk_flat = chunk.flatten()
                    chunks.append(chunk_flat)

                    # Resample for VAD (24kHz -> 16kHz)
                    resampled_length = int(len(chunk_flat) * VAD_SAMPLE_RATE / SAMPLE_RATE)
                    vad_chunk = signal.resample(chunk_flat, resampled_length)
                    vad_chunk = vad_chunk[:vad_chunk_samples].astype(np.int16)

                    # Check for speech
                    try:
                        is_speech = vad.is_speech(vad_chunk.tobytes(), VAD_SAMPLE_RATE)
                    except:
                        is_speech = True

                    # State machine
                    if not speech_detected:
                        if is_speech:
                            logger.info("Speech detected - starting active recording")
                            speech_detected = True
                            silence_duration_ms = 0
                    else:
                        if is_speech:
                            silence_duration_ms = 0
                        else:
                            silence_duration_ms += VAD_CHUNK_DURATION_MS

                            # Check stop condition
                            if recording_duration >= min_duration and silence_duration_ms >= silence_threshold_ms:
                                logger.info(f"Silence threshold reached after {recording_duration:.1f}s")
                                stop_recording = True

                    recording_duration += chunk_duration_s

                except queue.Empty:
                    continue

        # Beep to indicate recording ended
        play_beep(BEEP_END_FREQ, BEEP_DURATION, is_start=False)

        # Concatenate chunks
        if chunks:
            audio_data = np.concatenate(chunks)
            logger.info(f"Recorded {len(audio_data)} samples ({recording_duration:.1f}s)")
            return audio_data, speech_detected, recording_duration
        else:
            return np.array([]), False, 0

    except Exception as e:
        logger.error(f"Recording error: {e}")
        return np.array([]), False, 0


def speech_to_text(audio_data: np.ndarray) -> tuple[bool, dict]:
    """Convert speech to text using OpenAI Whisper API"""
    if not OPENAI_API_KEY:
        return False, {"error": "OPENAI_API_KEY not configured"}

    if len(audio_data) == 0:
        return False, {"error": "No audio data"}

    try:
        from openai import OpenAI
        from scipy.io.wavfile import write

        client = OpenAI(api_key=OPENAI_API_KEY)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            write(temp_path, SAMPLE_RATE, audio_data)

        try:
            start_time = time.perf_counter()

            with open(temp_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

            stt_time = time.perf_counter() - start_time

            transcript = response.text.strip()

            return True, {
                "transcript": transcript if transcript else None,
                "stt_time": round(stt_time, 2)
            }

        finally:
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"STT error: {e}")
        return False, {"error": str(e)}


# API Endpoints
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "sample_rate": SAMPLE_RATE
    }


@app.get("/devices")
async def list_devices():
    """List available audio devices"""
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        input_devices = []
        output_devices = []

        for i, d in enumerate(devices):
            device_info = {
                "index": i,
                "name": d["name"],
                "default": False
            }

            if d["max_input_channels"] > 0:
                device_info["default"] = (i == default_input)
                input_devices.append(device_info)

            if d["max_output_channels"] > 0:
                device_info["default"] = (i == default_output)
                output_devices.append(device_info)

        return {
            "input_devices": input_devices,
            "output_devices": output_devices
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak")
async def speak(request: SpeakRequest):
    """Text-to-speech only (no microphone)"""
    logger.info(f"TTS request: {request.message[:50]}...")

    success, result = text_to_speech(
        text=request.message,
        voice=request.voice,
        speed=request.speed
    )

    if success:
        return {
            "success": True,
            "timing": result
        }
    else:
        raise HTTPException(status_code=500, detail=result.get("error", "TTS failed"))


@app.post("/converse")
async def converse(request: ConverseRequest):
    """Full conversation: TTS + listen + STT"""
    logger.info(f"Converse request: {request.message[:50]}...")

    timings = {}

    # Step 1: TTS
    tts_start = time.perf_counter()
    tts_success, tts_result = text_to_speech(
        text=request.message,
        voice=request.voice,
        speed=request.speed
    )
    timings["tts"] = round(time.perf_counter() - tts_start, 2)

    if not tts_success:
        raise HTTPException(status_code=500, detail=tts_result.get("error", "TTS failed"))

    # If speak-only mode
    if not request.wait_for_response:
        return {
            "success": True,
            "transcript": None,
            "timing": timings
        }

    # Step 2: Record with VAD
    logger.info("Listening...")
    record_start = time.perf_counter()

    # Run recording in thread pool to not block
    loop = asyncio.get_event_loop()
    audio_data, speech_detected, rec_duration = await loop.run_in_executor(
        None,
        record_with_vad,
        request.max_duration,
        request.min_duration,
        request.vad_aggressiveness,
        request.silence_threshold_ms
    )

    timings["recording"] = round(time.perf_counter() - record_start, 2)

    if not speech_detected:
        logger.info("No speech detected")
        return {
            "success": True,
            "transcript": None,
            "speech_detected": False,
            "timing": timings
        }

    # Step 3: STT
    stt_start = time.perf_counter()
    stt_success, stt_result = speech_to_text(audio_data)
    timings["stt"] = round(time.perf_counter() - stt_start, 2)

    if not stt_success:
        raise HTTPException(status_code=500, detail=stt_result.get("error", "STT failed"))

    timings["total"] = round(sum(timings.values()), 2)

    transcript = stt_result.get("transcript")
    logger.info(f"Transcript: {transcript}")

    return {
        "success": True,
        "transcript": transcript,
        "speech_detected": True,
        "timing": timings
    }


def main():
    """Run the voice server"""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found! Check .tajne or ~/.claude.json")
        sys.exit(1)

    logger.info(f"Starting Voice Server on {args.host}:{args.port}")
    logger.info(f"OpenAI API Key: {OPENAI_API_KEY[:20]}...{OPENAI_API_KEY[-4:]}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
