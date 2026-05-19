"""
pipeline/tts.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Voice generation module.

Priority:
  1. ElevenLabs API  — emotional, human-like (requires API key)
  2. edge-tts        — free fallback, still good quality

Set ELEVENLABS_API_KEY in .env to enable ElevenLabs.
Leave it empty → auto-falls back to edge-tts.
"""

import asyncio
import logging
import os
from pathlib import Path

log = logging.getLogger("ugc.tts")

# ── ElevenLabs voice IDs (most natural ones) ────────────────────────────────
ELEVENLABS_VOICES = {
    "Rachel (Female, Warm)":    "21m00Tcm4TlvDq8ikWAM",
    "Bella (Female, Soft)":     "EXAVITQu4vr4xnSDxMaL",
    "Antoni (Male, Warm)":      "ErXwobaYiN019PkySvjV",
    "Josh (Male, Deep)":        "TxGEqnHWrfWFTfGW9XjX",
    "Arnold (Male, Strong)":    "VR6AewLTigWG4xSOukaG",
    "Elli (Female, Emotional)": "MF3mGyEYCl7XYWbV9V6O",
}

# ── edge-tts voice IDs ───────────────────────────────────────────────────────
EDGE_VOICES = {
    "Aria (Female, US)":        "en-US-AriaNeural",
    "Jenny (Female, US)":       "en-US-JennyNeural",
    "Sara (Female, US)":        "en-US-SaraNeural",
    "Christopher (Male, US)":   "en-US-ChristopherNeural",
    "Eric (Male, US)":          "en-US-EricNeural",
    "Guy (Male, US)":           "en-US-GuyNeural",
    "Sonia (Female, UK)":       "en-GB-SoniaNeural",
    "Ryan (Male, UK)":          "en-GB-RyanNeural",
}

ALL_VOICES = {**ELEVENLABS_VOICES, **EDGE_VOICES}


def generate_audio(
    script: str,
    voice_key: str,
    job_dir: Path,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
) -> Path:
    """
    Generate voiceover audio from script.
    Tries ElevenLabs first, falls back to edge-tts.

    Args:
        script:           Ad copy text.
        voice_key:        Display name from ALL_VOICES.
        job_dir:          Job working directory.
        rate:             edge-tts only — speech rate e.g. "+10%".
        pitch:            edge-tts only — pitch e.g. "+0Hz".
        stability:        ElevenLabs only — 0.0–1.0.
        similarity_boost: ElevenLabs only — 0.0–1.0.

    Returns:
        Path to generated .mp3 file.
    """
    if not script.strip():
        raise ValueError("Script cannot be empty.")

    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()

    if api_key and voice_key in ELEVENLABS_VOICES:
        log.info(f"[TTS] Using ElevenLabs: {voice_key}")
        return _elevenlabs_tts(
            script, voice_key, job_dir, api_key, stability, similarity_boost
        )
    else:
        if api_key and voice_key in ELEVENLABS_VOICES:
            log.warning("[TTS] ElevenLabs key found but voice not in EL list — using edge-tts")
        else:
            log.info("[TTS] No ElevenLabs key — using edge-tts")

        # Map EL voice names to edge-tts equivalents if needed
        edge_key = voice_key if voice_key in EDGE_VOICES else "Aria (Female, US)"
        return asyncio.run(_edge_tts(script, edge_key, job_dir, rate, pitch))


def _elevenlabs_tts(
    script: str,
    voice_key: str,
    job_dir: Path,
    api_key: str,
    stability: float,
    similarity_boost: float,
) -> Path:
    """ElevenLabs v2 API call."""
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    voice_id = ELEVENLABS_VOICES[voice_key]
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": script,
        "model_id": "eleven_multilingual_v2",  # best quality model
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": 0.3,              # slight style exaggeration for ads
            "use_speaker_boost": True,
        },
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"[TTS] ElevenLabs API error {response.status_code}: {response.text[:500]}\n"
            "Falling back to edge-tts is not automatic here — check your API key."
        )

    output_path = job_dir / "voiceover.mp3"
    output_path.write_bytes(response.content)
    log.info(f"[TTS] ElevenLabs audio saved: {output_path} ({len(response.content)} bytes)")
    return output_path


async def _edge_tts(
    script: str,
    voice_key: str,
    job_dir: Path,
    rate: str,
    pitch: str,
) -> Path:
    """edge-tts async TTS."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError("pip install edge-tts")

    voice_id = EDGE_VOICES.get(voice_key, "en-US-AriaNeural")
    output_path = job_dir / "voiceover.mp3"

    communicate = edge_tts.Communicate(
        text=script,
        voice=voice_id,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(str(output_path))

    if not output_path.exists() or output_path.stat().st_size < 1000:
        raise RuntimeError(
            "[TTS] edge-tts produced empty file. "
            "Check internet connection — it streams from Microsoft servers."
        )

    log.info(f"[TTS] edge-tts audio saved: {output_path}")
    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """Returns audio duration in seconds via ffprobe."""
    import json, subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(audio_path)],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])
