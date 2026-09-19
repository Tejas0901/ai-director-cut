"""Central configuration. Every tunable lives here, read once from .env."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- paths ----------------------------------------------------------------
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"
ASSETS_DIR = ROOT / "assets"
MUSIC_DIR = ASSETS_DIR / "music"
FIXTURES_DIR = ROOT / "fixtures"

for _d in (UPLOAD_DIR, CACHE_DIR, OUTPUT_DIR, MUSIC_DIR, FIXTURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _flag(key: str, default: bool = False) -> bool:
    raw = _env(key, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


# --- binaries -------------------------------------------------------------
FFMPEG = _env("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = _env("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"

# --- providers ------------------------------------------------------------
LLM_PROVIDER = _env("LLM_PROVIDER", "gemini")  # gemini | groq | mock
GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")

STT_PROVIDER = _env("STT_PROVIDER", "local")  # local | groq | mock
WHISPER_MODEL = _env("WHISPER_MODEL", "base.en")
WHISPER_COMPUTE = _env("WHISPER_COMPUTE", "int8")
GROQ_WHISPER_MODEL = _env("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

TTS_PROVIDER = _env("TTS_PROVIDER", "edge")  # edge | piper | mock
TTS_VOICE = _env("TTS_VOICE", "en-US-AndrewNeural")
PIPER_BIN = _env("PIPER_BIN")
PIPER_VOICE = _env("PIPER_VOICE")

# --- behaviour ------------------------------------------------------------
# Every stage can be stubbed independently. This is how we keep an
# end-to-end-green pipeline while individual stages are still being written.
STUB_STAGES = {s for s in _env("STUB_STAGES", "").split(",") if s.strip()}
USE_CACHE = _flag("USE_CACHE", True)

BUCKET_SECONDS = float(_env("BUCKET_SECONDS", "0.5"))
VISION_SAMPLE_FPS = float(_env("VISION_SAMPLE_FPS", "4"))
TARGET_CLIPS = int(_env("TARGET_CLIPS", "4"))

RENDER_WIDTH = int(_env("RENDER_WIDTH", "1280"))
RENDER_HEIGHT = int(_env("RENDER_HEIGHT", "720"))
RENDER_FPS = int(_env("RENDER_FPS", "30"))
MUSIC_VOLUME = float(_env("MUSIC_VOLUME", "0.12"))
DUCK_VOLUME = float(_env("DUCK_VOLUME", "0.15"))

# Windows needs an escaped drive colon inside an ffmpeg filter string.
FONT_FILE = _env("FONT_FILE", "C:/Windows/Fonts/arialbd.ttf")


def ffmpeg_fontfile() -> str:
    """Escape a Windows font path for use inside an ffmpeg filter graph."""
    return FONT_FILE.replace("\\", "/").replace(":", r"\:")


def is_stubbed(stage: str) -> bool:
    return stage in STUB_STAGES
