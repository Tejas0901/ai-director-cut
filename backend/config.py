"""Central configuration. Every tunable lives here, read once from .env."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
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
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.6-flash")
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

# Stills handed to a multimodal Director so it can see what the footage is of.
# The loudness and motion columns say where the interesting moments are; only
# these say what the video is about. Zero disables them entirely.
DIRECTOR_FRAMES = int(_env("DIRECTOR_FRAMES", "8"))
DIRECTOR_FRAME_WIDTH = int(_env("DIRECTOR_FRAME_WIDTH", "512"))

RENDER_WIDTH = int(_env("RENDER_WIDTH", "1280"))
RENDER_HEIGHT = int(_env("RENDER_HEIGHT", "720"))
RENDER_FPS = int(_env("RENDER_FPS", "30"))
MUSIC_VOLUME = float(_env("MUSIC_VOLUME", "0.12"))
DUCK_VOLUME = float(_env("DUCK_VOLUME", "0.15"))

# --- overlay font ---------------------------------------------------------
# Optional. Set it to force a specific face; leave it empty and we go looking
# for a bold sans on the usual paths. Overlay titles are the only thing that
# needs it, and the renderer drops them rather than failing if nothing turns
# up - so a machine with no fonts at all still produces a reel.
FONT_FILE = _env("FONT_FILE")

# Bold faces, because a caption burned over footage needs the weight to stay
# readable. Wrong-platform entries simply do not exist, so one flat list in
# rough order of likelihood beats branching on sys.platform.
FONT_CANDIDATES: tuple[str, ...] = (
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    # Linux - Debian/Ubuntu layout
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    # Linux - Fedora/Arch layout
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)


@lru_cache(maxsize=1)
def resolve_font() -> Path | None:
    """First usable bold font: FONT_FILE if set, else the known locations.

    Returns None when nothing is found. Existence is all we check - a file
    that exists but is corrupt fails inside ffmpeg, and the renderer's
    retry-without-overlays path is what covers that.
    """
    if FONT_FILE:
        chosen = Path(FONT_FILE)
        if chosen.is_file():
            return chosen
        # Set deliberately and wrong is worth saying out loud; a typo here
        # would otherwise look like "the captions just stopped appearing".
        print(f"[config] FONT_FILE={FONT_FILE} does not exist; searching the usual paths")

    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path

    return None


def ffmpeg_fontfile() -> str | None:
    """The resolved font, escaped for use inside an ffmpeg filter graph.

    None means no font was found, which is the renderer's cue to skip overlay
    titles entirely. Windows needs the drive colon escaped.
    """
    font = resolve_font()
    if font is None:
        return None
    return str(font).replace("\\", "/").replace(":", r"\:")


def is_stubbed(stage: str) -> bool:
    return stage in STUB_STAGES
