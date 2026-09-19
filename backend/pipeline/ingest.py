"""Stage 1 - probe the upload and extract a Whisper-ready audio track."""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import CACHE_DIR
from ..schemas import MediaInfo
from .ffmpeg_util import parse_fps, probe, run


def ingest(video_path: str | Path, key: str) -> MediaInfo:
    video_path = Path(video_path)
    info = probe(video_path)

    video_stream = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if video_stream is None:
        raise ValueError(f"{video_path.name} contains no video stream")

    duration = float(info.get("format", {}).get("duration") or 0.0)
    if duration <= 0:
        raise ValueError(f"could not determine duration of {video_path.name}")

    audio_path: str | None = None
    if audio_stream is not None:
        audio_path = str(extract_audio(video_path, key))

    return MediaInfo(
        path=str(video_path),
        duration=duration,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=parse_fps(video_stream.get("avg_frame_rate")),
        has_audio=audio_stream is not None,
        audio_path=audio_path,
    )


def extract_audio(video_path: str | Path, key: str) -> Path:
    """16 kHz mono WAV - exactly what Whisper wants, no resampling later."""
    out_dir = CACHE_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audio.wav"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    run([
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_path),
    ])
    return out_path


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.ingest <video>
    from ..cache import file_key

    target = sys.argv[1]
    media = ingest(target, file_key(target))
    print(media.model_dump_json(indent=2))
