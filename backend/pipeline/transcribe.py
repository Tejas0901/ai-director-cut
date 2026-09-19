"""Stage 2 - speech to timestamped text.

Two providers behind one function:
  local -> faster-whisper on CPU. No key, no quota, works offline. ~30s for
           a 2-minute clip on this machine.
  groq  -> whisper-large-v3-turbo over the free API. ~2s for the same clip.

Use `local` for the demo (it is the honest "runs on my laptop" story) and
`groq` while you are iterating, so you are not waiting on CPU every run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from ..config import (
    GROQ_API_KEY,
    GROQ_WHISPER_MODEL,
    STT_PROVIDER,
    WHISPER_COMPUTE,
    WHISPER_MODEL,
)
from ..schemas import Transcript, TranscriptSegment

_MODEL_CACHE: dict[str, object] = {}


def transcribe(audio_path: str | Path, provider: str | None = None) -> Transcript:
    provider = (provider or STT_PROVIDER).lower()
    if provider == "mock":
        return _mock()
    if provider == "groq":
        return _groq(audio_path)
    return _local(audio_path)


# --------------------------------------------------------------------------


def _local(audio_path: str | Path) -> Transcript:
    from faster_whisper import WhisperModel

    if WHISPER_MODEL not in _MODEL_CACHE:
        # int8 on CPU is roughly 4x faster than float32 with no useful
        # accuracy loss at this model size.
        _MODEL_CACHE[WHISPER_MODEL] = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE
        )
    model = _MODEL_CACHE[WHISPER_MODEL]

    segments, info = model.transcribe(  # type: ignore[attr-defined]
        str(audio_path),
        beam_size=1,           # greedy: ~2x faster, negligible quality cost here
        vad_filter=True,       # skip silence instead of hallucinating over it
        condition_on_previous_text=False,
    )

    collected = [
        TranscriptSegment(start=round(s.start, 3), end=round(s.end, 3), text=s.text.strip())
        for s in segments
        if s.text and s.text.strip()
    ]
    return Transcript(segments=collected, language=getattr(info, "language", "en") or "en")


def _groq(audio_path: str | Path) -> Transcript:
    if not GROQ_API_KEY:
        raise RuntimeError("STT_PROVIDER=groq but GROQ_API_KEY is not set")

    with open(audio_path, "rb") as fh:
        response = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (Path(audio_path).name, fh, "audio/wav")},
            data={
                "model": GROQ_WHISPER_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            },
            timeout=120.0,
        )
    response.raise_for_status()
    payload = response.json()

    collected = [
        TranscriptSegment(
            start=round(float(s["start"]), 3),
            end=round(float(s["end"]), 3),
            text=str(s["text"]).strip(),
        )
        for s in payload.get("segments", [])
        if str(s.get("text", "")).strip()
    ]
    return Transcript(segments=collected, language=payload.get("language", "en"))


def _mock() -> Transcript:
    return Transcript(segments=[
        TranscriptSegment(start=0.5, end=3.2, text="Okay so this is the part where it all goes wrong."),
        TranscriptSegment(start=3.4, end=7.1, text="I pushed straight to main and the build just exploded."),
        TranscriptSegment(start=7.4, end=11.0, text="Honestly the funniest part is nobody noticed for two hours."),
    ])


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.transcribe <audio-or-video>
    import time

    from ..cache import file_key
    from .ingest import ingest

    target = sys.argv[1]
    if target.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        target = ingest(target, file_key(target)).audio_path or ""

    started = time.time()
    result = transcribe(target)
    print(f"provider={STT_PROVIDER}  took={time.time() - started:.1f}s  "
          f"segments={len(result.segments)}\n")
    for seg in result.segments:
        print(f"[{seg.start:7.2f} -> {seg.end:7.2f}]  {seg.text}")
