"""Stage 4 - fuse transcript + visual buckets + audio loudness into one table.

This is the artefact the Director actually reads, so it is worth reading
yourself: `render_markdown()` is the exact text the LLM sees. If you cannot
tell from that text where the good moments are, the LLM cannot either, and
the fix belongs here rather than in the prompt.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from ..config import BUCKET_SECONDS
from ..schemas import MediaInfo, Timeline, TimelineBucket, Transcript, VisualBucket


def build(media: MediaInfo, transcript: Transcript, visual: list[VisualBucket]) -> Timeline:
    count = max(1, int(np.ceil(media.duration / BUCKET_SECONDS)))
    rms = _audio_rms(media.audio_path, count) if media.audio_path else [0.0] * count
    by_time = {round(v.t, 3): v for v in visual}

    buckets: list[TimelineBucket] = []
    for i in range(count):
        start = round(i * BUCKET_SECONDS, 3)
        vis = by_time.get(start)
        buckets.append(TimelineBucket(
            t=start,
            speech=_speech_at(transcript, start, start + BUCKET_SECONDS),
            audio_rms=round(rms[i] if i < len(rms) else 0.0, 4),
            motion=vis.motion if vis else 0.0,
            scene_cut=vis.scene_cut if vis else False,
            faces=vis.faces if vis else 0,
        ))

    return Timeline(media=media, transcript=transcript, buckets=buckets,
                    bucket_seconds=BUCKET_SECONDS)


def _speech_at(transcript: Transcript, start: float, end: float) -> str:
    """Words from any segment overlapping this window."""
    parts = [s.text for s in transcript.segments if s.start < end and s.end > start]
    return " ".join(parts).strip()


def _audio_rms(audio_path: str | None, count: int) -> list[float]:
    """Per-bucket loudness, normalised against this clip's own 95th percentile."""
    if not audio_path or not Path(audio_path).exists():
        return [0.0] * count

    try:
        with wave.open(str(audio_path), "rb") as wav:
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except Exception:
        return [0.0] * count

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return [0.0] * count

    per_bucket = max(1, int(rate * BUCKET_SECONDS))
    values: list[float] = []
    for i in range(count):
        chunk = samples[i * per_bucket:(i + 1) * per_bucket]
        values.append(float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0)

    ceiling = float(np.percentile(values, 95)) or 1.0
    return [min(1.0, v / ceiling) for v in values]


def render_markdown(timeline: Timeline, max_rows: int = 260) -> str:
    """Compact the timeline into the text block handed to the Director.

    Buckets are merged in pairs when the video is long enough to overflow the
    row budget, keeping the prompt roughly constant in size regardless of
    input length.
    """
    buckets = timeline.buckets
    stride = max(1, len(buckets) // max_rows + 1)

    lines = ["| time | speech | loud | motion | cut | faces |",
             "|------|--------|------|--------|-----|-------|"]
    for i in range(0, len(buckets), stride):
        window = buckets[i:i + stride]
        head = window[0]
        speech = " ".join(b.speech for b in window if b.speech).strip()
        lines.append(
            f"| {head.t:.1f} | {speech[:110] or '-'} | "
            f"{max(b.audio_rms for b in window):.2f} | "
            f"{max(b.motion for b in window):.2f} | "
            f"{'Y' if any(b.scene_cut for b in window) else ''} | "
            f"{max(b.faces for b in window)} |"
        )
    return "\n".join(lines)


def top_moments(timeline: Timeline, count: int, min_seconds: float = 4.0) -> list[tuple[float, float]]:
    """Heuristic clip picker - the fallback when the LLM is unavailable.

    Slides a window across the timeline, scores each position by mean energy,
    then greedily takes the highest-scoring non-overlapping windows.
    """
    buckets = timeline.buckets
    if not buckets:
        return []

    width = max(1, int(min_seconds / timeline.bucket_seconds))
    scores = [
        (float(np.mean([b.energy for b in buckets[i:i + width]])), i)
        for i in range(0, max(1, len(buckets) - width))
    ]
    scores.sort(reverse=True)

    chosen: list[int] = []
    for _, index in scores:
        if all(abs(index - existing) >= width for existing in chosen):
            chosen.append(index)
        if len(chosen) >= count:
            break

    chosen.sort()
    return [
        (round(i * timeline.bucket_seconds, 2),
         round(min((i + width) * timeline.bucket_seconds, timeline.media.duration), 2))
        for i in chosen
    ]


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.timeline <video>
    from ..cache import file_key
    from .ingest import ingest
    from .transcribe import transcribe
    from .vision import analyze

    target = sys.argv[1]
    key = file_key(target)
    media_info = ingest(target, key)
    tl = build(
        media_info,
        transcribe(media_info.audio_path) if media_info.audio_path else Transcript(),
        analyze(target, media_info.duration),
    )
    print(render_markdown(tl))
    print("\nHeuristic picks:", top_moments(tl, 4))
