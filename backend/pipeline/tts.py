"""Stage 6 - narration audio.

One interface, three implementations:
  edge   -> edge-tts. No API key at all, 200+ neural voices, sounds human.
  piper  -> a local binary + onnx voice. Fully offline, no third-party terms.
  mock   -> a silent track of plausible length, for rendering tests.

Swap with TTS_PROVIDER. The renderer only ever sees "an mp3 at this path",
so nothing downstream knows or cares which one produced it.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from ..config import PIPER_BIN, PIPER_VOICE, TTS_PROVIDER, TTS_VOICE
from .ffmpeg_util import run


class Narration(NamedTuple):
    """Where the audio landed, and whether it is speech or a stand-in.

    `spoken` is False whenever there were words to say and the track came out
    silent anyway - the render is fine either way, but the viewer is owed an
    explanation for a reel that never talks.
    """

    path: Path
    spoken: bool
    detail: str | None = None


# A badge or a warning strip has room for a sentence, not a stack trace.
MAX_DETAIL_CHARS = 160


def synthesize(text: str, out_path: str | Path, provider: str | None = None) -> Narration:
    """Render `text` to an audio file. Always returns a playable path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider = (provider or TTS_PROVIDER).lower()

    text = (text or "").strip()
    if not text:
        # Nothing to say, so silence is the correct output, not a shortfall.
        return Narration(_silence(out_path, 0.4), spoken=True)

    if provider == "mock":
        # Silent on purpose - but still worth reporting, or a demo run with
        # TTS_PROVIDER=mock looks like broken narration.
        return Narration(
            _silence(out_path, _estimate_seconds(text)),
            spoken=False,
            detail="TTS_PROVIDER=mock, so narration is intentionally silent",
        )

    try:
        path = _piper(text, out_path) if provider == "piper" else _edge(text, out_path)
        return Narration(path, spoken=True)
    except Exception as exc:
        # Narration is a nice-to-have; never let it take down a render.
        print(f"[tts] {provider} failed ({exc}); substituting silence")
        return Narration(
            _silence(out_path, _estimate_seconds(text)),
            spoken=False,
            detail=_short_detail(provider, exc),
        )


def _short_detail(provider: str, exc: Exception) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    if len(text) > MAX_DETAIL_CHARS:
        text = text[:MAX_DETAIL_CHARS - 1].rstrip() + "…"
    return f"{provider}: {text}"


def _estimate_seconds(text: str) -> float:
    """~165 words per minute is a natural narration pace."""
    return max(0.6, len(text.split()) / 165.0 * 60.0)


# --------------------------------------------------------------------------


def _edge(text: str, out_path: Path) -> Path:
    import edge_tts

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(str(out_path))

    asyncio.run(_run())
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("edge-tts produced an empty file")
    return out_path


def _piper(text: str, out_path: Path) -> Path:
    if not PIPER_BIN or not PIPER_VOICE:
        raise RuntimeError("TTS_PROVIDER=piper requires PIPER_BIN and PIPER_VOICE")

    wav_path = out_path.with_suffix(".wav")
    proc = subprocess.run(
        [PIPER_BIN, "--model", PIPER_VOICE, "--output_file", str(wav_path)],
        input=text,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0 or not wav_path.exists():
        raise RuntimeError((proc.stderr or "piper failed").strip()[-400:])

    if out_path.suffix.lower() != ".wav":
        run(["-y", "-i", str(wav_path), str(out_path)])
        wav_path.unlink(missing_ok=True)
        return out_path
    return wav_path


def _silence(out_path: Path, seconds: float) -> Path:
    run([
        "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", f"{max(0.2, seconds):.2f}",
        str(out_path),
    ])
    return out_path


def duration_of(path: str | Path) -> float:
    from .ffmpeg_util import probe

    try:
        return float(probe(path)["format"]["duration"])
    except Exception:
        return 0.0


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.tts "some words"
    from ..config import OUTPUT_DIR

    phrase = sys.argv[1] if len(sys.argv) > 1 else \
        "Here's everything worth watching, and nothing that isn't."
    target = OUTPUT_DIR / "tts_test.mp3"
    result = synthesize(phrase, target)
    print(f"provider={TTS_PROVIDER}  voice={TTS_VOICE}")
    print(f"wrote {result.path}  ({duration_of(result.path):.2f}s)")
    if not result.spoken:
        print(f"SILENT: {result.detail}")
