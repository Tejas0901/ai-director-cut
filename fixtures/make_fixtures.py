"""Build the test clips the pipeline is developed against.

Two fixtures, deliberately different, because the bugs live in the difference:

  synthetic.mp4  - hard scene cuts, plenty of motion, NO speech at all.
                   Exercises the "transcript is empty" path: the Director has
                   to choose on motion alone and the timeline has no words in
                   it. This is the clip that breaks naive prompt code.

  talkie.mp4     - real spoken audio (edge-tts) over changing scenes, with
                   deliberate quiet gaps between lines. Exercises Whisper,
                   timestamp alignment, and clip boundaries that are supposed
                   to land on sentence edges rather than mid-word.

Usage:  uv run python -m fixtures.make_fixtures [synthetic|talkie|all]

Everything is generated from ffmpeg's own sources plus a TTS voice, so the
fixtures carry no third-party footage and can be rebuilt on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.config import FIXTURES_DIR
from backend.pipeline.ffmpeg_util import probe, run
from backend.pipeline.tts import synthesize

WIDTH, HEIGHT, FPS = 1280, 720, 30

# Scene sources, chosen so consecutive scenes differ in BOTH colour histogram
# (so the scene-cut detector fires) and motion level (so the energy curve has
# actual peaks and troughs rather than a flat line).
SCENES = [
    "testsrc2=size={w}x{h}:rate={fps}",
    "life=size={w}x{h}:rate={fps}:mold=10:ratio=0.1:death_color=#c83232:life_color=#00ff66",
    "gradients=size={w}x{h}:rate={fps}:speed=0.05",
    "mandelbrot=size={w}x{h}:rate={fps}",
    "smptebars=size={w}x{h}:rate={fps}",
    "rgbtestsrc=size={w}x{h}:rate={fps}",
]

# A script with an actual shape: setup, escalation, payoff. The Director should
# be able to find a story here, and the quiet gaps give it clean cut points.
NARRATION = [
    (1.0, "Okay, we are recording. So this is the demo that almost did not happen."),
    (0.8, "Twenty minutes before the deadline, the render pipeline just stopped working."),
    (1.2, "And I mean completely stopped. Black screen, no audio, nothing."),
    (0.8, "Turns out I had been feeding it timestamps in milliseconds instead of seconds."),
    (1.0, "One divide by a thousand. That was the entire bug."),
    (0.8, "It has been running perfectly ever since, which is somehow more annoying."),
    (1.2, "Anyway. That is how this thing got built. Thanks for watching."),
]


def build_synthetic(out_path: Path, scene_seconds: float = 2.5) -> Path:
    """Silent clip: five hard cuts, no audio stream at all."""
    scenes = SCENES[:5]
    inputs: list[str] = []
    for source in scenes:
        inputs += ["-f", "lavfi", "-t", str(scene_seconds),
                   "-i", source.format(w=WIDTH, h=HEIGHT, fps=FPS)]

    chain = "".join(f"[{i}:v]" for i in range(len(scenes)))
    run([
        "-y", *inputs,
        "-filter_complex", f"{chain}concat=n={len(scenes)}:v=1:a=0,setsar=1[v]",
        "-map", "[v]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ])
    return out_path


def build_talkie(out_path: Path, work_dir: Path) -> Path:
    """Spoken clip: real narration, quiet gaps, scenes cut to the audio length."""
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Voice each line, then lay it out on a timeline with gaps between.
    pieces: list[Path] = []
    for i, (gap, line) in enumerate(NARRATION):
        pieces.append(_silence(work_dir / f"gap{i}.wav", gap))
        spoken = synthesize(line, work_dir / f"line{i}.mp3")
        pieces.append(_to_wav(spoken, work_dir / f"line{i}.wav"))
    pieces.append(_silence(work_dir / "gap_end.wav", 1.5))

    narration = _concat_audio(pieces, work_dir / "narration.wav")
    duration = float(probe(narration)["format"]["duration"])

    # 2. Build video of exactly that length, split evenly across the scenes.
    scene_seconds = duration / len(SCENES)
    inputs: list[str] = []
    for source in SCENES:
        inputs += ["-f", "lavfi", "-t", f"{scene_seconds:.3f}",
                   "-i", source.format(w=WIDTH, h=HEIGHT, fps=FPS)]

    chain = "".join(f"[{i}:v]" for i in range(len(SCENES)))
    run([
        "-y", *inputs, "-i", str(narration),
        "-filter_complex", f"{chain}concat=n={len(SCENES)}:v=1:a=0,setsar=1[v]",
        "-map", "[v]", "-map", f"{len(SCENES)}:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ])
    return out_path


# --------------------------------------------------------------------------


def _silence(path: Path, seconds: float) -> Path:
    run(["-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
         "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", str(path)])
    return path


def _to_wav(src: Path, path: Path) -> Path:
    run(["-y", "-i", str(src), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(path)])
    return path


def _concat_audio(pieces: list[Path], path: Path) -> Path:
    """Concat demuxer beats the filter here: no input-count limit, no relabelling."""
    listing = path.with_suffix(".txt")
    listing.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in pieces), encoding="utf-8"
    )
    run(["-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:a", "pcm_s16le", str(path)])
    return path


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    scratch = FIXTURES_DIR / "_work"

    if which in {"synthetic", "all"}:
        result = build_synthetic(FIXTURES_DIR / "synthetic.mp4")
        info = probe(result)["format"]
        print(f"synthetic.mp4  {float(info['duration']):.1f}s  (no audio)")

    if which in {"talkie", "all"}:
        result = build_talkie(FIXTURES_DIR / "talkie.mp4", scratch)
        info = probe(result)["format"]
        print(f"talkie.mp4     {float(info['duration']):.1f}s  "
              f"({len(NARRATION)} spoken lines)")
