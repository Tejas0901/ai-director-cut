"""Stage 7 - turn an EditPlan into an actual mp4.

Deliberately two passes:

  Pass 1 (cut)  - trim every clip, normalise it, concat, burn overlay titles.
  Pass 2 (mix)  - lay the narration over the head and tail, duck the original
                  audio underneath it, and bed in background music.

One giant filter_complex would be marginally faster and dramatically harder to
debug. When a render breaks you want to know which half broke, and you want to
be able to watch the intermediate file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import (
    DUCK_VOLUME,
    MUSIC_DIR,
    MUSIC_VOLUME,
    OUTPUT_DIR,
    RENDER_FPS,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    ffmpeg_fontfile,
)
from ..schemas import EditPlan, MediaInfo
from .ffmpeg_util import probe, run
from .tts import duration_of, synthesize

TITLE_HOLD_SECONDS = 3.0


def render(plan: EditPlan, media: MediaInfo, job_id: str,
           *, with_audio_mix: bool = True) -> Path:
    """Full render. Returns the path of the finished reel."""
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    cut_path = cut_reel(plan, media, work_dir / "reel_cut.mp4")
    if not with_audio_mix:
        return cut_path

    intro_path = synthesize(plan.intro_narration, work_dir / "intro.mp3")
    outro_path = synthesize(plan.outro_summary, work_dir / "outro.mp3")

    return mix_audio(
        cut_path,
        intro_path,
        outro_path,
        music_for(plan.music_mood.value),
        work_dir / "final.mp4",
    )


# --------------------------------------------------------------------------
# Pass 1 - the cut
# --------------------------------------------------------------------------


def cut_reel(plan: EditPlan, media: MediaInfo, out_path: Path) -> Path:
    """Trim, normalise and concatenate the chosen clips.

    Normalising every segment to identical resolution / SAR / frame rate /
    sample rate before `concat` is what prevents the classic garbled-output
    bug when the source has variable frame rate or odd pixel aspect.
    """
    inputs = ["-i", media.path]
    if not media.has_audio:
        # concat needs an audio stream on every segment; synthesise one.
        inputs += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
    audio_src = "0:a" if media.has_audio else "1:a"

    parts: list[str] = []
    labels: list[str] = []

    for i, clip in enumerate(plan.clips):
        title = _escape_drawtext(clip.overlay_title)
        drawtext = ""
        if title:
            drawtext = (
                f",drawtext=fontfile='{ffmpeg_fontfile()}':text='{title}'"
                f":fontcolor=white:fontsize=46:box=1:boxcolor=black@0.45:boxborderw=20"
                f":x=(w-text_w)/2:y=h-150"
                f":enable='between(t,0.25,{TITLE_HOLD_SECONDS})'"
            )

        parts.append(
            f"[0:v]trim=start={clip.start_time}:end={clip.end_time},"
            f"setpts=PTS-STARTPTS,"
            f"scale={RENDER_WIDTH}:{RENDER_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={RENDER_WIDTH}:{RENDER_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={RENDER_FPS}{drawtext}[v{i}]"
        )
        parts.append(
            f"[{audio_src}]atrim=start={clip.start_time}:end={clip.end_time},"
            f"asetpts=PTS-STARTPTS,aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo[a{i}]"
        )
        labels.append(f"[v{i}][a{i}]")

    parts.append(f"{''.join(labels)}concat=n={len(plan.clips)}:v=1:a=1[outv][outa]")

    run([
        "-y", *inputs,
        "-filter_complex", ";".join(parts),
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_path),
    ])
    return out_path


def _escape_drawtext(text: str) -> str:
    """drawtext treats ' : \\ % specially. Strip rather than escape.

    Escaping these correctly through Python -> shell -> ffmpeg filter parsing
    is three layers of quoting and a reliable source of silent failures. An
    on-screen caption does not need punctuation badly enough to risk it.
    """
    cleaned = "".join(c for c in (text or "") if c.isalnum() or c in " -?!,.")
    return cleaned.strip()[:48]


# --------------------------------------------------------------------------
# Pass 2 - the mix
# --------------------------------------------------------------------------


def mix_audio(video_path: Path, intro_path: Path, outro_path: Path,
              music_path: Path | None, out_path: Path) -> Path:
    """Overlay narration on head and tail, duck the bed, add music."""
    total = float(probe(video_path)["format"]["duration"])
    intro_len = min(duration_of(intro_path), max(total - 0.5, 0.5))
    outro_len = min(duration_of(outro_path), max(total - intro_len - 0.5, 0.5))
    outro_start = max(intro_len + 0.5, total - outro_len)

    inputs = ["-i", str(video_path), "-i", str(intro_path), "-i", str(outro_path)]
    if music_path is not None:
        inputs += ["-stream_loop", "-1", "-i", str(music_path)]

    # Duck the original audio only while narration is actually speaking.
    # Two timeline-gated volume filters beat sidechaincompress here: one
    # number to tune, and it behaves identically on every input.
    parts = [
        f"[0:a]volume=enable='between(t,0,{intro_len:.2f})':volume={DUCK_VOLUME},"
        f"volume=enable='between(t,{outro_start:.2f},{total:.2f})':volume={DUCK_VOLUME}[bed]",
        f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"adelay=0|0[intro]",
        f"[2:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
        f"adelay={int(outro_start * 1000)}|{int(outro_start * 1000)}[outro]",
    ]
    mix_labels = "[bed][intro][outro]"
    mix_count = 3

    if music_path is not None:
        parts.append(
            f"[3:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
            f"volume={MUSIC_VOLUME},atrim=0:{total:.2f},asetpts=PTS-STARTPTS[music]"
        )
        mix_labels += "[music]"
        mix_count = 4

    parts.append(
        f"{mix_labels}amix=inputs={mix_count}:duration=first:dropout_transition=0"
        f":normalize=0,alimiter=limit=0.95,aresample=48000[aout]"
    )

    run([
        "-y", *inputs,
        "-filter_complex", ";".join(parts),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-t", f"{total:.2f}",
        str(out_path),
    ])
    return out_path


def music_for(mood: str) -> Path | None:
    """Pick the bed the Director asked for; silently skip if absent."""
    for ext in (".mp3", ".m4a", ".wav", ".ogg"):
        candidate = MUSIC_DIR / f"{mood}{ext}"
        if candidate.exists() and candidate.stat().st_size > 1024:
            return candidate
    return None


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.render <video> [--cut-only]
    from ..cache import file_key
    from ..schemas import Transcript
    from .director import direct
    from .ingest import ingest
    from .timeline import build
    from .transcribe import transcribe
    from .vision import analyze

    target = sys.argv[1]
    cut_only = "--cut-only" in sys.argv

    key = file_key(target)
    media_info = ingest(target, key)
    tl = build(
        media_info,
        transcribe(media_info.audio_path) if media_info.audio_path else Transcript(),
        analyze(target, media_info.duration),
    )
    edit_plan = direct(tl)
    print(edit_plan.model_dump_json(indent=2))

    result = render(edit_plan, media_info, "cli_test", with_audio_mix=not cut_only)
    print(f"\nwrote {result}")
