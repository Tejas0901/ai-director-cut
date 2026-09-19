"""Thin wrapper around the ffmpeg/ffprobe binaries.

Every subprocess call in the project goes through here so failures surface as
one exception type carrying the tail of stderr - which is the only part of
ffmpeg's output that ever tells you what actually went wrong.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import FFMPEG, FFPROBE


class FFmpegError(RuntimeError):
    pass


def run(args: list[str], *, binary: str | None = None) -> str:
    cmd = [binary or FFMPEG, *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise FFmpegError(f"{Path(cmd[0]).name} failed ({proc.returncode}):\n{tail}")
    return proc.stdout


def probe(path: str | Path) -> dict:
    out = run(
        [
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        binary=FFPROBE,
    )
    return json.loads(out)


def parse_fps(rate: str | None) -> float:
    """ffprobe reports frame rate as a rational string like '30000/1001'."""
    if not rate:
        return 30.0
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 30.0
        return float(rate)
    except (ValueError, ZeroDivisionError):
        return 30.0
