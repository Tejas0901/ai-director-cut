"""Stage 3 - sample frames and score visual energy.

No neural network here on purpose: frame differencing plus a histogram
comparison runs in a couple of seconds on CPU and is more than enough signal
for "where does something actually happen". Face detection uses the Haar
cascade that ships inside OpenCV, so there is nothing to download.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

from ..config import (
    BUCKET_SECONDS,
    DIRECTOR_FRAME_WIDTH,
    DIRECTOR_FRAMES,
    VISION_SAMPLE_FPS,
)
from ..schemas import Keyframe, VisualBucket

SCENE_CUT_CORRELATION = 0.55  # below this, the frame content changed hard
_FACE_CASCADE: cv2.CascadeClassifier | None = None
_FACE_CASCADE_TRIED = False


def _face_cascade() -> cv2.CascadeClassifier | None:
    """Load the bundled frontal-face cascade once; tolerate its absence."""
    global _FACE_CASCADE, _FACE_CASCADE_TRIED
    if _FACE_CASCADE_TRIED:
        return _FACE_CASCADE
    _FACE_CASCADE_TRIED = True
    try:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if path.exists():
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                _FACE_CASCADE = cascade
    except Exception:
        _FACE_CASCADE = None
    return _FACE_CASCADE


def analyze(video_path: str | Path, duration: float) -> list[VisualBucket]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    cascade = _face_cascade()
    step = 1.0 / max(VISION_SAMPLE_FPS, 0.5)

    samples: list[tuple[float, float, bool, int]] = []  # t, motion_raw, cut, faces
    prev_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None

    t = 0.0
    try:
        while t < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            motion_raw = 0.0
            if prev_gray is not None:
                motion_raw = float(np.mean(cv2.absdiff(gray, prev_gray)))

            hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None,
                                [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            scene_cut = False
            if prev_hist is not None:
                correlation = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                scene_cut = correlation < SCENE_CUT_CORRELATION

            faces = 0
            if cascade is not None:
                detections = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                      minSize=(24, 24))
                faces = len(detections)

            samples.append((t, motion_raw, scene_cut, faces))
            prev_gray, prev_hist = gray, hist
            t += step
    finally:
        capture.release()

    return _to_buckets(samples, duration)


def keyframes(video_path: str | Path, duration: float,
              count: int = DIRECTOR_FRAMES,
              width: int = DIRECTOR_FRAME_WIDTH) -> list[Keyframe]:
    """Evenly spaced stills for a Director that can actually look at them.

    Even spacing rather than the high-energy peaks, deliberately. The peaks
    are already what the clip picker selects on, and they are the least
    representative part of a video - sampling the whole span is what catches
    the setting, a title card or on-screen branding that the busy moments
    never show.

    Samples land at segment midpoints so the first frame is not the black
    one every encoder starts on. Never raises: a Director with no pictures
    still writes a plan, it just writes a vaguer one.
    """
    if count <= 0 or duration <= 0:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[vision] could not open {video_path} for keyframes")
        return []

    frames: list[Keyframe] = []
    try:
        for i in range(count):
            t = duration * (i + 0.5) / count
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            height = max(1, round(frame.shape[0] * width / frame.shape[1]))
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                continue

            frames.append(Keyframe(
                t=round(t, 2),
                jpeg_b64=base64.b64encode(buffer.tobytes()).decode("ascii"),
            ))
    except Exception as exc:  # noqa: BLE001 - pictures are an enhancement
        print(f"[vision] keyframe extraction failed ({exc}); continuing without")
    finally:
        capture.release()

    return frames


def _to_buckets(samples: list[tuple[float, float, bool, int]], duration: float) -> list[VisualBucket]:
    """Normalise motion against this video's own range and bin into buckets.

    Normalising per-video matters: absolute frame-difference values mean
    nothing across different footage, but "busy relative to the rest of THIS
    video" is exactly the signal the Director needs.
    """
    if not samples:
        return []

    motions = np.array([s[1] for s in samples], dtype=np.float32)
    ceiling = float(np.percentile(motions, 95)) or 1.0

    count = max(1, int(np.ceil(duration / BUCKET_SECONDS)))
    buckets: list[VisualBucket] = []

    for i in range(count):
        start = i * BUCKET_SECONDS
        end = start + BUCKET_SECONDS
        inside = [s for s in samples if start <= s[0] < end]
        if inside:
            motion = float(np.mean([s[1] for s in inside])) / ceiling
            buckets.append(VisualBucket(
                t=round(start, 3),
                motion=round(min(1.0, motion), 4),
                scene_cut=any(s[2] for s in inside),
                faces=max(s[3] for s in inside),
            ))
        else:
            buckets.append(VisualBucket(t=round(start, 3)))

    return buckets


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.vision <video>
    from .ffmpeg_util import probe

    target = sys.argv[1]
    dur = float(probe(target)["format"]["duration"])
    results = analyze(target, dur)
    print(f"{len(results)} buckets over {dur:.1f}s\n")
    print(f"{'t':>7}  {'motion':>7}  {'cut':>4}  {'faces':>5}  bar")
    for bucket in results:
        bar = "#" * int(bucket.motion * 40)
        print(f"{bucket.t:7.1f}  {bucket.motion:7.3f}  {'CUT' if bucket.scene_cut else '':>4}"
              f"  {bucket.faces:5d}  {bar}")
