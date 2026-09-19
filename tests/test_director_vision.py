"""The Director has to be able to see, and has to know when it cannot.

A timeline of loudness and motion says where the notable moments are and
nothing about what the video is of. Handed only numbers, a model fills the
gap with the most common shape of video - which is how a chess match came
back captioned "Peak Intensity" on a reel titled "High Octane Highlights".

So: frames go to the multimodal provider, and the text-only provider is told
in the prompt that it is blind.
"""

from __future__ import annotations

import base64
import json
import subprocess

import pytest

from backend.config import FFMPEG
from backend.pipeline import director, vision
from backend.schemas import Keyframe, MediaInfo, Timeline, TimelineBucket, Transcript


@pytest.fixture
def sample_video(tmp_path):
    """Six seconds of generated colour bars. Real decode, no fixture checked in."""
    path = tmp_path / "bars.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x240:rate=15:duration=6",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def make_timeline(frames: list[Keyframe] | None = None, duration: float = 40.0) -> Timeline:
    buckets = [TimelineBucket(t=round(i * 0.5, 2), audio_rms=0.5, motion=0.5)
               for i in range(int(duration / 0.5))]
    return Timeline(
        media=MediaInfo(path="in.mp4", duration=duration, width=1280, height=720,
                        fps=30.0, has_audio=True),
        transcript=Transcript(),
        buckets=buckets,
        keyframes=frames or [],
    )


def fake_frames(n: int = 3) -> list[Keyframe]:
    return [Keyframe(t=float(i), jpeg_b64=base64.b64encode(f"frame{i}".encode()).decode())
            for i in range(n)]


def flat(text: str) -> str:
    """Collapse the prompt's hard wrapping so assertions survive a rewrap."""
    return " ".join(text.split())


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def test_keyframes_span_the_video_and_decode_as_jpeg(sample_video):
    frames = vision.keyframes(sample_video, duration=6.0, count=4, width=128)

    assert len(frames) == 4
    assert [f.t for f in frames] == [0.75, 2.25, 3.75, 5.25], "segment midpoints"

    for frame in frames:
        raw = base64.b64decode(frame.jpeg_b64)
        assert raw[:2] == b"\xff\xd8", "JPEG start-of-image marker"
        assert raw[-2:] == b"\xff\xd9", "JPEG end-of-image marker"


def test_the_first_frame_is_not_the_black_one_at_zero(sample_video):
    """Encoders habitually open on black; a midpoint sample skips it."""
    frames = vision.keyframes(sample_video, duration=6.0, count=4, width=128)

    assert frames[0].t > 0.0


def test_keyframes_can_be_switched_off(sample_video):
    assert vision.keyframes(sample_video, duration=6.0, count=0) == []


def test_an_unreadable_video_yields_no_frames_rather_than_raising(tmp_path):
    """Pictures are an enhancement; losing them must not lose the reel."""
    broken = tmp_path / "not-a-video.mp4"
    broken.write_bytes(b"certainly not an mp4")

    assert vision.keyframes(broken, duration=10.0, count=4) == []


# --------------------------------------------------------------------------
# prompt assembly
# --------------------------------------------------------------------------


def test_a_seeing_director_is_told_to_read_the_frames():
    system, _ = director._build_prompts(make_timeline(fake_frames(8)), can_see=True)

    assert "8 still frames" in flat(system)
    assert "WHAT YOU CANNOT SEE" not in flat(system)


def test_a_blind_director_is_told_not_to_invent_a_subject():
    system, _ = director._build_prompts(make_timeline(), can_see=False)

    assert "WHAT YOU CANNOT SEE" in flat(system)
    assert "still frames sampled evenly" not in flat(system)


def test_a_text_only_provider_is_blind_even_when_frames_exist():
    """The timeline carries frames for whoever can use them. Groq cannot."""
    system, _ = director._build_prompts(make_timeline(fake_frames(8)), can_see=False)

    assert "WHAT YOU CANNOT SEE" in flat(system)


def test_frames_that_failed_to_extract_fall_back_to_the_blind_note():
    system, _ = director._build_prompts(make_timeline(frames=[]), can_see=True)

    assert "WHAT YOU CANNOT SEE" in flat(system)


def test_the_prompt_says_the_numbers_are_relative():
    """The other half of the chess bug: 1.00 motion is not 'a car chase'."""
    system, _ = director._build_prompts(make_timeline(fake_frames(2)), can_see=True)

    assert "normalised against THIS video's own range" in flat(system)
    assert "Never infer subject, genre, pace or energy from them" in flat(system)


# --------------------------------------------------------------------------
# request assembly
# --------------------------------------------------------------------------


@pytest.fixture
def captured_request(monkeypatch):
    """Capture the JSON body instead of calling Gemini."""
    sent: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json or {})
        plan = {"title": "t", "music_mood": "chill", "intro_narration": "i",
                "outro_summary": "o",
                "clips": [{"start_time": 1.0, "end_time": 9.0,
                           "overlay_title": "c", "reason": "r"}]}
        return _Ok({"candidates": [{"content": {"parts": [{"text": _dumps(plan)}]}}]})

    monkeypatch.setattr(director, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(director.httpx, "post", fake_post)
    return sent


class _Ok:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def _dumps(obj) -> str:
    return json.dumps(obj)


def test_every_frame_reaches_gemini_labelled_with_its_timestamp(captured_request):
    frames = fake_frames(3)

    plan = director.direct(make_timeline(frames), provider="gemini")

    parts = captured_request["contents"][0]["parts"]
    images = [p for p in parts if "inline_data" in p]
    assert len(images) == 3
    assert all(p["inline_data"]["mime_type"] == "image/jpeg" for p in images)
    assert [p["inline_data"]["data"] for p in images] == [f.jpeg_b64 for f in frames]

    # Each image is preceded by its own timestamp label, so the model can tie
    # a picture to a row of the timeline.
    labels = [p["text"] for p in parts if "text" in p]
    assert "Frame at 0.0s:" in labels
    assert "Frame at 2.0s:" in labels
    assert plan.source == "llm"


def test_the_timeline_text_still_leads_the_request(captured_request):
    director.direct(make_timeline(fake_frames(2)), provider="gemini")

    parts = captured_request["contents"][0]["parts"]
    assert "text" in parts[0] and "Timeline:" in parts[0]["text"]


def test_a_director_with_no_frames_sends_no_images(captured_request):
    director.direct(make_timeline(), provider="gemini")

    parts = captured_request["contents"][0]["parts"]
    assert not [p for p in parts if "inline_data" in p]
