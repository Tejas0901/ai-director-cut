"""Every fallback in the pipeline has to be visible to the person watching.

The Director drops to a heuristic and the TTS drops to silence, and both were
previously indistinguishable from a working run producing dull output. These
tests pin down that the reel still comes out AND that it says how it was made.

httpx is mocked at `director.httpx.post` rather than at the socket, because
what matters is our handling of a response, not how it was fetched.
"""

from __future__ import annotations

import json

import pytest

from backend.pipeline import director, render, tts
from backend.schemas import Clip, EditPlan, MediaInfo, Timeline, TimelineBucket, Transcript


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_timeline(duration: float = 40.0) -> Timeline:
    """A timeline with enough shape for the heuristic to pick peaks out of."""
    buckets = [
        TimelineBucket(
            t=round(i * 0.5, 2),
            audio_rms=0.9 if i % 20 < 6 else 0.2,
            motion=0.8 if i % 20 < 6 else 0.1,
            scene_cut=i % 20 == 0,
            faces=1,
        )
        for i in range(int(duration / 0.5))
    ]
    media = MediaInfo(path="in.mp4", duration=duration, width=1280, height=720,
                      fps=30.0, has_audio=True)
    return Timeline(media=media, transcript=Transcript(), buckets=buckets)


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        return self._payload


def gemini_body(clips) -> dict:
    """A well-formed Gemini response wrapping an edit plan."""
    plan = {
        "title": "Real LLM Title",
        "music_mood": "chill",
        "intro_narration": "intro",
        "outro_summary": "outro",
        "clips": clips,
    }
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(plan)}]}}]}


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """`_post_json` sleeps between retries; three real backoffs is 10s."""
    monkeypatch.setattr(director.time, "sleep", lambda _s: None)


@pytest.fixture
def gemini_key(monkeypatch):
    monkeypatch.setattr(director, "GEMINI_API_KEY", "test-key")


# --------------------------------------------------------------------------
# director
# --------------------------------------------------------------------------


def test_a_live_llm_is_reported_as_the_source(monkeypatch, gemini_key):
    body = gemini_body([{"start_time": 2.0, "end_time": 9.0,
                         "overlay_title": "Bang", "reason": "it bangs"}])
    monkeypatch.setattr(director.httpx, "post",
                        lambda *a, **k: FakeResponse(200, body))

    plan = director.direct(make_timeline(), provider="gemini")

    assert plan.source == "llm"
    assert plan.fallback_reason is None
    assert plan.title == "Real LLM Title"


def test_a_dead_key_falls_back_and_says_so(monkeypatch, gemini_key):
    """400s are not retried, so this is the one-shot failure path."""
    body = {"error": {"code": 400,
                      "message": "API key not valid. Please pass a valid API key.",
                      "status": "INVALID_ARGUMENT"}}
    monkeypatch.setattr(director.httpx, "post",
                        lambda *a, **k: FakeResponse(400, body))

    plan = director.direct(make_timeline(), provider="gemini")

    assert plan.source == "heuristic"
    assert plan.fallback_reason is not None
    assert "gemini" in plan.fallback_reason
    assert "API key not valid" in plan.fallback_reason
    # The sentence, not the envelope it arrived in.
    assert "INVALID_ARGUMENT" not in plan.fallback_reason
    assert '{"error"' not in plan.fallback_reason
    assert plan.clips, "a fallback still has to produce a reel"


def test_a_persistent_503_exhausts_retries_then_falls_back(monkeypatch, gemini_key):
    calls: list[int] = []

    def flaky(*args, **kwargs):
        calls.append(1)
        return FakeResponse(503, {}, text="the model is overloaded")

    monkeypatch.setattr(director.httpx, "post", flaky)

    plan = director.direct(make_timeline(), provider="gemini")

    assert len(calls) == director.MAX_ATTEMPTS, "503 is retryable"
    assert plan.source == "heuristic"
    assert "503" in plan.fallback_reason
    assert plan.clips


def test_a_transport_error_falls_back(monkeypatch, gemini_key):
    """No internet at all - the case the README promises still works."""
    def offline(*args, **kwargs):
        raise director.httpx.ConnectError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(director.httpx, "post", offline)

    plan = director.direct(make_timeline(), provider="gemini")

    assert plan.source == "heuristic"
    assert "gemini" in plan.fallback_reason
    assert plan.clips


def test_a_missing_key_falls_back_without_calling_out(monkeypatch):
    monkeypatch.setattr(director, "GEMINI_API_KEY", "")
    called: list[int] = []
    monkeypatch.setattr(director.httpx, "post",
                        lambda *a, **k: called.append(1) or FakeResponse(200, {}))

    plan = director.direct(make_timeline(), provider="gemini")

    assert not called, "no key means no request"
    assert plan.source == "heuristic"
    assert "GEMINI_API_KEY" in plan.fallback_reason


def test_an_unusable_plan_falls_back_rather_than_raising(monkeypatch, gemini_key):
    """Every clip past the end of the video sanitises down to nothing.

    That used to escape `direct` as a ValueError and fail the whole job, which
    breaks the promise that the app always produces a reel.
    """
    body = gemini_body([{"start_time": 900.0, "end_time": 950.0,
                         "overlay_title": "Nope", "reason": "hallucinated"}])
    monkeypatch.setattr(director.httpx, "post",
                        lambda *a, **k: FakeResponse(200, body))

    plan = director.direct(make_timeline(duration=40.0), provider="gemini")

    assert plan.source == "heuristic"
    assert "no usable clips" in plan.fallback_reason
    assert plan.clips


def test_malformed_json_falls_back(monkeypatch, gemini_key):
    body = {"candidates": [{"content": {"parts": [{"text": "sorry, I can't"}]}}]}
    monkeypatch.setattr(director.httpx, "post",
                        lambda *a, **k: FakeResponse(200, body))

    plan = director.direct(make_timeline(), provider="gemini")

    assert plan.source == "heuristic"
    assert plan.clips


def test_mock_provider_is_heuristic_but_not_a_failure(monkeypatch):
    """Choosing the heuristic deliberately is not a degradation to report."""
    plan = director.direct(make_timeline(), provider="mock")

    assert plan.source == "heuristic"
    assert plan.fallback_reason is None


def test_the_fallback_reason_stays_short_enough_for_a_badge(monkeypatch, gemini_key):
    monkeypatch.setattr(director.httpx, "post",
                        lambda *a, **k: FakeResponse(400, {}, text="x" * 500))

    plan = director.direct(make_timeline(), provider="gemini")

    assert len(plan.fallback_reason) <= director.MAX_REASON_CHARS + len("gemini: ")
    assert plan.fallback_reason.endswith("…")


def test_a_hallucinated_source_field_cannot_forge_the_badge(monkeypatch, gemini_key):
    """Groq's plain JSON mode lets a model return whatever keys it likes."""
    plan_json = {
        "title": "Sneaky", "music_mood": "chill",
        "intro_narration": "i", "outro_summary": "o",
        "source": "llm", "fallback_reason": "totally fine",
        "clips": [{"start_time": 900.0, "end_time": 950.0,
                   "overlay_title": "t", "reason": "r"}],
    }
    body = {"candidates": [{"content": {"parts": [{"text": json.dumps(plan_json)}]}}]}
    monkeypatch.setattr(director.httpx, "post",
                        lambda *a, **k: FakeResponse(200, body))

    plan = director.direct(make_timeline(duration=40.0), provider="gemini")

    # The clips were unusable, so this is a heuristic cut whatever it claimed.
    assert plan.source == "heuristic"
    assert plan.fallback_reason != "totally fine"


# --------------------------------------------------------------------------
# tts
# --------------------------------------------------------------------------


@pytest.fixture
def silent_ffmpeg(monkeypatch, tmp_path):
    """Stub `_silence`'s ffmpeg call so these tests stay pure Python."""
    def fake_run(args, **kwargs):
        from pathlib import Path
        Path(args[-1]).write_bytes(b"\x00" * 64)
        return ""

    monkeypatch.setattr(tts, "run", fake_run)


def test_working_tts_reports_spoken(monkeypatch, tmp_path, silent_ffmpeg):
    monkeypatch.setattr(tts, "_edge", lambda text, out: out)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert result.spoken is True
    assert result.detail is None


def test_failing_tts_substitutes_silence_and_reports_why(monkeypatch, tmp_path, silent_ffmpeg):
    def boom(text, out):
        raise RuntimeError("edge-tts produced an empty file")

    monkeypatch.setattr(tts, "_edge", boom)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert result.spoken is False
    assert "edge-tts produced an empty file" in result.detail
    assert result.path.exists(), "silence still has to be playable"


def test_mock_tts_is_silent_and_says_so(tmp_path, silent_ffmpeg):
    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="mock")

    assert result.spoken is False
    assert "mock" in result.detail


def test_nothing_to_say_is_not_a_failure(tmp_path, silent_ffmpeg):
    result = tts.synthesize("   ", tmp_path / "intro.mp3", provider="edge")

    assert result.spoken is True
    assert result.detail is None


# --------------------------------------------------------------------------
# render -> the warning that reaches the Job
# --------------------------------------------------------------------------


def one_clip_plan() -> EditPlan:
    return EditPlan(title="t", intro_narration="hello", outro_summary="bye",
                    clips=[Clip(start_time=0.0, end_time=5.0)])


def test_render_surfaces_a_narration_failure_as_a_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "cut_reel", lambda *a, **k: tmp_path / "cut.mp4")
    monkeypatch.setattr(render, "mix_audio", lambda *a, **k: tmp_path / "final.mp4")
    monkeypatch.setattr(render, "music_for", lambda mood: None)
    monkeypatch.setattr(render, "synthesize", lambda text, out, **k: tts.Narration(
        out, spoken=False, detail="edge: network unreachable"))

    result = render.render(one_clip_plan(), MediaInfo(
        path="in.mp4", duration=10.0, width=1280, height=720, fps=30.0,
        has_audio=True), "job123")

    assert result.narration_ok is False
    assert len(result.warnings) == 1, "one message, not one per narration half"
    assert "network unreachable" in result.warnings[0]
    assert result.path == tmp_path / "final.mp4"


def test_a_clean_render_carries_no_warnings(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "cut_reel", lambda *a, **k: tmp_path / "cut.mp4")
    monkeypatch.setattr(render, "mix_audio", lambda *a, **k: tmp_path / "final.mp4")
    monkeypatch.setattr(render, "music_for", lambda mood: None)
    monkeypatch.setattr(render, "synthesize",
                        lambda text, out, **k: tts.Narration(out, spoken=True))

    result = render.render(one_clip_plan(), MediaInfo(
        path="in.mp4", duration=10.0, width=1280, height=720, fps=30.0,
        has_audio=True), "job123")

    assert result.narration_ok is True
    assert result.warnings == ()
