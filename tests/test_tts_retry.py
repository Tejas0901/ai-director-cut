"""edge-tts intermittently returns no audio, so narration retries.

Microsoft's endpoint accepts the socket and then closes it having sent zero
audio frames - no error, nothing to react to. Roughly one call in twelve, and
the failures are NOT independent: they arrive in bursts of one or two,
separated by long clean runs.

Length, voice, content and idle time were each measured against it and ruled
out, so there is no input to avoid - only a window to outlast. That is why
the budget is five attempts spanning ~15s rather than three inside 2.25s.
"""

from __future__ import annotations

import pytest

from backend.pipeline import tts


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    """Two real backoffs is 2.25s per test."""
    monkeypatch.setattr(tts.time, "sleep", lambda _s: None)


@pytest.fixture
def silent_ffmpeg(monkeypatch):
    """Stub the ffmpeg call behind `_silence` so these stay pure Python."""
    def fake_run(args, **kwargs):
        from pathlib import Path
        Path(args[-1]).write_bytes(b"\x00" * 64)
        return ""

    monkeypatch.setattr(tts, "run", fake_run)


class FlakyEndpoint:
    """Fails its first `failures` calls, then writes a plausible mp3."""

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def __call__(self, text, out_path):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(
                "No audio was received. Please verify that your parameters are correct.")
        out_path.write_bytes(b"ID3" + b"\x00" * 128)


def test_a_single_flake_is_retried_into_a_success(monkeypatch, tmp_path):
    endpoint = FlakyEndpoint(failures=1)
    monkeypatch.setattr(tts, "_edge_once", endpoint)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert endpoint.calls == 2
    assert result.spoken is True, "the retry succeeded, so this is real speech"
    assert result.detail is None


def test_a_burst_of_consecutive_failures_is_ridden_out(monkeypatch, tmp_path):
    """Two in a row was the longest burst measured; leave headroom past it."""
    endpoint = FlakyEndpoint(failures=3)
    monkeypatch.setattr(tts, "_edge_once", endpoint)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert endpoint.calls == 4, "three failures then the fourth call lands"
    assert result.spoken is True


def test_a_genuine_outage_gives_up_and_reports_the_attempts(monkeypatch, tmp_path, silent_ffmpeg):
    endpoint = FlakyEndpoint(failures=99)
    monkeypatch.setattr(tts, "_edge_once", endpoint)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert endpoint.calls == tts.EDGE_MAX_ATTEMPTS, "bounded, not an infinite loop"
    assert result.spoken is False
    assert f"{tts.EDGE_MAX_ATTEMPTS} attempts" in result.detail
    assert "No audio was received" in result.detail, "keep the endpoint's own words"
    assert result.path.exists(), "silence still has to be playable"


def test_a_working_endpoint_is_called_exactly_once(monkeypatch, tmp_path):
    endpoint = FlakyEndpoint(failures=0)
    monkeypatch.setattr(tts, "_edge_once", endpoint)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert endpoint.calls == 1, "no retry cost on the happy path"
    assert result.spoken is True


def test_an_empty_file_counts_as_a_failure(monkeypatch, tmp_path):
    """A failed call can leave a zero-byte mp3, which must not reach the mixer.

    `_edge_once` is the real one here - only the network call underneath it
    is stubbed - so this exercises the size check itself.
    """
    class EmptyThenReal:
        def __init__(self):
            self.calls = 0

        def save_to(self, path):
            self.calls += 1
            path.write_bytes(b"" if self.calls == 1 else b"ID3" + b"\x00" * 128)

    endpoint = EmptyThenReal()

    def fake_once(text, out_path):
        endpoint.save_to(out_path)
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError("edge-tts produced an empty file")

    monkeypatch.setattr(tts, "_edge_once", fake_once)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert endpoint.calls == 2
    assert result.spoken is True
    assert result.path.stat().st_size > 0


def test_piper_is_not_retried(monkeypatch, tmp_path, silent_ffmpeg):
    """A missing binary or voice model fails identically every time."""
    calls: list[int] = []

    def boom(text, out_path):
        calls.append(1)
        raise RuntimeError("TTS_PROVIDER=piper requires PIPER_BIN and PIPER_VOICE")

    monkeypatch.setattr(tts, "_piper", boom)

    result = tts.synthesize("some words", tmp_path / "intro.mp3", provider="piper")

    assert len(calls) == 1, "retrying a deterministic failure just adds delay"
    assert result.spoken is False


# --------------------------------------------------------------------------
# burst tolerance
#
# The failures are not independent: they arrive in bursts of one or two calls
# separated by long clean runs. Length, voice, content and idle time were each
# measured and ruled out, so there is nothing to avoid - only a window to
# outlast. Three attempts 0.75s apart all fell inside the same burst.
# --------------------------------------------------------------------------


def test_the_retries_span_long_enough_to_outlast_a_burst(monkeypatch, tmp_path):
    slept: list[float] = []
    monkeypatch.setattr(tts.time, "sleep", slept.append)
    monkeypatch.setattr(tts, "_edge_once", FlakyEndpoint(failures=99))
    monkeypatch.setattr(tts, "run", lambda args, **kw: __import__("pathlib")
                        .Path(args[-1]).write_bytes(b"\x00" * 64))

    tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    # The old budget was 2.25s, which a burst outlived.
    assert sum(slept) >= 10.0, f"attempts span only {sum(slept):.1f}s"


def test_backoff_grows_rather_than_hammering(monkeypatch, tmp_path):
    slept: list[float] = []
    monkeypatch.setattr(tts.time, "sleep", slept.append)
    monkeypatch.setattr(tts, "_edge_once", FlakyEndpoint(failures=99))
    monkeypatch.setattr(tts, "run", lambda args, **kw: __import__("pathlib")
                        .Path(args[-1]).write_bytes(b"\x00" * 64))

    tts.synthesize("some words", tmp_path / "intro.mp3", provider="edge")

    assert slept == sorted(slept)
    assert len(slept) == tts.EDGE_MAX_ATTEMPTS - 1
