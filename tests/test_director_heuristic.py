"""The offline Director and the HTTP layer underneath the online one.

`_mock` is what runs with no key and no internet, so it is the thing standing
between an offline machine and no reel at all. `_post_json` decides which
failures are worth another try - too eager and a dead key costs three
timeouts, too shy and one 503 loses the demo.
"""

from __future__ import annotations

import json

import httpx
import pytest

from backend.pipeline import director
from backend.schemas import (
    MediaInfo,
    MusicMood,
    Timeline,
    TimelineBucket,
    Transcript,
    TranscriptSegment,
)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Three real backoffs is 4.5s per test."""
    monkeypatch.setattr(director.time, "sleep", lambda _s: None)


def timeline_of(buckets: list[TimelineBucket], duration: float = 60.0,
                transcript: Transcript | None = None) -> Timeline:
    return Timeline(
        media=MediaInfo(path="in.mp4", duration=duration, width=1280, height=720,
                        fps=30.0, has_audio=True),
        transcript=transcript or Transcript(),
        buckets=buckets,
    )


def busy(n: int = 120, **kw) -> list[TimelineBucket]:
    """A timeline with four clear energy spikes."""
    out = []
    for i in range(n):
        peak = (i // 10) % 3 == 0
        out.append(TimelineBucket(t=round(i * 0.5, 2),
                                  audio_rms=0.9 if peak else 0.1,
                                  motion=0.9 if peak else 0.1,
                                  scene_cut=peak and i % 10 == 0,
                                  **kw))
    return out


# --------------------------------------------------------------------------
# _mock
# --------------------------------------------------------------------------


def test_the_heuristic_always_produces_a_usable_plan():
    plan = director._mock(timeline_of(busy()))

    assert plan.clips
    assert plan.title
    assert plan.intro_narration and plan.outro_summary
    assert all(c.end_time > c.start_time for c in plan.clips)


def test_the_heuristic_stays_inside_the_video():
    plan = director._mock(timeline_of(busy(60), duration=30.0))

    assert all(0 <= c.start_time < c.end_time <= 30.0 for c in plan.clips)


def test_an_empty_timeline_still_yields_one_clip():
    """No buckets at all - a video too short or too odd to analyse."""
    plan = director._mock(timeline_of([], duration=10.0))

    assert len(plan.clips) == 1
    assert plan.clips[0].start_time == 0.0
    assert plan.clips[0].end_time == 6.0


def test_a_video_shorter_than_the_default_clip_is_not_over_run():
    plan = director._mock(timeline_of([], duration=2.5))

    assert plan.clips[0].end_time == 2.5


def test_every_clip_gets_a_reason():
    plan = director._mock(timeline_of(busy()))

    assert all(c.reason for c in plan.clips)


def test_windows_with_different_character_get_different_reasons():
    """The UI prints one reason per clip, so four identical sentences read as
    a bug. The reason names whichever signal actually won that window - which
    only varies when the windows themselves do."""
    buckets: list[TimelineBucket] = []
    for i in range(80):
        t = round(i * 0.5, 2)
        if i < 20:        # loud dialogue
            buckets.append(TimelineBucket(t=t, audio_rms=0.95, motion=0.1,
                                          speech="and then it completely exploded"))
        elif i < 40:      # rapid cutting
            buckets.append(TimelineBucket(t=t, audio_rms=0.2, motion=0.3,
                                          scene_cut=i % 2 == 0))
        elif i < 60:      # pure movement
            buckets.append(TimelineBucket(t=t, audio_rms=0.1, motion=0.95))
        else:             # faces, calm
            buckets.append(TimelineBucket(t=t, audio_rms=0.2, motion=0.2, faces=2))

    tl = timeline_of(buckets)
    reasons = [director._heuristic_reason(tl, start, start + 8.0)
               for start in (0.0, 10.0, 20.0, 30.0)]

    assert len(set(reasons)) == 4, f"reasons did not differentiate: {reasons}"


def test_a_spoken_window_is_explained_by_its_loudest_line():
    buckets = [TimelineBucket(t=round(i * 0.5, 2), audio_rms=0.9, motion=0.2,
                              speech="and then it exploded" if i < 20 else "")
               for i in range(60)]

    reason = director._heuristic_reason(timeline_of(buckets), 0.0, 8.0)

    assert "exploded" in reason


def test_a_window_with_no_data_says_so_rather_than_inventing():
    reason = director._heuristic_reason(timeline_of([]), 0.0, 5.0)

    assert "Fallback" in reason


def test_overlay_titles_are_assigned_and_non_empty():
    plan = director._mock(timeline_of(busy()))

    assert all(c.overlay_title for c in plan.clips)


# --- mood ---------------------------------------------------------------


def test_loud_and_busy_footage_gets_the_energetic_bed():
    buckets = [TimelineBucket(t=i * 0.5, motion=0.8, audio_rms=0.8) for i in range(40)]

    assert director._heuristic_mood(timeline_of(buckets)) is MusicMood.ENERGETIC


def test_talky_and_still_footage_gets_the_chill_bed():
    buckets = [TimelineBucket(t=i * 0.5, motion=0.1, audio_rms=0.3, speech="words")
               for i in range(40)]

    assert director._heuristic_mood(timeline_of(buckets)) is MusicMood.CHILL


def test_quiet_and_wordless_footage_gets_the_dramatic_bed():
    buckets = [TimelineBucket(t=i * 0.5, motion=0.2, audio_rms=0.1) for i in range(40)]

    assert director._heuristic_mood(timeline_of(buckets)) is MusicMood.DRAMATIC


def test_an_empty_timeline_has_a_mood_anyway():
    assert director._heuristic_mood(timeline_of([])) is MusicMood.ENERGETIC


# --------------------------------------------------------------------------
# _post_json
# --------------------------------------------------------------------------


class Response:
    def __init__(self, status_code, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


@pytest.fixture
def post(monkeypatch):
    """Queue up responses; returns the list of calls actually made."""
    calls: list[dict] = []

    def install(*responses):
        queue = list(responses)

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append({"url": url, "json": json})
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(director.httpx, "post", fake_post)
        return calls

    return install


def test_a_successful_call_is_made_once(post):
    calls = post(Response(200, {"ok": True}))

    assert director._post_json("u", {}, {}) == {"ok": True}
    assert len(calls) == 1


def test_a_503_is_retried_and_can_succeed(post):
    calls = post(Response(503, text="high demand"), Response(200, {"ok": True}))

    assert director._post_json("u", {}, {}) == {"ok": True}
    assert len(calls) == 2


@pytest.mark.parametrize("status", sorted(director.RETRY_STATUS))
def test_every_transient_status_is_retried(status, post):
    calls = post(Response(status, text="transient"))

    with pytest.raises(RuntimeError):
        director._post_json("u", {}, {})

    assert len(calls) == director.MAX_ATTEMPTS


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_permanent_failure_is_not_retried(status, post):
    """A bad key or a retired model fails identically however often you ask."""
    calls = post(Response(status, text="nope"))

    with pytest.raises(RuntimeError):
        director._post_json("u", {}, {})

    assert len(calls) == 1


def test_a_transport_error_is_retried(post):
    calls = post(httpx.ConnectError("getaddrinfo failed"))

    with pytest.raises(RuntimeError):
        director._post_json("u", {}, {})

    assert len(calls) == director.MAX_ATTEMPTS


def test_exhausting_the_attempts_reports_the_count_and_the_last_error(post):
    post(Response(503, text="the model is overloaded"))

    with pytest.raises(RuntimeError) as exc:
        director._post_json("u", {}, {})

    assert f"all {director.MAX_ATTEMPTS} attempts failed" in str(exc.value)
    assert "overloaded" in str(exc.value)


def test_backoff_grows_between_attempts(monkeypatch, post):
    slept: list[float] = []
    monkeypatch.setattr(director.time, "sleep", slept.append)
    post(Response(503, text="busy"))

    with pytest.raises(RuntimeError):
        director._post_json("u", {}, {})

    assert len(slept) == director.MAX_ATTEMPTS - 1, "no sleep after the last try"
    assert slept == sorted(slept) and slept[0] < slept[-1]


# --- _describe ----------------------------------------------------------


def test_the_providers_own_message_is_lifted_out_of_the_envelope():
    body = {"error": {"code": 400, "message": "API key not valid",
                      "status": "INVALID_ARGUMENT"}}

    described = director._describe(Response(400, body))

    assert described == "HTTP 400: API key not valid"


def test_a_body_that_is_not_the_expected_shape_falls_back_to_raw_text():
    described = director._describe(Response(500, {}, text="<html>gateway error</html>"))

    assert "500" in described and "gateway error" in described


def test_a_very_long_raw_body_is_truncated():
    described = director._describe(Response(500, {}, text="x" * 900))

    assert len(described) < 300


# --------------------------------------------------------------------------
# heuristic narration
#
# The fallback used to return two fixed sentences and the title "The
# Director's Cut" on every reel the app had ever produced, which is exactly
# how a fallback gives itself away as one. It cannot know what the footage is
# of - that is the LLM's job and it is unavailable here - so it reports what
# it measured instead, and those measurements differ per video.
# --------------------------------------------------------------------------


def speechy(n: int = 80) -> list[TimelineBucket]:
    return [TimelineBucket(t=round(i * 0.5, 2), audio_rms=0.7, motion=0.1,
                           speech="talking here") for i in range(n)]


def cutty(n: int = 80) -> list[TimelineBucket]:
    return [TimelineBucket(t=round(i * 0.5, 2), audio_rms=0.2, motion=0.2,
                           scene_cut=i % 3 == 0) for i in range(n)]


def movey(n: int = 80) -> list[TimelineBucket]:
    return [TimelineBucket(t=round(i * 0.5, 2), audio_rms=0.1, motion=0.9)
            for i in range(n)]


def test_two_different_videos_do_not_get_the_same_narration():
    """The bug this replaces: identical copy on every reel."""
    a = director._mock(timeline_of(speechy(80), duration=40.0,
                                   transcript=Transcript(segments=[
                                       TranscriptSegment(start=0, end=5, text="hello")])))
    b = director._mock(timeline_of(movey(240), duration=120.0))

    assert a.intro_narration != b.intro_narration
    assert a.outro_summary != b.outro_summary
    assert a.title != b.title


def test_the_old_boilerplate_is_gone():
    plan = director._mock(timeline_of(busy()))

    assert plan.title != "The Director's Cut"
    assert "nothing that isn't" not in plan.intro_narration
    assert "You're welcome" not in plan.outro_summary


def test_the_narration_reports_the_real_numbers():
    tl = timeline_of(movey(240), duration=120.0)

    plan = director._mock(tl)

    assert "2 minutes" in plan.intro_narration, plan.intro_narration
    # The count opens the sentence, so it arrives capitalised.
    assert director._spell(len(plan.clips)) in plan.intro_narration.lower()


def test_the_intro_opens_with_a_capital():
    """It is a spoken sentence and a headline; "four moments out of..." is neither."""
    plan = director._mock(timeline_of(movey(240), duration=120.0))

    assert plan.intro_narration[0].isupper(), plan.intro_narration
    assert plan.outro_summary[0].isupper(), plan.outro_summary


def test_a_video_of_a_different_length_reads_differently():
    short = director._mock(timeline_of(movey(60), duration=30.0))
    long = director._mock(timeline_of(movey(400), duration=200.0))

    assert short.intro_narration != long.intro_narration


@pytest.mark.parametrize("buckets,transcript,expected", [
    (speechy(), Transcript(segments=[TranscriptSegment(start=0, end=5, text="hi")]), "speech"),
    (cutty(), None, "cuts"),
    (movey(), None, "motion"),
    ([TimelineBucket(t=i * 0.5, audio_rms=0.8, motion=0.1) for i in range(80)], None, "loudness"),
    ([TimelineBucket(t=i * 0.5, audio_rms=0.1, motion=0.1) for i in range(80)], None, "energy"),
])
def test_the_dominant_signal_is_identified(buckets, transcript, expected):
    assert director._dominant_signal(timeline_of(buckets, transcript=transcript)) == expected


def test_speech_is_not_claimed_when_there_is_no_transcript():
    """Buckets can carry speech text while the transcript is empty only if
    something upstream is wrong; either way, do not promise dialogue."""
    assert director._dominant_signal(timeline_of(speechy())) != "speech"


def test_an_empty_timeline_still_gets_narration():
    plan = director._mock(timeline_of([], duration=8.0))

    assert plan.title and plan.intro_narration and plan.outro_summary


def test_the_narration_invents_no_subject():
    """The heuristic is blind. It may describe measurements, never content."""
    plan = director._mock(timeline_of(movey(240), duration=120.0))

    words = (plan.title + " " + plan.intro_narration + " " + plan.outro_summary).lower()
    for invented in ("chess", "squash", "family", "game", "match", "team", "player"):
        assert invented not in words


def test_the_intro_stays_within_the_narration_budget():
    plan = director._mock(timeline_of(movey(240), duration=120.0))

    assert len(plan.intro_narration.split()) <= 30
    assert len(plan.outro_summary.split()) <= 20


@pytest.mark.parametrize("seconds,expected", [
    (5.0, "5 seconds"),
    (45.4, "45 seconds"),
    (89.0, "89 seconds"),
    (95.0, "a minute and a half"),
    (120.0, "2 minutes"),
    (185.0, "3 minutes"),
])
def test_durations_are_spoken_the_way_a_person_would_say_them(seconds, expected):
    assert director._say_duration(seconds) == expected


def test_a_sub_second_clip_does_not_read_as_zero_seconds():
    assert director._say_duration(0.2) == "1 seconds"
