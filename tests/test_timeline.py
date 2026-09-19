"""The timeline is the artefact the Director reads, and the heuristic ranks.

`build` fuses three independent sources onto one grid; `top_moments` is the
fallback clip picker that runs whenever the LLM is unreachable, so it is the
difference between a reel and no reel on an offline machine.
"""

from __future__ import annotations

import math

from backend.config import BUCKET_SECONDS
from backend.pipeline import timeline as tl
from backend.schemas import (
    Keyframe,
    MediaInfo,
    Timeline,
    TimelineBucket,
    Transcript,
    TranscriptSegment,
    VisualBucket,
)


def media(duration: float = 30.0, audio: bool = False) -> MediaInfo:
    return MediaInfo(path="in.mp4", duration=duration, width=1280, height=720,
                     fps=30.0, has_audio=audio, audio_path=None)


def flat_timeline(energies: list[float], bucket_seconds: float = 0.5) -> Timeline:
    """A timeline whose energy profile is exactly what we say it is.

    Energy is driven through `motion` alone: audio_rms would work equally well,
    but pinning one input makes a failure point at the picker rather than at
    the weighting.
    """
    buckets = [TimelineBucket(t=round(i * bucket_seconds, 3), motion=e)
               for i, e in enumerate(energies)]
    return Timeline(media=media(len(energies) * bucket_seconds),
                    transcript=Transcript(), buckets=buckets,
                    bucket_seconds=bucket_seconds)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def test_build_covers_the_whole_duration():
    result = tl.build(media(10.0), Transcript(), [])

    assert len(result.buckets) == math.ceil(10.0 / BUCKET_SECONDS)
    assert result.buckets[0].t == 0.0
    assert result.bucket_seconds == BUCKET_SECONDS


def test_a_ragged_duration_rounds_up_to_a_whole_bucket():
    """7.3s at half-second buckets is 15 rows, not 14.6."""
    result = tl.build(media(7.3), Transcript(), [])

    assert len(result.buckets) == 15


def test_visual_buckets_land_on_their_own_timestamps():
    visual = [
        VisualBucket(t=0.0, motion=0.1),
        VisualBucket(t=1.0, motion=0.9, scene_cut=True, faces=2),
    ]

    result = tl.build(media(2.0), Transcript(), visual)

    assert result.buckets[0].motion == 0.1
    assert result.buckets[2].motion == 0.9
    assert result.buckets[2].scene_cut is True
    assert result.buckets[2].faces == 2


def test_buckets_with_no_visual_data_fall_back_to_defaults():
    result = tl.build(media(2.0), Transcript(), [VisualBucket(t=0.0, motion=0.5)])

    assert result.buckets[1].motion == 0.0
    assert result.buckets[1].scene_cut is False
    assert result.buckets[1].faces == 0


def test_speech_reaches_every_bucket_the_segment_overlaps():
    script = Transcript(segments=[TranscriptSegment(start=0.6, end=1.4, text="hello there")])

    result = tl.build(media(3.0), script, [])

    assert result.buckets[0].speech == ""        # 0.0-0.5, before the segment
    assert result.buckets[1].speech == "hello there"  # 0.5-1.0
    assert result.buckets[2].speech == "hello there"  # 1.0-1.5
    assert result.buckets[3].speech == ""        # 1.5-2.0, after it


def test_a_bucket_spanned_by_two_segments_carries_both():
    script = Transcript(segments=[
        TranscriptSegment(start=0.0, end=0.4, text="first"),
        TranscriptSegment(start=0.4, end=0.9, text="second"),
    ])

    result = tl.build(media(2.0), script, [])

    assert result.buckets[0].speech == "first second"


def test_no_audio_track_means_no_loudness_rather_than_an_error():
    result = tl.build(media(5.0, audio=False), Transcript(), [])

    assert all(b.audio_rms == 0.0 for b in result.buckets)


def test_keyframes_ride_along_when_supplied():
    frames = [Keyframe(t=1.0, jpeg_b64="abc")]

    result = tl.build(media(5.0), Transcript(), [], frames)

    assert result.keyframes == frames


def test_keyframes_default_to_empty():
    assert tl.build(media(5.0), Transcript(), []).keyframes == []


# --------------------------------------------------------------------------
# render_markdown
# --------------------------------------------------------------------------


def test_the_prompt_table_stays_bounded_for_a_long_video():
    """Row merging is what keeps the prompt roughly constant in size."""
    long_timeline = flat_timeline([0.5] * 4000)

    rows = tl.render_markdown(long_timeline, max_rows=260).splitlines()

    assert len(rows) - 2 <= 261, "two header rows plus the budget"


def test_the_table_carries_the_speech_and_the_peaks():
    buckets = [TimelineBucket(t=0.0, speech="hello", audio_rms=0.8, motion=0.4,
                              scene_cut=True, faces=2)]
    one = Timeline(media=media(0.5), transcript=Transcript(), buckets=buckets)

    table = tl.render_markdown(one)

    assert "hello" in table
    assert "0.80" in table and "0.40" in table
    assert "| Y |" in table and "| 2 |" in table


# --------------------------------------------------------------------------
# top_moments
# --------------------------------------------------------------------------


def test_top_moments_finds_the_peaks():
    # 60s of quiet with four clear spikes.
    energies = [0.05] * 120
    for centre in (10, 40, 70, 100):
        for i in range(centre, centre + 8):
            energies[i] = 1.0

    picks = tl.top_moments(flat_timeline(energies), count=4)

    assert len(picks) == 4
    starts = [s for s, _ in picks]
    for expected in (5.0, 20.0, 35.0, 50.0):  # index * 0.5
        assert any(abs(s - expected) < 2.0 for s in starts), f"missed peak at {expected}"


def test_picks_are_sorted_and_do_not_overlap():
    energies = [(i % 17) / 17 for i in range(200)]

    picks = tl.top_moments(flat_timeline(energies), count=4)

    assert picks == sorted(picks)
    for (_, end), (start, _) in zip(picks, picks[1:]):
        assert start >= end, "a highlight reel must not repeat footage"


def test_picks_are_spread_across_the_video():
    """Four windows butted together render as one chunk, not as an edit."""
    energies = [0.5] * 240  # 120s, perfectly flat: nothing but spacing to go on

    picks = tl.top_moments(flat_timeline(energies), count=4)

    starts = [s for s, _ in picks]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 4.0 for g in gaps), f"picks bunched together: {starts}"


def test_every_window_is_at_least_the_requested_length():
    picks = tl.top_moments(flat_timeline([0.5] * 200), count=3, min_seconds=6.0)

    assert all(end - start >= 6.0 - 1e-9 for start, end in picks)


def test_no_window_runs_past_the_end_of_the_video():
    energies = [0.1] * 40
    energies[-5:] = [1.0] * 5  # the peak sits right on the end

    picks = tl.top_moments(flat_timeline(energies), count=2)

    duration = 40 * 0.5
    assert all(end <= duration for _, end in picks)


def test_an_empty_timeline_yields_no_moments():
    """The caller substitutes a default clip; it must not crash getting there."""
    empty = Timeline(media=media(10.0), transcript=Transcript(), buckets=[])

    assert tl.top_moments(empty, count=4) == []


def test_a_video_too_short_for_the_full_count_returns_what_it_can():
    picks = tl.top_moments(flat_timeline([0.5] * 12), count=4, min_seconds=4.0)

    assert 0 < len(picks) <= 4
