"""`sanitize_plan` is the airlock between the LLM and FFmpeg.

Models hallucinate timestamps past the end of the video, invert start and end,
emit zero-length clips and overlap their own picks. None of that may reach a
filter graph, so every plan passes through here first.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.schemas import (
    MAX_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    Clip,
    EditPlan,
    MusicMood,
    sanitize_plan,
)


def plan_with(*spans, **kwargs) -> EditPlan:
    """An EditPlan carrying the given (start, end) pairs."""
    return EditPlan(
        title=kwargs.pop("title", "Test"),
        intro_narration="intro",
        outro_summary="outro",
        clips=[Clip(start_time=a, end_time=b,
                    overlay_title=kwargs.pop("overlay_title", "t"),
                    reason="r")
               for a, b in spans],
        **kwargs,
    )


def spans(plan: EditPlan) -> list[tuple[float, float]]:
    return [(c.start_time, c.end_time) for c in plan.clips]


# --------------------------------------------------------------------------
# inverted and out-of-range timestamps
# --------------------------------------------------------------------------


def test_inverted_times_are_swapped_not_dropped():
    result = sanitize_plan(plan_with((9.0, 3.0)), duration=60.0)

    assert spans(result) == [(3.0, 9.0)]


def test_times_past_the_end_are_clamped_to_the_duration():
    result = sanitize_plan(plan_with((50.0, 900.0)), duration=60.0)

    assert spans(result) == [(50.0, 60.0)]


def test_negative_times_are_clamped_to_zero():
    result = sanitize_plan(plan_with((-20.0, 5.0)), duration=60.0)

    assert spans(result) == [(0.0, 5.0)]


def test_a_clip_entirely_past_the_end_collapses_and_is_dropped():
    """Clamping both ends to the duration leaves nothing of length."""
    with pytest.raises(ValueError, match="no usable clips"):
        sanitize_plan(plan_with((900.0, 950.0)), duration=60.0)


# --------------------------------------------------------------------------
# length bounds
# --------------------------------------------------------------------------


def test_a_clip_shorter_than_the_minimum_is_dropped():
    keeper = (10.0, 20.0)
    runt = (30.0, 30.0 + MIN_CLIP_SECONDS - 0.1)

    result = sanitize_plan(plan_with(keeper, runt), duration=60.0)

    assert spans(result) == [keeper]


def test_a_clip_exactly_at_the_minimum_is_kept():
    span = (10.0, 10.0 + MIN_CLIP_SECONDS)

    result = sanitize_plan(plan_with(span), duration=60.0)

    assert spans(result) == [span]


def test_a_zero_length_clip_is_dropped():
    with pytest.raises(ValueError, match="no usable clips"):
        sanitize_plan(plan_with((12.0, 12.0)), duration=60.0)


def test_an_over_long_clip_is_truncated_rather_than_dropped():
    result = sanitize_plan(plan_with((0.0, 90.0)), duration=200.0)

    assert spans(result) == [(0.0, MAX_CLIP_SECONDS)]


def test_a_truncated_clip_still_fits_inside_the_video():
    result = sanitize_plan(plan_with((10.0, 190.0)), duration=200.0)

    start, end = spans(result)[0]
    assert end <= 200.0
    assert end - start == MAX_CLIP_SECONDS


# --------------------------------------------------------------------------
# ordering and overlap
# --------------------------------------------------------------------------


def test_clips_come_back_in_time_order():
    result = sanitize_plan(plan_with((40.0, 48.0), (5.0, 12.0), (20.0, 27.0)),
                           duration=60.0)

    assert spans(result) == [(5.0, 12.0), (20.0, 27.0), (40.0, 48.0)]


def test_a_clip_overlapping_an_earlier_keeper_is_dropped():
    """Duplicated footage in a highlight reel reads as a broken render."""
    result = sanitize_plan(plan_with((10.0, 20.0), (15.0, 25.0)), duration=60.0)

    assert spans(result) == [(10.0, 20.0)]


def test_clips_that_merely_touch_are_both_kept():
    """Abutting is not overlapping - the second starts where the first ends."""
    result = sanitize_plan(plan_with((10.0, 20.0), (20.0, 30.0)), duration=60.0)

    assert spans(result) == [(10.0, 20.0), (20.0, 30.0)]


def test_overlap_is_judged_after_sorting_not_in_arrival_order():
    result = sanitize_plan(plan_with((15.0, 25.0), (10.0, 20.0)), duration=60.0)

    assert spans(result) == [(10.0, 20.0)]


def test_no_more_than_five_clips_survive():
    result = sanitize_plan(
        plan_with(*[(i * 10.0, i * 10.0 + 5.0) for i in range(9)]), duration=200.0)

    assert len(result.clips) == 5


# --------------------------------------------------------------------------
# emptiness
# --------------------------------------------------------------------------


def test_an_edit_plan_cannot_be_built_with_no_clips_at_all():
    """The model itself rejects this, before sanitisation is ever reached."""
    with pytest.raises(ValidationError):
        EditPlan(title="t", intro_narration="i", outro_summary="o", clips=[])


def test_everything_unusable_raises_rather_than_returning_an_empty_plan():
    """A plan with no clips would reach ffmpeg as concat=n=0 and fail there."""
    with pytest.raises(ValueError, match="no usable clips"):
        sanitize_plan(plan_with((5.0, 5.2), (9.0, 9.1)), duration=60.0)


# --------------------------------------------------------------------------
# field hygiene and provenance
# --------------------------------------------------------------------------


def test_overlay_titles_are_trimmed_and_capped():
    plan = plan_with((10.0, 20.0), overlay_title="   " + "x" * 90 + "   ")

    result = sanitize_plan(plan, duration=60.0)

    assert len(result.clips[0].overlay_title) == 60
    assert not result.clips[0].overlay_title.startswith(" ")


def test_timestamps_are_rounded_to_milliseconds():
    result = sanitize_plan(plan_with((1.00049999, 20.0)), duration=60.0)

    assert result.clips[0].start_time == 1.0


def test_everything_but_the_clips_is_carried_through():
    plan = plan_with((10.0, 20.0), title="Keep Me")
    plan = plan.model_copy(update={"music_mood": MusicMood.CHILL,
                                   "source": "heuristic",
                                   "fallback_reason": "gemini: boom"})

    result = sanitize_plan(plan, duration=60.0)

    assert result.title == "Keep Me"
    assert result.music_mood is MusicMood.CHILL
    assert result.source == "heuristic"
    assert result.fallback_reason == "gemini: boom"
    assert result.intro_narration == "intro"


def test_the_original_plan_is_not_mutated():
    plan = plan_with((9.0, 3.0))

    sanitize_plan(plan, duration=60.0)

    assert (plan.clips[0].start_time, plan.clips[0].end_time) == (9.0, 3.0)
