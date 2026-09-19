"""The two pure string-builders in the renderer, which are also the two that
fail silently.

A malformed duck expression does not error - ffmpeg evaluates it to something
and the narration sits under the original audio. A mis-escaped caption does
not error either; drawtext just swallows the rest of the filter chain. So both
are checked by meaning here: the gain curve is evaluated as numbers, and the
caption is checked for the characters that actually break the parser.
"""

from __future__ import annotations

import pytest

from backend.config import DUCK_VOLUME
from backend.pipeline.render import (
    APOSTROPHE,
    DUCK_FADE_SECONDS,
    _duck_expression,
    _escape_drawtext,
)


def gain_at(expression: str, t: float) -> float:
    """Evaluate one of ffmpeg's `if(lt(...))` expressions at time t.

    ffmpeg's expression grammar is close enough to Python that renaming the
    two functions makes it evaluable. Both branches are computed eagerly,
    which is safe here: every divisor is a literal baked into the string.
    """
    python = expression.replace("if(", "IF(").replace("lt(", "LT(")
    return float(eval(python, {  # noqa: S307 - our own generated expression
        "IF": lambda c, a, b: a if c else b,
        "LT": lambda a, b: a < b,
        "t": t,
    }))


# --------------------------------------------------------------------------
# _duck_expression
# --------------------------------------------------------------------------


def test_the_original_audio_is_ducked_under_the_intro():
    expr = _duck_expression(intro_len=5.0, outro_start=25.0, total=30.0)

    assert gain_at(expr, 0.0) == pytest.approx(DUCK_VOLUME)
    assert gain_at(expr, 4.9) == pytest.approx(DUCK_VOLUME)


def test_the_original_audio_is_ducked_under_the_outro():
    expr = _duck_expression(intro_len=5.0, outro_start=25.0, total=30.0)

    assert gain_at(expr, 25.0) == pytest.approx(DUCK_VOLUME)
    assert gain_at(expr, 29.9) == pytest.approx(DUCK_VOLUME)


def test_the_original_audio_is_at_full_volume_in_between():
    expr = _duck_expression(intro_len=5.0, outro_start=25.0, total=30.0)

    assert gain_at(expr, 15.0) == pytest.approx(1.0)


def test_the_gain_ramps_rather_than_stepping_at_the_intro_edge():
    """An instant gain change is the pop this whole expression exists to avoid."""
    expr = _duck_expression(intro_len=5.0, outro_start=25.0, total=30.0)

    midpoint = 5.0 + DUCK_FADE_SECONDS / 2
    halfway = DUCK_VOLUME + (1.0 - DUCK_VOLUME) / 2
    assert gain_at(expr, midpoint) == pytest.approx(halfway, abs=0.01)


def test_the_gain_ramps_back_down_before_the_outro():
    expr = _duck_expression(intro_len=5.0, outro_start=25.0, total=30.0)

    midpoint = 25.0 - DUCK_FADE_SECONDS / 2
    halfway = DUCK_VOLUME + (1.0 - DUCK_VOLUME) / 2
    assert gain_at(expr, midpoint) == pytest.approx(halfway, abs=0.01)


def test_the_curve_is_continuous_across_every_boundary():
    expr = _duck_expression(intro_len=5.0, outro_start=25.0, total=30.0)

    for edge in (5.0, 5.0 + DUCK_FADE_SECONDS, 25.0 - DUCK_FADE_SECONDS, 25.0):
        before = gain_at(expr, edge - 0.001)
        after = gain_at(expr, edge + 0.001)
        assert abs(after - before) < 0.02, f"step of {after - before:.3f} at t={edge}"


def test_the_gain_never_leaves_the_intended_range():
    expr = _duck_expression(intro_len=5.0, outro_start=25.0, total=30.0)

    for i in range(0, 300):
        assert DUCK_VOLUME - 1e-9 <= gain_at(expr, i / 10) <= 1.0 + 1e-9


def test_a_gap_too_small_to_ramp_falls_back_to_a_hard_gate():
    """Two 0.4s ramps do not fit in a 0.01s gap; better a step than a divide by zero."""
    expr = _duck_expression(intro_len=10.0, outro_start=10.005, total=20.0)

    assert gain_at(expr, 5.0) == pytest.approx(DUCK_VOLUME)
    assert gain_at(expr, 15.0) == pytest.approx(DUCK_VOLUME)


def test_no_gap_at_all_still_produces_a_usable_expression():
    expr = _duck_expression(intro_len=10.0, outro_start=10.0, total=20.0)

    assert gain_at(expr, 0.0) == pytest.approx(DUCK_VOLUME)
    assert gain_at(expr, 19.0) == pytest.approx(DUCK_VOLUME)


def test_an_outro_before_the_intro_ends_does_not_divide_by_zero():
    """`gap` clamps at zero, so an inverted pair degrades instead of exploding."""
    expr = _duck_expression(intro_len=12.0, outro_start=8.0, total=20.0)

    assert gain_at(expr, 10.0) == pytest.approx(DUCK_VOLUME)


def test_a_short_reel_shrinks_its_ramps_to_fit():
    """The ramps must never overlap, however little room there is."""
    expr = _duck_expression(intro_len=1.0, outro_start=1.3, total=3.0)

    for i in range(0, 30):
        assert DUCK_VOLUME - 1e-9 <= gain_at(expr, i / 10) <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# _escape_drawtext
# --------------------------------------------------------------------------

# Anything in here ends the quoted drawtext argument or escapes what follows,
# and no caption is worth a broken filter graph.
FORBIDDEN = ("'", ":", "\\", "%")


@pytest.mark.parametrize("raw", [
    "Didn't see that",
    "Time: 12:30",
    "100% certain",
    "back\\slash",
    "semi;colon",
    "bracket[0]",
    "a'b:c\\d%e",
    "=equals=",
])
def test_nothing_that_breaks_the_filter_survives(raw):
    cleaned = _escape_drawtext(raw)

    for char in FORBIDDEN:
        assert char not in cleaned, f"{char!r} survived in {cleaned!r}"


def test_an_apostrophe_is_translated_rather_than_dropped():
    """Captions come from an LLM and are full of contractions; losing the
    apostrophe turns every one of them into a visible typo."""
    assert _escape_drawtext("Didn't") == f"Didn{APOSTROPHE}t"
    assert _escape_drawtext("Here`s") == f"Here{APOSTROPHE}s"


def test_ordinary_punctuation_is_kept():
    assert _escape_drawtext("Wait - what?!") == "Wait - what?!"
    assert _escape_drawtext("One, two.") == "One, two."


def test_accented_and_non_latin_text_survives():
    assert _escape_drawtext("café") == "café"
    assert _escape_drawtext("Grüße") == "Grüße"
    assert _escape_drawtext("日本語") == "日本語"


@pytest.mark.parametrize("word", [
    "शतरंज",      # Hindi, anusvara (Mn)
    "हिन्दी",      # Devanagari with matras and virama
    "বাংলা",      # Bengali
    "ไทย",       # Thai, above-vowel marks
    "العربية",    # Arabic
    "é",   # decomposed é
])
def test_combining_marks_are_not_stripped_out_of_a_word(word):
    """A mark is category M, not alnum. Dropping one does not remove a
    character, it corrupts the word: शतरंज would become शतरज."""
    assert _escape_drawtext(word) == word


def test_emoji_are_stripped():
    assert _escape_drawtext("🔥 fire 🔥") == "fire"


def test_a_caption_is_capped_at_the_overlay_width():
    assert len(_escape_drawtext("x" * 200)) == 48


def test_surrounding_whitespace_goes():
    assert _escape_drawtext("   padded   ") == "padded"


def test_empty_and_missing_text_are_handled():
    assert _escape_drawtext("") == ""
    assert _escape_drawtext(None) == ""


def test_text_that_is_entirely_forbidden_becomes_empty():
    """Empty is the renderer's signal to skip drawtext for this clip."""
    assert _escape_drawtext(":::%%%\\\\\\") == ""
