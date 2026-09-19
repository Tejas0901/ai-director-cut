"""FONT_FILE is optional now; these pin down what "optional" actually resolves to.

The overlay title is the only part of the render that needs a font, so font
resolution has to end in either a real file or a clean None - never an
exception, never a path that does not exist. And when drawtext fails anyway
the reel still has to come out, which is the second half of this file.

Fonts here are real files under tmp_path rather than a patched `Path.is_file`,
so the checks being exercised are the ones that actually ship.
"""

from __future__ import annotations

import pytest

from backend import config
from backend.pipeline import render
from backend.pipeline.ffmpeg_util import FFmpegError
from backend.schemas import Clip, EditPlan, MediaInfo


@pytest.fixture(autouse=True)
def clear_font_cache():
    """`resolve_font` is lru_cached, so every test must start from cold."""
    config.resolve_font.cache_clear()
    yield
    config.resolve_font.cache_clear()


def make_font(tmp_path, name: str):
    path = tmp_path / name
    path.write_bytes(b"not a real typeface, but a real file")
    return path


# --------------------------------------------------------------------------
# config.resolve_font
# --------------------------------------------------------------------------


def test_font_file_from_env_wins_over_the_search_path(tmp_path, monkeypatch):
    chosen = make_font(tmp_path, "Chosen-Bold.ttf")
    monkeypatch.setattr(config, "FONT_FILE", str(chosen))
    monkeypatch.setattr(config, "FONT_CANDIDATES", (str(make_font(tmp_path, "DejaVuSans-Bold.ttf")),))

    assert config.resolve_font() == chosen


def test_a_missing_font_file_falls_back_to_the_search_path(tmp_path, monkeypatch):
    """A typo in .env should cost you the face you asked for, not the render."""
    fallback = make_font(tmp_path, "LiberationSans-Bold.ttf")
    monkeypatch.setattr(config, "FONT_FILE", str(tmp_path / "typo.ttf"))
    monkeypatch.setattr(config, "FONT_CANDIDATES", (str(fallback),))

    assert config.resolve_font() == fallback


def test_the_first_existing_candidate_wins(tmp_path, monkeypatch):
    present = make_font(tmp_path, "arialbd.ttf")
    later = make_font(tmp_path, "DejaVuSans-Bold.ttf")
    monkeypatch.setattr(config, "FONT_FILE", "")
    monkeypatch.setattr(
        config, "FONT_CANDIDATES",
        (str(tmp_path / "absent.ttf"), str(present), str(later)),
    )

    assert config.resolve_font() == present


def test_no_font_anywhere_resolves_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FONT_FILE", "")
    monkeypatch.setattr(config, "FONT_CANDIDATES",
                        (str(tmp_path / "a.ttf"), str(tmp_path / "b.ttf")))

    assert config.resolve_font() is None
    assert config.ffmpeg_fontfile() is None


def test_a_directory_is_not_a_font(tmp_path, monkeypatch):
    """`is_file`, not `exists`: /usr/share/fonts/<family> is often a directory."""
    family = tmp_path / "DejaVuSans-Bold.ttf"
    family.mkdir()
    monkeypatch.setattr(config, "FONT_FILE", "")
    monkeypatch.setattr(config, "FONT_CANDIDATES", (str(family),))

    assert config.resolve_font() is None


def test_ffmpeg_fontfile_escapes_the_path_for_the_filter_graph(tmp_path, monkeypatch):
    chosen = make_font(tmp_path, "arialbd.ttf")
    monkeypatch.setattr(config, "FONT_FILE", str(chosen))
    monkeypatch.setattr(config, "FONT_CANDIDATES", ())

    escaped = config.ffmpeg_fontfile()

    # The only backslash and the only colon allowed to survive are the two
    # halves of an escaped drive colon. A raw colon ends the drawtext option;
    # a raw backslash escapes whatever follows it.
    bare = escaped.replace(r"\:", "")
    assert "\\" not in bare, "a stray backslash escapes the next filter character"
    assert ":" not in bare, "an unescaped colon splits the option"
    assert escaped.endswith("/arialbd.ttf")


# --------------------------------------------------------------------------
# render.cut_reel
# --------------------------------------------------------------------------


def one_clip_plan(overlay_title: str = "Peak Chaos") -> EditPlan:
    return EditPlan(
        title="Test Reel",
        intro_narration="intro",
        outro_summary="outro",
        clips=[Clip(start_time=0.0, end_time=5.0, overlay_title=overlay_title)],
    )


def fake_media() -> MediaInfo:
    return MediaInfo(path="in.mp4", duration=10.0, width=1280, height=720,
                     fps=30.0, has_audio=True)


def filter_graph(args: list[str]) -> str:
    """The -filter_complex value, which is the only place drawtext can appear.

    Matching against the whole argv looks equivalent and is not: pytest builds
    `tmp_path` from the test's own name, so a test named ...drawtext... puts
    that word in the output path and every command appears to contain a
    drawtext filter.
    """
    return args[args.index("-filter_complex") + 1]


@pytest.fixture
def ffmpeg_calls(monkeypatch):
    """Capture the filter graph per ffmpeg invocation instead of encoding."""
    calls: list[str] = []

    def fake_run(args, **kwargs):
        calls.append(filter_graph(args))
        return ""

    monkeypatch.setattr(render, "run", fake_run)
    return calls


def test_overlays_are_skipped_entirely_when_no_font_exists(tmp_path, monkeypatch, ffmpeg_calls):
    monkeypatch.setattr(render, "ffmpeg_fontfile", lambda: None)

    render.cut_reel(one_clip_plan(), fake_media(), tmp_path / "reel.mp4")

    assert len(ffmpeg_calls) == 1
    assert "drawtext" not in ffmpeg_calls[0]


def test_a_failed_drawtext_is_re_cut_without_overlays(tmp_path, monkeypatch):
    """The whole point: a broken caption must not cost you the reel."""
    monkeypatch.setattr(render, "ffmpeg_fontfile", lambda: "C\\:/Windows/Fonts/arialbd.ttf")
    calls: list[str] = []

    def fake_run(args, **kwargs):
        graph = filter_graph(args)
        calls.append(graph)
        if "drawtext" in graph:
            raise FFmpegError("No such filter: 'drawtext'")
        return ""

    monkeypatch.setattr(render, "run", fake_run)

    out = render.cut_reel(one_clip_plan(), fake_media(), tmp_path / "reel.mp4")

    assert len(calls) == 2, "exactly one retry, not a loop"
    assert "drawtext" in calls[0]
    assert "drawtext" not in calls[1]
    assert out == tmp_path / "reel.mp4"


def test_a_failure_with_no_overlay_to_drop_is_not_retried(tmp_path, monkeypatch):
    """Retrying an identical command just doubles the wait before the error."""
    monkeypatch.setattr(render, "ffmpeg_fontfile", lambda: "C\\:/Windows/Fonts/arialbd.ttf")
    calls: list[str] = []

    def fake_run(args, **kwargs):
        calls.append(filter_graph(args))
        raise FFmpegError("Invalid data found when processing input")

    monkeypatch.setattr(render, "run", fake_run)

    # No overlay text on this plan, so the two graphs would be identical.
    with pytest.raises(FFmpegError):
        render.cut_reel(one_clip_plan(overlay_title=""), fake_media(), tmp_path / "reel.mp4")

    assert len(calls) == 1
