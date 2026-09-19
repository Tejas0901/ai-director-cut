"""End to end, with every provider on mock: upload in, playable reel out.

This is the test that would have caught most of what actually broke in this
project, because the failures were never in one stage - they were in the
handover between two. It runs the real runner, the real OpenCV pass and real
ffmpeg, and only stubs the three things that need network or a key.

The promise being checked is the one in the README: the app always produces a
reel, with no keys and no internet.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from backend import cache, db, jobs, runner
from backend.config import FFMPEG, FFPROBE, FIXTURES_DIR
from backend.pipeline import director, ffmpeg_util, render, transcribe, tts

pytestmark = pytest.mark.integration


def _missing_ffmpeg() -> bool:
    return shutil.which(FFMPEG) is None or shutil.which(FFPROBE) is None


needs_ffmpeg = pytest.mark.skipif(_missing_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")


def fixture_or_skip(name: str) -> Path:
    """Fixtures are generated, not committed - `uv run python -m fixtures.make_fixtures`."""
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run `python -m fixtures.make_fixtures`")
    return path


@pytest.fixture
def offline_pipeline(tmp_path, monkeypatch):
    """Every provider on mock, every output path inside tmp_path.

    The provider constants are read from each stage's own module namespace
    (`from ..config import X` binds the value at import), so they are patched
    there rather than on config.
    """
    monkeypatch.setattr(transcribe, "STT_PROVIDER", "mock")
    monkeypatch.setattr(director, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(tts, "TTS_PROVIDER", "mock")

    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(render, "OUTPUT_DIR", output)
    monkeypatch.setattr(runner, "OUTPUT_DIR", output)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")

    jobs._jobs.clear()
    jobs._subscribers.clear()
    yield output
    jobs._jobs.clear()
    jobs._subscribers.clear()


def run_pipeline(source: Path, upload_dir: Path) -> tuple[str, object]:
    """Drive one file through the real runner. Returns (job_id, job)."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload = upload_dir / source.name
    shutil.copyfile(source, upload)

    job = jobs.create(source.name, f"/media/uploads/{upload.name}")
    asyncio.run(runner.run_job(job.id, str(upload)))
    return job.id, jobs.get(job.id)


def streams_of(path: Path) -> dict:
    info = ffmpeg_util.probe(path)
    return {s.get("codec_type"): s for s in info["streams"]} | {"_format": info["format"]}


# --------------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("fixture_name", ["synthetic.mp4", "talkie.mp4"])
def test_the_whole_pipeline_produces_a_playable_reel(fixture_name, tmp_path, offline_pipeline):
    source = fixture_or_skip(fixture_name)

    job_id, job = run_pipeline(source, tmp_path)

    assert job.stage == "done", f"pipeline failed: {job.error}"
    assert job.error is None

    reel = offline_pipeline / job_id / "final.mp4"
    assert reel.exists(), "no final.mp4 was written"
    assert reel.stat().st_size > 0

    streams = streams_of(reel)
    assert "video" in streams, "the reel has no video stream"
    assert "audio" in streams, "the reel has no audio stream"
    assert float(streams["_format"]["duration"]) > 0


@needs_ffmpeg
def test_a_source_with_no_audio_track_still_gets_one(tmp_path, offline_pipeline):
    """synthetic.mp4 has no audio stream at all, not a silent one. concat
    needs a stream on every segment, so the renderer has to synthesise it -
    this is the only test that exercises the anullsrc branch."""
    source = fixture_or_skip("synthetic.mp4")
    assert "audio" not in streams_of(source), "fixture no longer tests this path"

    job_id, job = run_pipeline(source, tmp_path)

    assert job.stage == "done", f"pipeline failed: {job.error}"
    assert "audio" in streams_of(offline_pipeline / job_id / "final.mp4")


@needs_ffmpeg
def test_the_reel_is_shorter_than_the_source(tmp_path, offline_pipeline):
    """It is a highlight reel; if it is not shorter, nothing was cut."""
    source = fixture_or_skip("synthetic.mp4")

    job_id, job = run_pipeline(source, tmp_path)

    reel = float(streams_of(offline_pipeline / job_id / "final.mp4")["_format"]["duration"])
    assert 0 < reel < job.duration


@needs_ffmpeg
def test_the_job_ends_up_describing_what_it_produced(tmp_path, offline_pipeline):
    source = fixture_or_skip("synthetic.mp4")

    _, job = run_pipeline(source, tmp_path)

    assert job.plan is not None and job.plan.clips
    assert job.plan.source == "heuristic", "no key was configured"
    assert job.output_url and job.output_url.startswith("/media/")
    assert job.progress == 1.0


@needs_ffmpeg
def test_the_rendered_clips_match_the_plan(tmp_path, offline_pipeline):
    """The reel's length is the sum of the plan's clips, which is the one
    assertion that ties the Director's output to the renderer's input."""
    source = fixture_or_skip("synthetic.mp4")

    job_id, job = run_pipeline(source, tmp_path)

    expected = sum(c.end_time - c.start_time for c in job.plan.clips)
    actual = float(streams_of(offline_pipeline / job_id / "final.mp4")["_format"]["duration"])
    assert actual == pytest.approx(expected, abs=1.5)


@needs_ffmpeg
def test_a_second_run_reuses_the_cache(tmp_path, offline_pipeline):
    """The vision pass is the slow one; it must not run twice for one file."""
    source = fixture_or_skip("synthetic.mp4")

    run_pipeline(source, tmp_path)
    cached = list((tmp_path / "cache").rglob("visual.json"))
    assert cached, "the vision pass was not cached"

    calls: list[int] = []
    real_analyze = runner.vision.analyze

    def counting_analyze(*args, **kwargs):
        calls.append(1)
        return real_analyze(*args, **kwargs)

    runner.vision.analyze = counting_analyze
    try:
        _, job = run_pipeline(source, tmp_path / "second")
    finally:
        runner.vision.analyze = real_analyze

    assert job.stage == "done"
    assert not calls, "vision re-ran despite a warm cache"


@needs_ffmpeg
def test_a_file_that_is_not_a_video_fails_the_job_instead_of_the_server(
        tmp_path, offline_pipeline):
    """`run_job` is the pipeline's terminal error handler and must not raise."""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not an mp4")

    job = jobs.create("broken.mp4", "/media/uploads/broken.mp4")
    asyncio.run(runner.run_job(job.id, str(broken)))

    finished = jobs.get(job.id)
    assert finished.stage == "failed"
    assert finished.error
