"""Pipeline orchestration.

The only place that knows the order of the stages. Each stage is called,
cached, and announced to the job's subscribers. Stages listed in STUB_STAGES
are skipped with a canned result, which is how we keep the whole pipeline
green while individual stages are still half-written.
"""

from __future__ import annotations

import asyncio
import traceback
from pathlib import Path

from . import cache, jobs
from .config import OUTPUT_DIR, UPLOAD_DIR, is_stubbed
from .pipeline import director, ingest, render, timeline, transcribe, vision
from .schemas import EditPlan, MediaInfo, Timeline, Transcript, VisualBucket


async def run_job(job_id: str, video_path: str) -> None:
    """Drive one upload through every stage. Never raises."""
    try:
        await asyncio.to_thread(_run_sync, job_id, video_path)
    except Exception as exc:  # noqa: BLE001 - terminal handler, must catch all
        traceback.print_exc()
        jobs.fail(job_id, f"{type(exc).__name__}: {exc}")


def _run_sync(job_id: str, video_path: str) -> None:
    job = jobs.get(job_id)
    if job is None:
        return

    key = cache.file_key(video_path)

    # --- 1. ingest --------------------------------------------------------
    jobs.set_stage(job_id, "ingesting")
    media = ingest.ingest(video_path, key)
    job.duration = media.duration

    # --- 2. transcribe ----------------------------------------------------
    jobs.set_stage(job_id, "transcribing")
    script = _transcribe(media, key)
    job.transcript = script

    # --- 3. vision --------------------------------------------------------
    jobs.set_stage(job_id, "analyzing_video",
                   detail=f"{len(script.segments)} speech segments found")
    visual = _vision(media, key)

    # --- 4. timeline ------------------------------------------------------
    jobs.set_stage(job_id, "building_timeline")
    tl = timeline.build(media, script, visual)

    # --- 5. director ------------------------------------------------------
    jobs.set_stage(job_id, "directing")
    plan = _direct(tl, key)
    job.plan = plan

    # --- 6 + 7. narrate and render ---------------------------------------
    jobs.set_stage(job_id, "narrating",
                   detail=f'"{plan.title}" - {len(plan.clips)} clips selected')
    jobs.set_stage(job_id, "rendering")
    output = _render(plan, media, job_id)

    job.output_url = _media_url(output)
    jobs.set_stage(job_id, "done")


def _media_url(path: Path) -> str:
    """Map a rendered file onto its served URL.

    In skeleton mode the renderer hands back the original upload, which lives
    under a different mount - so resolve against both roots rather than
    assuming the output directory.
    """
    path = Path(path).resolve()
    for root, mount in ((OUTPUT_DIR, "/media/output"), (UPLOAD_DIR, "/media/uploads")):
        try:
            return f"{mount}/{path.relative_to(root.resolve()).as_posix()}"
        except ValueError:
            continue
    return f"/media/output/{path.name}"


# --------------------------------------------------------------------------
# Stage wrappers: cache -> stub -> real
# --------------------------------------------------------------------------


def _transcribe(media: MediaInfo, key: str) -> Transcript:
    if is_stubbed("transcribing") or not media.audio_path:
        return transcribe.transcribe("", provider="mock") if is_stubbed("transcribing") \
            else Transcript()

    cached = cache.load(key, "transcript", Transcript)
    if cached is not None:
        return cached

    result = transcribe.transcribe(media.audio_path)
    cache.store(key, "transcript", result)
    return result


def _vision(media: MediaInfo, key: str) -> list[VisualBucket]:
    if is_stubbed("analyzing_video"):
        return []

    cached = cache.load_list(key, "visual", VisualBucket)
    if cached is not None:
        return cached

    result = vision.analyze(media.path, media.duration)
    cache.store_list(key, "visual", list(result))
    return result


def _direct(tl: Timeline, key: str) -> EditPlan:
    provider = "mock" if is_stubbed("directing") else None
    # The plan is intentionally NOT cached: re-running should be able to give
    # you a different creative take on the same footage.
    return director.direct(tl, provider=provider)


def _render(plan: EditPlan, media: MediaInfo, job_id: str) -> Path:
    if is_stubbed("rendering"):
        # Skeleton mode: hand back the original file so the UI has something
        # to play and the end-to-end path stays exercised.
        return Path(media.path)
    return render.render(plan, media, job_id)
