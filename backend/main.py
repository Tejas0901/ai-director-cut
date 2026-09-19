"""FastAPI surface.

Routing and serialisation only. Every route either records a job, reads a
job, or streams a job's events - the pipeline itself lives in runner.py and
knows nothing about HTTP.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import db, jobs, runner
from .pipeline import render
from .config import (
    LLM_PROVIDER,
    OUTPUT_DIR,
    STT_PROVIDER,
    STUB_STAGES,
    TTS_PROVIDER,
    UPLOAD_DIR,
)
from .schemas import STAGE_LABELS, STAGE_ORDER, JobSummary

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

app = FastAPI(title="AI Director's Cut", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/media/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.on_event("startup")
def _startup() -> None:
    db.init()


@app.get("/api/health")
def health() -> dict:
    """Shows which providers are live - the first thing to check on demo day."""
    return {
        "status": "ok",
        "llm": LLM_PROVIDER,
        "stt": STT_PROVIDER,
        "tts": TTS_PROVIDER,
        "stubbed_stages": sorted(STUB_STAGES),
        "stages": [{"key": s, "label": STAGE_LABELS[s]} for s in STAGE_ORDER],
    }


@app.post("/api/jobs")
async def create_job(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix or '?'}'")

    job = jobs.create(
        filename=file.filename or "upload.mp4",
        original_url="",  # filled in below, once we know the stored name
    )

    stored = UPLOAD_DIR / f"{job.id}{suffix}"
    with open(stored, "wb") as out:
        shutil.copyfileobj(file.file, out)

    job.original_url = f"/media/uploads/{stored.name}"

    asyncio.create_task(runner.run_job(job.id, str(stored)))
    return {"job_id": job.id, "original_url": job.original_url}


@app.get("/api/jobs")
def list_jobs(limit: int = 24) -> list[dict]:
    """The library: every reel this machine has cut, newest first.

    Reads through SQLite rather than the in-memory registry so the list
    survives a restart - which is the entire point of having a library.
    """
    summaries = [JobSummary.of(job) for job in db.load_all()]
    summaries.sort(key=_sort_key, reverse=True)
    return [s.model_dump(mode="json") for s in summaries[:max(1, limit)]]


def _sort_key(summary: JobSummary) -> float:
    """Newest first, with a fallback for rows written before created_at.

    Job ids are random hex, so they carry no ordering of their own. Older
    rows have no timestamp at all; their rendered file does, so use that.
    """
    if summary.created_at is not None:
        return summary.created_at
    final = OUTPUT_DIR / summary.id / "final.mp4"
    try:
        return final.stat().st_mtime
    except OSError:
        return 0.0


def _checked_id(job_id: str) -> str:
    """Reject anything that is not a real job id before it becomes a path.

    Job ids are hex from uuid4, and these routes turn one into a directory we
    delete recursively. Existence checks alone would probably cover it, but
    "probably" is the wrong standard for a recursive delete, so the shape is
    enforced explicitly.
    """
    if not re.fullmatch(r"[0-9a-f]{6,32}", job_id):
        raise HTTPException(400, "malformed job id")
    return job_id


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Remove a reel and everything it produced. Not reversible."""
    _checked_id(job_id)
    if jobs.get(job_id) is None:
        raise HTTPException(404, "no such job")

    removed: list[str] = []

    work_dir = OUTPUT_DIR / job_id
    if work_dir.is_dir():
        shutil.rmtree(work_dir, ignore_errors=True)
        removed.append(work_dir.name)

    # The upload keeps its original suffix, so match on the stem.
    for upload in UPLOAD_DIR.glob(f"{job_id}.*"):
        upload.unlink(missing_ok=True)
        removed.append(upload.name)

    jobs.remove(job_id)
    return {"deleted": job_id, "files": removed}


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    """The finished reel, named after the cut rather than 'final.mp4'."""
    _checked_id(job_id)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")

    reel = OUTPUT_DIR / job_id / "final.mp4"
    if not reel.exists():
        raise HTTPException(404, "this job has no rendered reel")

    title = job.plan.title if job.plan else ""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or f"reel-{job_id}"
    return FileResponse(reel, media_type="video/mp4", filename=f"{slug}.mp4")


@app.get("/api/jobs/{job_id}/poster")
def job_poster(job_id: str) -> FileResponse:
    """Thumbnail for one reel. Generated on first request, cached after."""
    if jobs.get(job_id) is None:
        raise HTTPException(404, "no such job")

    poster = render.poster_for(job_id)
    if poster is None:
        raise HTTPException(404, "no poster available")
    return FileResponse(poster, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}")
def read_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.model_dump(mode="json")


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-sent events for the live progress panel."""
    if jobs.get(job_id) is None:
        raise HTTPException(404, "no such job")

    async def stream():
        async for event in jobs.subscribe(job_id):
            yield f"data: {event.model_dump_json()}\n\n"
        # Final frame carries the completed job so the client does not need
        # a follow-up request to render the result.
        job = jobs.get(job_id)
        if job is not None:
            yield f"event: complete\ndata: {json.dumps(job.model_dump(mode='json'))}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
