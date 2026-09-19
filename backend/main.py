"""FastAPI surface.

Routing and serialisation only. Every route either records a job, reads a
job, or streams a job's events - the pipeline itself lives in runner.py and
knows nothing about HTTP.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import db, jobs, runner
from .config import (
    LLM_PROVIDER,
    OUTPUT_DIR,
    STT_PROVIDER,
    STUB_STAGES,
    TTS_PROVIDER,
    UPLOAD_DIR,
)
from .schemas import STAGE_LABELS, STAGE_ORDER

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
