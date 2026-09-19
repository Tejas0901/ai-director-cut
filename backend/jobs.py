"""Job registry, state machine and live event bus.

Holds every job in memory (single uvicorn worker) and mirrors it to SQLite so
a restart mid-demo does not lose finished renders. Subscribers receive stage
transitions over an asyncio queue, which `main.py` turns into an SSE stream.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator

from . import db
from .schemas import STAGE_LABELS, STAGE_ORDER, Job, JobEvent

_jobs: dict[str, Job] = {}
_subscribers: dict[str, list[asyncio.Queue[JobEvent | None]]] = {}


def create(filename: str, original_url: str) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, filename=filename, original_url=original_url,
              created_at=time.time())
    _jobs[job_id] = job
    _subscribers[job_id] = []
    db.save(job)
    return job


def get(job_id: str) -> Job | None:
    job = _jobs.get(job_id)
    if job is None:
        job = db.load(job_id)
        if job is not None:
            _jobs[job_id] = job
            _subscribers.setdefault(job_id, [])
    return job


def all_jobs() -> list[Job]:
    return sorted(_jobs.values(), key=lambda j: j.id)


def remove(job_id: str) -> None:
    """Forget a job entirely: memory, subscribers, and the database row.

    Any stream still attached is closed first. Without that, a browser sitting
    on the deleted job's SSE endpoint would wait forever for events that can
    no longer arrive.
    """
    _close(job_id)
    _jobs.pop(job_id, None)
    _subscribers.pop(job_id, None)
    db.delete(job_id)


def _progress_for(stage: str) -> float:
    if stage == "done":
        return 1.0
    if stage not in STAGE_ORDER:
        return 0.0
    return round((STAGE_ORDER.index(stage) + 1) / (len(STAGE_ORDER) + 1), 3)


def set_stage(job_id: str, stage: str, detail: str = "") -> None:
    """Advance a job and fan the transition out to every live subscriber."""
    job = _jobs.get(job_id)
    if job is None:
        return

    job.stage = stage
    job.label = STAGE_LABELS.get(stage, stage)
    job.progress = _progress_for(stage)
    db.save(job)

    _publish(job_id, JobEvent(stage=stage, label=job.label, progress=job.progress, detail=detail))

    if stage in {"done", "failed"}:
        _close(job_id)


def fail(job_id: str, error: str) -> None:
    job = _jobs.get(job_id)
    if job is not None:
        job.error = error
    set_stage(job_id, "failed", detail=error)


def _publish(job_id: str, event: JobEvent) -> None:
    for queue in list(_subscribers.get(job_id, [])):
        queue.put_nowait(event)


def _close(job_id: str) -> None:
    """Signal end-of-stream to every subscriber on this job."""
    for queue in list(_subscribers.get(job_id, [])):
        queue.put_nowait(None)


async def subscribe(job_id: str) -> AsyncIterator[JobEvent]:
    """Yield events for a job until it reaches a terminal stage.

    Replays the job's current stage first, so a browser that connects late
    (or reconnects after a refresh) immediately sees the right state instead
    of an empty progress bar.
    """
    job = get(job_id)
    if job is None:
        return

    queue: asyncio.Queue[JobEvent | None] = asyncio.Queue()
    _subscribers.setdefault(job_id, []).append(queue)

    try:
        yield JobEvent(stage=job.stage, label=job.label, progress=job.progress, detail="")

        if job.stage in {"done", "failed"}:
            return

        while True:
            event = await queue.get()
            if event is None:
                return
            yield event
    finally:
        subs = _subscribers.get(job_id, [])
        if queue in subs:
            subs.remove(queue)
