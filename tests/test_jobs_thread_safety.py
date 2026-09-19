"""The event bus is written from a worker thread. These tests hold it to that.

`runner.py` drives the whole pipeline inside `asyncio.to_thread`, so every
`set_stage()` call reaches `jobs.py` from a thread that is not the event loop.
Poking `asyncio.Queue.put_nowait` from there lands the item in the deque
without waking the loop, and a browser sitting idle on the SSE stream sees
nothing until something unrelated nudges the selector. Both tests below time
out against that version and pass against `call_soon_threadsafe`.

The loop must be genuinely idle while it waits - no polling, no other tasks -
or the bug hides behind whatever else was waking it.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from backend import db, jobs
from backend.schemas import STAGE_ORDER, JobEvent

# How long any single event may take to arrive. Generous beside the
# microseconds a threadsafe wakeup costs, and far under the multi-second
# selector stall the unpatched code produced.
DEADLINE = 1.0

# Separate, longer bound that only stops the suite hanging. It has to sit
# clear of DEADLINE: a stalled event tends to arrive the instant some other
# timer wakes the loop, so a cancellation armed at exactly DEADLINE races the
# delivery it is meant to catch. Measure against DEADLINE, cancel well after.
HARD_STOP = 5.0

# Stand-in for a stage doing real work, so the loop is already asleep by the
# time the next event is published.
STAGE_WORK_SECONDS = 0.01


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Keep the tests off `data/jobs.db` and out of each other's registry."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "jobs.db")
    jobs._jobs.clear()
    jobs._subscribers.clear()
    yield
    jobs._jobs.clear()
    jobs._subscribers.clear()


def fake_pipeline(job_id: str, stages: list[str]) -> None:
    """Stand in for `runner._run_sync`: the real calls, none of the work."""
    for stage in stages:
        time.sleep(STAGE_WORK_SECONDS)
        if stage == "failed":
            jobs.fail(job_id, "ffmpeg exploded")
        else:
            jobs.set_stage(job_id, stage, detail=f"{stage} finished")


async def subscribe_while_pipeline_runs(
    job_id: str, stages: list[str]
) -> tuple[JobEvent, list[tuple[JobEvent, float]]]:
    """Subscribe from the loop, run the pipeline in a thread, time every event.

    Every event is asserted prompt here rather than in the tests, so a test
    that only checks ordering still fails when delivery stalls.

    Returns the replayed current-state event and then one
    `(event, seconds waited)` pair per event the pipeline published.
    """
    stream = jobs.subscribe(job_id).__aiter__()

    # The generator body does not start until the first `__anext__`, and it is
    # that first step which registers the subscriber. Pump the replayed event
    # before starting the thread, or the pipeline races ahead and publishes
    # into an empty subscriber list.
    async with asyncio.timeout(HARD_STOP):
        replay = await stream.__anext__()

    worker = threading.Thread(target=fake_pipeline, args=(job_id, stages), daemon=True)
    worker.start()

    received: list[tuple[JobEvent, float]] = []
    try:
        while True:
            waited_from = time.monotonic()
            try:
                async with asyncio.timeout(HARD_STOP):
                    event = await stream.__anext__()
            except StopAsyncIteration:
                # `_close` crosses the same boundary as `_publish`, so hold
                # the end-of-stream sentinel to the same deadline.
                closed_after = time.monotonic() - waited_from
                assert closed_after < DEADLINE, (
                    f"end-of-stream sentinel took {closed_after:.3f}s"
                )
                break
            waited = time.monotonic() - waited_from
            assert waited < DEADLINE, (
                f"event {len(received) + 1} ({event.stage}) took {waited:.3f}s "
                f"to cross the thread boundary"
            )
            received.append((event, waited))
    finally:
        await stream.aclose()
        worker.join(timeout=HARD_STOP)

    return replay, received


def test_events_published_from_a_worker_thread_arrive_promptly_and_in_order():
    job = jobs.create("clip.mp4", "/media/uploads/clip.mp4")
    stages = [*STAGE_ORDER, "done"]

    replay, received = asyncio.run(subscribe_while_pipeline_runs(job.id, stages))

    assert replay.stage == "queued"
    assert [event.stage for event, _ in received] == stages
    assert [event.detail for event, _ in received] == [f"{s} finished" for s in stages]

    # Promptness is asserted per event inside the helper.
    progress = [event.progress for event, _ in received]
    assert progress == sorted(progress), "progress went backwards"
    assert received[-1][0].progress == 1.0


def test_failure_from_a_worker_thread_reaches_the_stream_and_closes_it():
    """`_close` crosses the same thread boundary as `_publish`.

    If its sentinel never wakes the loop the iteration above never ends, so
    reaching the assertions at all is the real check here.
    """
    job = jobs.create("broken.mp4", "/media/uploads/broken.mp4")
    stages = ["ingesting", "transcribing", "failed"]

    _, received = asyncio.run(subscribe_while_pipeline_runs(job.id, stages))

    assert [event.stage for event, _ in received] == stages
    assert received[-1][0].detail == "ffmpeg exploded"

    finished = jobs.get(job.id)
    assert finished is not None and finished.error == "ffmpeg exploded"
