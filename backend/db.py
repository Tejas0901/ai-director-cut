"""Minimal SQLite persistence for jobs.

One table, JSON blob per job. We are not querying across jobs, so a schema
would be ceremony; the point is surviving a server restart during the demo.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import DATA_DIR
from .schemas import Job

DB_PATH: Path = DATA_DIR / "jobs.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    return conn


def init() -> None:
    _connect().close()


def save(job: Job) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, payload) VALUES (?, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
            (job.id, job.model_dump_json()),
        )


def load(job_id: str) -> Job | None:
    with _connect() as conn:
        row = conn.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    try:
        return Job.model_validate(json.loads(row[0]))
    except Exception:
        return None


def delete(job_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def load_all() -> list[Job]:
    with _connect() as conn:
        rows = conn.execute("SELECT payload FROM jobs").fetchall()
    jobs: list[Job] = []
    for (payload,) in rows:
        try:
            jobs.append(Job.model_validate(json.loads(payload)))
        except Exception:
            continue
    return jobs
