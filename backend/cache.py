"""Content-addressed stage cache.

Whisper is the slowest thing in the pipeline and its output never changes for
a given file. During development you will run the pipeline dozens of times on
the same clip, so every expensive stage writes its result to
`data/cache/<sha256-of-file>/<stage>.json` and reads it back on the next run.

This is the single biggest development-speed lever in the project.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .config import CACHE_DIR, USE_CACHE

T = TypeVar("T", bound=BaseModel)


def file_key(path: str | Path) -> str:
    """SHA-256 of the file's bytes, truncated. Same video -> same key."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _path(key: str, stage: str) -> Path:
    return CACHE_DIR / key / f"{stage}.json"


def load(key: str, stage: str, model: type[T]) -> T | None:
    if not USE_CACHE:
        return None
    path = _path(key, stage)
    if not path.exists():
        return None
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        # A corrupt or stale-schema cache entry must never break a run.
        return None


def store(key: str, stage: str, value: BaseModel) -> None:
    if not USE_CACHE:
        return
    path = _path(key, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2), encoding="utf-8")


def load_list(key: str, stage: str, model: type[T]) -> list[T] | None:
    if not USE_CACHE:
        return None
    path = _path(key, stage)
    if not path.exists():
        return None
    try:
        return [model.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    except Exception:
        return None


def store_list(key: str, stage: str, values: list[BaseModel]) -> None:
    if not USE_CACHE:
        return
    path = _path(key, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [v.model_dump(mode="json") for v in values]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
