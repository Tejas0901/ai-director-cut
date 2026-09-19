"""The stage cache trades correctness for speed, so it has to fail safe.

A cache entry that is corrupt, truncated, or written by an older version of a
schema must read back as "nothing cached" and cost one slow run - never an
exception that kills the job it was meant to accelerate.
"""

from __future__ import annotations

import pytest

from backend import cache
from backend.schemas import Transcript, TranscriptSegment, VisualBucket


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """cache.py binds CACHE_DIR and USE_CACHE at import, so patch them there."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "USE_CACHE", True)
    return tmp_path


def a_transcript() -> Transcript:
    return Transcript(segments=[TranscriptSegment(start=0.0, end=1.5, text="hello")])


def entry(root, key: str, stage: str):
    return root / key / f"{stage}.json"


# --------------------------------------------------------------------------
# file_key
# --------------------------------------------------------------------------


def test_the_same_bytes_give_the_same_key(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"identical content")
    b.write_bytes(b"identical content")

    assert cache.file_key(a) == cache.file_key(b)


def test_different_bytes_give_different_keys(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"one")
    b.write_bytes(b"two")

    assert cache.file_key(a) != cache.file_key(b)


def test_a_key_is_filename_safe(tmp_path):
    """It becomes a directory name, so it cannot carry a separator."""
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")

    key = cache.file_key(f)

    assert key.isalnum() and "/" not in key and "\\" not in key


def test_a_file_larger_than_one_chunk_still_hashes(tmp_path):
    """The reader loops in 1 MiB chunks; a video is many of them."""
    f = tmp_path / "big.mp4"
    f.write_bytes(b"z" * (3 * (1 << 20) + 17))

    assert len(cache.file_key(f)) == 16


# --------------------------------------------------------------------------
# single-model round trip
# --------------------------------------------------------------------------


def test_a_stored_model_reads_back_equal():
    cache.store("k", "transcript", a_transcript())

    assert cache.load("k", "transcript", Transcript) == a_transcript()


def test_a_missing_entry_is_a_miss_not_an_error():
    assert cache.load("never-written", "transcript", Transcript) is None


def test_two_stages_under_one_key_do_not_collide():
    cache.store("k", "transcript", a_transcript())
    cache.store("k", "other", Transcript())

    assert cache.load("k", "transcript", Transcript) == a_transcript()
    assert cache.load("k", "other", Transcript) == Transcript()


def test_storing_twice_overwrites():
    cache.store("k", "transcript", a_transcript())
    cache.store("k", "transcript", Transcript())

    assert cache.load("k", "transcript", Transcript) == Transcript()


# --------------------------------------------------------------------------
# corruption
# --------------------------------------------------------------------------


@pytest.mark.parametrize("junk", [
    "",                                  # truncated to nothing
    "{",                                 # truncated mid-write
    "not json at all",
    "null",
    "[]",                                # right JSON, wrong kind
    '{"segments": "not a list"}',        # right shape, wrong types
    '{"segments": [{"start": "x"}]}',    # right keys, unparseable values
])
def test_a_corrupt_entry_reads_as_a_miss(isolated_cache, junk):
    """One slow run beats an exception out of a cache read."""
    path = entry(isolated_cache, "k", "transcript")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(junk, encoding="utf-8")

    assert cache.load("k", "transcript", Transcript) is None


def test_a_corrupt_entry_can_be_overwritten_by_a_good_one(isolated_cache):
    path = entry(isolated_cache, "k", "transcript")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ truncated", encoding="utf-8")

    assert cache.load("k", "transcript", Transcript) is None
    cache.store("k", "transcript", a_transcript())
    assert cache.load("k", "transcript", Transcript) == a_transcript()


def test_an_entry_from_an_older_schema_reads_as_a_miss(isolated_cache):
    """Adding a required field must not make old entries explode."""
    path = entry(isolated_cache, "k", "visual")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[{"motion": 0.5}]', encoding="utf-8")  # no `t`

    assert cache.load_list("k", "visual", VisualBucket) is None


# --------------------------------------------------------------------------
# list round trip
# --------------------------------------------------------------------------


def test_a_stored_list_reads_back_equal():
    buckets = [VisualBucket(t=0.0, motion=0.1),
               VisualBucket(t=0.5, motion=0.9, scene_cut=True, faces=2)]

    cache.store_list("k", "visual", buckets)

    assert cache.load_list("k", "visual", VisualBucket) == buckets


def test_an_empty_list_round_trips_as_empty_not_as_a_miss():
    """`[]` is a real answer - a video with no visual activity."""
    cache.store_list("k", "visual", [])

    assert cache.load_list("k", "visual", VisualBucket) == []


def test_a_missing_list_entry_is_a_miss():
    assert cache.load_list("nothing", "visual", VisualBucket) is None


@pytest.mark.parametrize("junk", ["", "{", "not json", '{"not": "a list"}'])
def test_a_corrupt_list_entry_reads_as_a_miss(isolated_cache, junk):
    path = entry(isolated_cache, "k", "visual")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(junk, encoding="utf-8")

    assert cache.load_list("k", "visual", VisualBucket) is None


# --------------------------------------------------------------------------
# the off switch
# --------------------------------------------------------------------------


def test_disabling_the_cache_stops_reads(monkeypatch):
    cache.store("k", "transcript", a_transcript())
    monkeypatch.setattr(cache, "USE_CACHE", False)

    assert cache.load("k", "transcript", Transcript) is None
    assert cache.load_list("k", "visual", VisualBucket) is None


def test_disabling_the_cache_stops_writes(monkeypatch, isolated_cache):
    monkeypatch.setattr(cache, "USE_CACHE", False)

    cache.store("k", "transcript", a_transcript())
    cache.store_list("k", "visual", [VisualBucket(t=0.0)])

    assert not entry(isolated_cache, "k", "transcript").exists()
    assert not entry(isolated_cache, "k", "visual").exists()
