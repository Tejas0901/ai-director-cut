"""Shared data contracts.

Every pipeline stage consumes and produces these types. No stage imports
another stage; they communicate only through the models defined here.
That is what lets us stub a stage, test a stage in isolation, and swap a
provider without touching anything downstream.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Stage 1 output: ingest
# --------------------------------------------------------------------------


class MediaInfo(BaseModel):
    """Probed facts about the uploaded file."""

    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_path: str | None = None


# --------------------------------------------------------------------------
# Stage 2 output: transcription
# --------------------------------------------------------------------------


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Transcript(BaseModel):
    segments: list[TranscriptSegment] = Field(default_factory=list)
    language: str = "en"

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def is_empty(self) -> bool:
        return not self.full_text


# --------------------------------------------------------------------------
# Stage 3 output: visual analysis
# --------------------------------------------------------------------------


class VisualBucket(BaseModel):
    """Visual activity in one fixed-width slice of time."""

    t: float
    motion: float = 0.0  # 0..1 normalised frame-difference magnitude
    scene_cut: bool = False  # histogram correlation broke the threshold
    faces: int = 0


# --------------------------------------------------------------------------
# Stage 4 output: the unified timeline handed to the Director
# --------------------------------------------------------------------------


class TimelineBucket(BaseModel):
    """One row of the unified timeline: what is happening at time `t`."""

    t: float
    speech: str = ""
    audio_rms: float = 0.0
    motion: float = 0.0
    scene_cut: bool = False
    faces: int = 0

    @property
    def energy(self) -> float:
        """Cheap heuristic score, used for fallback clip selection."""
        return (
            0.45 * self.audio_rms
            + 0.35 * self.motion
            + 0.10 * (1.0 if self.scene_cut else 0.0)
            + 0.10 * min(self.faces, 2) / 2.0
        )


class Timeline(BaseModel):
    media: MediaInfo
    transcript: Transcript
    buckets: list[TimelineBucket] = Field(default_factory=list)
    bucket_seconds: float = 0.5


# --------------------------------------------------------------------------
# Stage 5 output: the Director's decision (the LLM's structured response)
# --------------------------------------------------------------------------


class MusicMood(str, Enum):
    ENERGETIC = "energetic"
    CHILL = "chill"
    DRAMATIC = "dramatic"


class Clip(BaseModel):
    start_time: float
    end_time: float
    overlay_title: str = ""
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class EditPlan(BaseModel):
    """Exactly what the LLM must return. Also what the renderer consumes."""

    title: str
    music_mood: MusicMood = MusicMood.ENERGETIC
    intro_narration: str
    outro_summary: str
    clips: list[Clip] = Field(default_factory=list)

    @field_validator("clips")
    @classmethod
    def _at_least_one_clip(cls, v: list[Clip]) -> list[Clip]:
        if not v:
            raise ValueError("EditPlan must contain at least one clip")
        return v


MIN_CLIP_SECONDS = 1.5
MAX_CLIP_SECONDS = 25.0


def sanitize_plan(plan: EditPlan, duration: float) -> EditPlan:
    """Make an LLM-authored plan safe for the renderer.

    LLMs hallucinate timestamps past the end of the video, emit zero-length
    clips, and occasionally invert start/end. The renderer must never see any
    of that, so every plan passes through here before it is executed.
    """
    cleaned: list[Clip] = []
    for clip in plan.clips:
        start, end = sorted((float(clip.start_time), float(clip.end_time)))
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start < MIN_CLIP_SECONDS:
            continue
        if end - start > MAX_CLIP_SECONDS:
            end = start + MAX_CLIP_SECONDS
        cleaned.append(
            Clip(
                start_time=round(start, 3),
                end_time=round(end, 3),
                overlay_title=clip.overlay_title.strip()[:60],
                reason=clip.reason.strip(),
            )
        )

    cleaned.sort(key=lambda c: c.start_time)

    # Drop clips that overlap an earlier keeper - duplicated footage looks broken.
    deduped: list[Clip] = []
    for clip in cleaned:
        if deduped and clip.start_time < deduped[-1].end_time:
            continue
        deduped.append(clip)

    if not deduped:
        raise ValueError("no usable clips survived sanitisation")

    return plan.model_copy(update={"clips": deduped[:5]})


# --------------------------------------------------------------------------
# Job state (what the API and the SSE stream expose)
# --------------------------------------------------------------------------

StageName = Literal[
    "queued",
    "ingesting",
    "transcribing",
    "analyzing_video",
    "building_timeline",
    "directing",
    "narrating",
    "rendering",
    "done",
    "failed",
]

STAGE_LABELS: dict[str, str] = {
    "queued": "Queued",
    "ingesting": "Reading the footage...",
    "transcribing": "Listening to the audio...",
    "analyzing_video": "Watching for the good bits...",
    "building_timeline": "Building the timeline...",
    "directing": "Consulting the AI Director...",
    "narrating": "Recording the voiceover...",
    "rendering": "Cutting the reel...",
    "done": "Your cut is ready",
    "failed": "Something broke",
}

STAGE_ORDER: list[str] = [
    "ingesting",
    "transcribing",
    "analyzing_video",
    "building_timeline",
    "directing",
    "narrating",
    "rendering",
]


class JobEvent(BaseModel):
    """One line on the live progress feed."""

    stage: str
    label: str
    progress: float = 0.0  # 0..1 across the whole pipeline
    detail: str = ""


class Job(BaseModel):
    id: str
    filename: str
    stage: str = "queued"
    label: str = STAGE_LABELS["queued"]
    progress: float = 0.0
    error: str | None = None

    original_url: str | None = None
    output_url: str | None = None

    plan: EditPlan | None = None
    transcript: Transcript | None = None
    duration: float | None = None
