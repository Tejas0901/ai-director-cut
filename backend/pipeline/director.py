"""Stage 5 - the Creative Director.

Hands the unified timeline to an LLM and gets back a structured EditPlan.
Providers: gemini (default, native JSON schema), groq (OpenAI-compatible JSON
mode), mock (heuristic, no network). Whatever comes back is forced through
`sanitize_plan` before the renderer is allowed near it.
"""

from __future__ import annotations

import json
import sys

import httpx

from ..config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    TARGET_CLIPS,
)
from ..schemas import Clip, EditPlan, MusicMood, Timeline, sanitize_plan
from .timeline import render_markdown, top_moments

SYSTEM_PROMPT = """\
You are a Hollywood film editor cutting a short highlight reel from raw footage.

You receive a timeline of the source video. Each row is a moment, with the
words spoken, how loud the audio is (0-1), how much the picture is moving
(0-1), whether a scene cut happened, and how many faces are visible.

Pick the {target} most engaging moments and write narration for the reel.

How to choose:
- A great clip lands on a complete thought. Start just before someone begins a
  sentence and end just after they finish it. Never cut mid-word.
- Prefer moments where loudness AND motion are both high, or where the words
  themselves are surprising, funny, or the emotional turn of the video.
- Spread your picks across the whole video. Do not take three clips from the
  same 20 seconds.
- Each clip must be between 4 and 15 seconds long.
- If there is no speech at all, choose purely on motion, scene cuts and faces.

Narration:
- intro_narration: one or two sentences, spoken before the reel starts. Set up
  what the viewer is about to see. Confident, a little playful. Under 30 words.
- outro_summary: one sentence to close on, under 20 words.
- Write words a person would say out loud. No emoji, no stage directions, no
  markdown, no "In this video we will".

For every clip, `reason` explains in one sentence why this moment earned its
place. The viewer reads these, so make them sharp rather than mechanical.
`overlay_title` is a punchy on-screen caption of at most four words.
"""

USER_PROMPT = """\
Video duration: {duration:.1f} seconds.

Full transcript:
{transcript}

Timeline:
{timeline}

Return the edit plan. Every timestamp must fall between 0 and {duration:.1f}.
"""

# Gemini's structured-output schema. Keeping this next to the prompt (rather
# than generating it from the Pydantic model) makes the contract greppable and
# lets us tune descriptions without touching schemas.py.
GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING", "description": "Punchy title for the reel, max 6 words"},
        "music_mood": {"type": "STRING", "enum": ["energetic", "chill", "dramatic"]},
        "intro_narration": {"type": "STRING"},
        "outro_summary": {"type": "STRING"},
        "clips": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start_time": {"type": "NUMBER"},
                    "end_time": {"type": "NUMBER"},
                    "overlay_title": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["start_time", "end_time", "overlay_title", "reason"],
            },
        },
    },
    "required": ["title", "music_mood", "intro_narration", "outro_summary", "clips"],
}


def direct(timeline: Timeline, provider: str | None = None) -> EditPlan:
    provider = (provider or LLM_PROVIDER).lower()

    if provider == "mock":
        plan = _mock(timeline)
    else:
        prompts = _build_prompts(timeline)
        try:
            plan = _gemini(*prompts) if provider == "gemini" else _groq(*prompts)
        except Exception as exc:
            # A dead API key or a rate limit must not kill the demo.
            print(f"[director] {provider} failed ({exc}); falling back to heuristic")
            plan = _mock(timeline)

    return sanitize_plan(plan, timeline.media.duration)


def _build_prompts(timeline: Timeline) -> tuple[str, str]:
    system = SYSTEM_PROMPT.format(target=TARGET_CLIPS)
    user = USER_PROMPT.format(
        duration=timeline.media.duration,
        transcript=timeline.transcript.full_text or "(no speech detected)",
        timeline=render_markdown(timeline),
    )
    return system, user


# --------------------------------------------------------------------------


def _gemini(system: str, user: str) -> EditPlan:
    if not GEMINI_API_KEY:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")

    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_SCHEMA,
                "temperature": 0.7,
            },
        },
        timeout=120.0,
    )
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return EditPlan.model_validate(json.loads(text))


def _groq(system: str, user: str) -> EditPlan:
    if not GROQ_API_KEY:
        raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system + "\n\nRespond with JSON matching:\n"
                 + json.dumps(GEMINI_SCHEMA)},
                {"role": "user", "content": user},
            ],
        },
        timeout=120.0,
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    return EditPlan.model_validate(json.loads(text))


def _mock(timeline: Timeline) -> EditPlan:
    """Heuristic director - energy peaks, no network, always available."""
    moments = top_moments(timeline, TARGET_CLIPS, min_seconds=5.0)
    if not moments:
        moments = [(0.0, min(6.0, timeline.media.duration))]

    titles = ["The Setup", "It Escalates", "Peak Chaos", "The Payoff", "One More Thing"]
    clips = [
        Clip(
            start_time=start,
            end_time=end,
            overlay_title=titles[i % len(titles)],
            reason="Selected by the energy heuristic: sustained peak in combined "
                   "audio loudness and on-screen motion.",
        )
        for i, (start, end) in enumerate(moments)
    ]

    return EditPlan(
        title="The Director's Cut",
        music_mood=MusicMood.ENERGETIC,
        intro_narration="Here's everything worth watching, and nothing that isn't.",
        outro_summary="That was the good part. You're welcome.",
        clips=clips,
    )


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.director <video>
    from ..cache import file_key
    from ..schemas import Transcript
    from .ingest import ingest
    from .timeline import build
    from .transcribe import transcribe
    from .vision import analyze

    target = sys.argv[1]
    key = file_key(target)
    media_info = ingest(target, key)
    tl = build(
        media_info,
        transcribe(media_info.audio_path) if media_info.audio_path else Transcript(),
        analyze(target, media_info.duration),
    )
    result = direct(tl)
    print(result.model_dump_json(indent=2))
