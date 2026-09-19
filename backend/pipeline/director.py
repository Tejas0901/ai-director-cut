"""Stage 5 - the Creative Director.

Hands the unified timeline to an LLM and gets back a structured EditPlan.
Providers: gemini (default, native JSON schema), groq (OpenAI-compatible JSON
mode), mock (heuristic, no network). Whatever comes back is forced through
`sanitize_plan` before the renderer is allowed near it.
"""

from __future__ import annotations

import json
import sys
import time

import httpx
import numpy as np

from ..config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    TARGET_CLIPS,
)
from ..schemas import Clip, EditPlan, Keyframe, MusicMood, Timeline, sanitize_plan
from .timeline import render_markdown, top_moments

SYSTEM_PROMPT = """\
You are a Hollywood film editor cutting a short highlight reel from raw footage.

You receive a timeline of the source video. Each row is a moment, with the
words spoken, how loud the audio is (0-1), how much the picture is moving
(0-1), whether a scene cut happened, and how many faces are visible.

WHAT THE NUMBERS MEAN. `loud` and `motion` are normalised against THIS video's
own range, not an absolute scale. 1.00 means "the most movement in this
video" - which in a chess match is a player reaching across the board, and in
a car chase is a crash. They tell you WHERE the notable moments are. They tell
you NOTHING about what kind of video this is. Never infer subject, genre, pace
or energy from them: a quiet video still has a 1.00 in it somewhere.

{vision_note}

WHAT YOU MAY NOT CLAIM. You are working from sampled stills and a transcript,
not the whole video, so anything you write has to be checkable against them.

- Never state how something turned out. You cannot see who won, whether the
  plan worked, or what happened between two sampled moments. No "in the end",
  no "finally", no "proves", no declaring a winner or a lesson learned.
- Never invent names, ages, relationships, roles or motives. Use them only if
  the transcript says them or they are written on screen. Two people near each
  other are not automatically family, rivals, teammates or colleagues.
- Never identify a real person from their face. Name someone only when the
  footage shows their name - a caption, a scoreboard, a name plate. The same
  goes for naming the event or competition.
- Say what kind of footage this is when it is not live action. Animation, a
  cartoon, a screen recording, gameplay or a video call must not be narrated
  as though it were a real event happening to real people.
- Describe what is happening rather than explaining what it means.

Staying general is always better than asserting something you cannot check.

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
- Ground every word in what you can actually see or hear. The title and the
  narration must describe THIS footage - its real setting, subject and pace.
  Never reach for stock excitement ("high-octane", "thrill ride", "hold onto
  your seats") unless the footage genuinely is that. A calm video deserves
  calm narration; writing it up as an action movie is the worst failure here.

For every clip, `reason` explains in one sentence why this moment earned its
place. The viewer reads these, so make them sharp rather than mechanical.
`overlay_title` is a punchy on-screen caption of at most four words.
"""

# Which of these the Director gets decides whether it can describe the footage
# or merely locate the interesting parts of it. Getting this wrong is what
# produced "high-octane thrill ride" over a chess match: given only numbers,
# a model fills the vacuum with the most common shape of video.
SEEING_NOTE = """\
WHAT YOU CAN SEE. Attached are {frames} still frames sampled evenly across the
video, each labelled with its timestamp. These are your only evidence of what
this footage is actually of, so read them before you write anything: identify
the medium (live action, animation, a screen recording), the setting, the
activity, who is present, and any on-screen text, branding or scoreboard. The
title, the narration and every clip caption must describe what is in those
frames. If the frames show a chess match, do not write about speed.

They are {frames} moments out of thousands. They tell you what this video IS.
They do not tell you the story between them, so do not narrate one.\
"""

BLIND_NOTE = """\
WHAT YOU CANNOT SEE. You have no frames from this video - the transcript is
your only evidence of subject matter. If it is thin or empty you genuinely do
not know what the footage shows, so write narration that works without
knowing: talk about moments, structure and rhythm rather than naming a
subject, sport, place or genre. Inventing one is worse than staying general.\
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
    """Pick the clips, and say who picked them.

    A dead key, a retired model or a rate limit must not kill the demo, so
    every failure lands on the heuristic. But a fallback that is invisible is
    its own bug - a dull cut looks exactly like a working app with dull
    footage - so the plan carries how it was made and, when it degraded, why.
    """
    provider = (provider or LLM_PROVIDER).lower()
    duration = timeline.media.duration
    reason: str | None = None

    if provider != "mock":
        try:
            # sanitize_plan is inside the try on purpose: a plan whose clips
            # all fall outside the video is as unusable as a 503, and should
            # degrade the same way rather than raise past here.
            # Only Gemini is multimodal here; the configured Groq model is
            # text-only, so it is told plainly that it cannot see.
            seeing = provider == "gemini"
            system, user = _build_prompts(timeline, can_see=seeing)
            plan = _gemini(system, user, timeline.keyframes) if seeing \
                else _groq(system, user)
            return _stamp(sanitize_plan(plan, duration), "llm")
        except Exception as exc:
            reason = _short_reason(provider, exc)
            print(f"[director] {provider} failed ({exc}); falling back to heuristic")

    # provider="mock" is a deliberate choice rather than a degradation, so it
    # reports the heuristic as its source with no reason attached.
    return _stamp(sanitize_plan(_mock(timeline), duration), "heuristic", reason)


def _stamp(plan: EditPlan, source: str, reason: str | None = None) -> EditPlan:
    """Record provenance on a finished plan.

    Set here rather than trusted from the response: under Groq's plain JSON
    mode a model is free to invent a `source` field, and a hallucinated
    "llm" on a heuristic cut would be worse than no badge at all.
    """
    return plan.model_copy(update={"source": source, "fallback_reason": reason})


# The badge this ends up in has room for a sentence, not a stack trace, and
# _post_json attaches up to 200 characters of response body to its message.
MAX_REASON_CHARS = 160


def _short_reason(provider: str, exc: Exception) -> str:
    """One line a viewer can act on: which provider, and what it said."""
    text = " ".join(str(exc).split()) or type(exc).__name__
    if len(text) > MAX_REASON_CHARS:
        text = text[:MAX_REASON_CHARS - 1].rstrip() + "…"
    return f"{provider}: {text}"


def _build_prompts(timeline: Timeline, *, can_see: bool) -> tuple[str, str]:
    """`can_see` is whether the frames will actually be sent, not whether we
    have any - a text-only provider must be told it is blind even when the
    timeline is carrying pictures."""
    frames = len(timeline.keyframes)
    system = SYSTEM_PROMPT.format(
        target=TARGET_CLIPS,
        vision_note=SEEING_NOTE.format(frames=frames) if can_see and frames else BLIND_NOTE,
    )
    user = USER_PROMPT.format(
        duration=timeline.media.duration,
        transcript=timeline.transcript.full_text or "(no speech detected)",
        timeline=render_markdown(timeline),
    )
    return system, user


# --------------------------------------------------------------------------

# Codes worth a second attempt: the model is briefly overloaded or we are being
# rate limited. Anything else (bad key, retired model, malformed request) will
# fail identically no matter how many times we ask, so we surface it at once.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


def _describe(response) -> str:
    """The sentence worth reading out of an error response, not the envelope.

    Gemini and Groq both bury the real explanation in
    {"error": {"message": ...}}. This message ends up on a badge in the UI, so
    "API key not valid" is what the viewer needs - not 200 characters of JSON
    with that phrase somewhere in the middle.
    """
    try:
        message = response.json()["error"]["message"]
        if isinstance(message, str) and message.strip():
            return f"HTTP {response.status_code}: {message.strip()}"
    except Exception:  # noqa: BLE001 - any shape but the expected one
        pass
    return f"HTTP {response.status_code}: {(response.text or '').strip()[:200]}"


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    """POST with backoff on transient failures.

    The free Gemini tier returns 503 "high demand" fairly often - roughly two
    in five cold calls during testing. Without a retry a single blip drops the
    whole run to the heuristic director, which is a bad way to lose the most
    interesting part of a demo.
    """
    last = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=120.0)
            if response.status_code == 200:
                return response.json()

            last = _describe(response)
            if response.status_code not in RETRY_STATUS:
                raise RuntimeError(last)
        except httpx.RequestError as exc:  # connection reset, timeout, DNS
            last = f"{type(exc).__name__}: {exc}"

        if attempt < MAX_ATTEMPTS - 1:
            delay = 1.5 * (2 ** attempt)
            print(f"[director] {last} - retrying in {delay:.1f}s "
                  f"({attempt + 2}/{MAX_ATTEMPTS})")
            time.sleep(delay)

    raise RuntimeError(f"all {MAX_ATTEMPTS} attempts failed - {last}")


def _gemini(system: str, user: str, keyframes: list[Keyframe] | None = None) -> EditPlan:
    if not GEMINI_API_KEY:
        raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")

    # Each frame is labelled with its own timestamp immediately before the
    # image, so the model can tie what it sees to a row in the timeline.
    parts: list[dict] = [{"text": user}]
    for frame in keyframes or []:
        parts.append({"text": f"Frame at {frame.t:.1f}s:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": frame.jpeg_b64}})

    payload = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_SCHEMA,
                "temperature": 0.7,
            },
        },
    )
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return EditPlan.model_validate(json.loads(text))


def _groq(system: str, user: str) -> EditPlan:
    if not GROQ_API_KEY:
        raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set")

    payload = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {GROQ_API_KEY}"},
        {
            "model": GROQ_MODEL,
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system + "\n\nRespond with JSON matching:\n"
                 + json.dumps(GEMINI_SCHEMA)},
                {"role": "user", "content": user},
            ],
        },
    )
    text = payload["choices"][0]["message"]["content"]
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
            reason=_heuristic_reason(timeline, start, end),
        )
        for i, (start, end) in enumerate(moments)
    ]

    return EditPlan(
        title="The Director's Cut",
        music_mood=_heuristic_mood(timeline),
        intro_narration="Here's everything worth watching, and nothing that isn't.",
        outro_summary="That was the good part. You're welcome.",
        clips=clips,
    )


def _heuristic_reason(timeline: Timeline, start: float, end: float) -> str:
    """Name the signal that actually won this window.

    The UI prints one reason per clip. Repeating a single generic sentence
    four times reads as a bug, so say which measurement drove the pick - it is
    the honest answer and it varies on its own.
    """
    window = [b for b in timeline.buckets if start <= b.t < end]
    if not window:
        return "Fallback selection: no timeline data covered this window."

    loud = max(b.audio_rms for b in window)
    motion = max(b.motion for b in window)
    cuts = sum(1 for b in window if b.scene_cut)
    faces = max(b.faces for b in window)
    spoken = " ".join(b.speech for b in window if b.speech).strip()

    if spoken and loud > 0.55:
        quote = spoken[:60].rsplit(" ", 1)[0] if len(spoken) > 60 else spoken
        return f'Loudest delivery in this stretch - the line lands on "{quote}".'
    if cuts >= 2:
        return f"{cuts} scene changes packed into {end - start:.0f} seconds - the busiest edit in the video."
    if motion > 0.7 and loud > 0.5:
        return "Motion and volume peak together here, which is usually where the good part is."
    if motion > 0.7:
        return f"Highest on-screen movement in the clip ({motion:.0%} of this video's peak)."
    if faces:
        return f"{faces} face{'s' if faces > 1 else ''} on camera with steady energy behind it."
    if spoken:
        return "Picked for the dialogue - a complete thought with clean silence either side."
    return f"Sustained energy through this window ({max(loud, motion):.0%} of peak)."


def _heuristic_mood(timeline: Timeline) -> MusicMood:
    """Match the bed to the footage instead of always shouting ENERGETIC."""
    if not timeline.buckets:
        return MusicMood.ENERGETIC

    motion = float(np.mean([b.motion for b in timeline.buckets]))
    loud = float(np.mean([b.audio_rms for b in timeline.buckets]))
    talky = sum(1 for b in timeline.buckets if b.speech) / len(timeline.buckets)

    if motion > 0.45 and loud > 0.4:
        return MusicMood.ENERGETIC
    if talky > 0.6 and motion < 0.35:
        return MusicMood.CHILL
    return MusicMood.DRAMATIC


if __name__ == "__main__":
    # Standalone check:  uv run python -m backend.pipeline.director <video>
    from ..cache import file_key
    from ..schemas import Transcript
    from .ingest import ingest
    from .timeline import build
    from .transcribe import transcribe
    from .vision import analyze, keyframes as extract_keyframes

    target = sys.argv[1]
    key = file_key(target)
    media_info = ingest(target, key)
    tl = build(
        media_info,
        transcribe(media_info.audio_path) if media_info.audio_path else Transcript(),
        analyze(target, media_info.duration),
        extract_keyframes(target, media_info.duration),
    )
    print(f"[director] showing the model {len(tl.keyframes)} frames")
    result = direct(tl)
    # Dumping the frames would bury the plan under a megabyte of base64.
    print(result.model_dump_json(indent=2))
