# AI Director's Cut

Drop in raw footage. Get back a cut, scored, narrated highlight reel — entirely on your own machine.

A local web app that transcribes your video, watches it for visual energy, asks an LLM to act as
a film editor, then programmatically cuts the reel with FFmpeg and lays AI narration over the top.

---

## Run it

Two commands, two terminals:

```bash
uv run uvicorn backend.main:app --reload    # http://127.0.0.1:8000
npm --prefix frontend run dev               # http://localhost:5173
```

Open **http://localhost:5173** and drop in a video.

### First-time setup

```bash
uv python install 3.12 && uv venv --python 3.12
uv pip install -r pyproject.toml
npm --prefix frontend install
cp .env.example .env

uv run python -m assets.make_music          # background beds  (~10s)
uv run python -m fixtures.make_fixtures     # test clips       (~30s)
```

FFmpeg must be on your PATH (`ffmpeg -version`). Nothing else is required — the app runs
with a completely empty `.env`, just in offline heuristic mode.

The last two commands generate binaries that are deliberately not committed: three music
loops and two test videos. Skip them and the app still runs, it just renders without music
and you have nothing to test against.

---

## How it works

```
upload ─→ ingest ─→ transcribe ─→ vision ─→ timeline ─→ director ─→ narrate ─→ render ─→ UI
          ffprobe    Whisper       OpenCV    fusion      LLM         TTS        FFmpeg
```

| Stage | Module | What it does |
|---|---|---|
| Ingest | `pipeline/ingest.py` | Probes the file, extracts 16 kHz mono WAV |
| Transcribe | `pipeline/transcribe.py` | Timestamped transcript (faster-whisper or Groq) |
| Vision | `pipeline/vision.py` | Frame-diff motion, histogram scene cuts, face counts |
| Timeline | `pipeline/timeline.py` | Fuses speech + loudness + motion into one table |
| Director | `pipeline/director.py` | LLM picks the clips and writes the narration |
| Narrate | `pipeline/tts.py` | Voiceover audio (edge-tts or Piper) |
| Render | `pipeline/render.py` | Two-pass FFmpeg: cut, then mix |

The architectural rule: **`render.py` never imports the LLM, `director.py` never touches FFmpeg.**
Every stage communicates only through the Pydantic models in `backend/schemas.py`, which is what
lets any stage be stubbed, tested alone, or swapped for a different provider.

### The unified timeline

Each row is half a second of video. This table *is* the prompt:

```
| time | speech                        | loud | motion | cut | faces |
| 12.5 | and then it completely explod | 0.81 | 0.94   | Y   | 2     |
```

### The Director's output

Schema-enforced JSON — clips, narration, music mood, and a `reason` per clip that the UI shows
the viewer, so the AI's thinking is visible rather than hidden.

---

## Configuration

Everything lives in `.env` (see `.env.example`). The three that matter:

| Variable | Options | Default |
|---|---|---|
| `LLM_PROVIDER` | `gemini` · `groq` · `mock` | `gemini` |
| `STT_PROVIDER` | `local` · `groq` · `mock` | `local` |
| `TTS_PROVIDER` | `edge` · `piper` · `mock` | `edge` |
| `GEMINI_MODEL` | any model the key can call | `gemini-3.6-flash` |

`mock` needs no network anywhere: the Director falls back to an energy heuristic and the
narration becomes silence. **The app always produces a reel**, even with no keys and no internet.

Free API keys: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) ·
[console.groq.com/keys](https://console.groq.com/keys)

> **Models get retired.** Google returns `404 no longer available to new users` for older
> Gemini models on freshly-issued keys, and because the Director catches every failure and
> falls back, a dead model looks exactly like a working app producing dull cuts. If the
> Director seems uncreative, check the server log for `[director] gemini failed` before
> anything else. `GET /api/health` reports which providers are live.

### Development flags

```bash
STUB_STAGES=transcribing,directing   # skip stages with canned results
USE_CACHE=1                          # reuse Whisper/vision output per file hash
```

Every expensive stage caches to `data/cache/<file-hash>/`. Whisper runs once per video,
not once per run — this is the difference between a 30-second iteration loop and a 2-second one.

---

## Testing

Each stage runs standalone, no server needed:

```bash
uv run python -m backend.pipeline.ingest      fixtures/talkie.mp4
uv run python -m backend.pipeline.vision      fixtures/talkie.mp4   # prints a motion bar chart
uv run python -m backend.pipeline.transcribe  fixtures/talkie.mp4   # prints who said what, when
uv run python -m backend.pipeline.timeline    fixtures/talkie.mp4   # prints the LLM's prompt
uv run python -m backend.pipeline.director    fixtures/talkie.mp4
uv run python -m backend.pipeline.render      fixtures/talkie.mp4 --cut-only
```

Run both fixtures before demoing. They are different on purpose:

| Fixture | Length | What it proves |
|---|---|---|
| `talkie.mp4` | 36s | The happy path — real speech, so Whisper, transcript alignment and sentence-boundary cuts all run |
| `synthetic.mp4` | 12.5s | **No audio stream at all.** Forces the renderer to synthesise a silent track before concat, and the Director to choose on motion alone |

The second one matters more than it looks. A video with a *silent* audio track and a video
with *no audio track* take different paths through `cut_reel`, and only the latter exercises
the `anullsrc` branch — so a fixture that merely has no speech does not test it.

---

## Assets

The Director picks a `music_mood`, and the renderer looks for
`assets/music/<mood>.mp3`. Generate all three — they are synthesised from
scratch with numpy, so there is nothing to download and no licence to credit:

```bash
uv run python -m assets.make_music          # energetic.mp3, chill.mp3, dramatic.mp3
```

| Mood | Tempo | Character |
|---|---|---|
| `energetic` | 124 bpm | A minor, eighth-note arpeggio, four-on-the-floor kick |
| `chill` | 82 bpm | D major sevenths, soft pad, no drums |
| `dramatic` | 68 bpm | D minor, low sustained pad, sparse movement |

Each is a seamless loop — the release tail is folded back over the head — and
gets bedded in under the narration at `MUSIC_VOLUME`. Substitute your own files
of the same name if you prefer. If a file is missing the reel renders without
music rather than failing.

### Test fixtures

```bash
uv run python -m fixtures.make_fixtures     # synthetic.mp4 + talkie.mp4
```

`synthetic.mp4` is silent with hard scene cuts — the "no transcript" edge case.
`talkie.mp4` carries real spoken narration over changing scenes, which is what
exercises Whisper and sentence-boundary cutting. Both are generated, so neither
ships as committed binary.

---

## Demo day

Pre-flight, in order:

```bash
curl http://127.0.0.1:8000/api/health        # confirms which providers are actually live
ls assets/music                              # three mp3s, or there is no music bed
uv run python -m backend.pipeline.director fixtures/talkie.mp4
```

That last command is the real check: if the plan comes back titled *The Director's Cut* with
identical `reason` strings, the LLM is not being reached and you are watching the heuristic.
A live Director writes a title about your actual footage.

### When something breaks

| Symptom | Cause |
|---|---|
| `error while attempting to bind on 127.0.0.1:8000` | A server is already running. Find it with `Get-NetTCPConnection -LocalPort 8000`. A stale one serves **old code and old `.env`** — restart rather than reuse. |
| Director output is bland, no error shown | The LLM call failed and fell back silently. Check the log for `[director] gemini failed`. |
| `404 no longer available to new users` | `GEMINI_MODEL` is retired. Pick another; `GET /v1beta/models` lists what your key can call. |
| `503 high demand` | The free tier is busy. Requests already retry three times with backoff; if it still fails the heuristic takes over and the render completes. |
| Whisper returns 0 segments | Usually correct. Quiet or ambient-only audio is genuinely not speech — check the level before assuming a bug. |
| Reel has no music | `assets/music/` is empty. Run `make_music`. A missing bed is skipped, never fatal. |
| Uploads reappear after restart | Jobs persist in `data/jobs.db`. Delete it to start clean. |

### Recording the demo

Film **60–90 seconds**, facing the camera, with three or four distinct beats and a beat of
silence between thoughts — the pauses are where clip boundaries land. Vary your volume and
movement: clip selection is 45% loudness and 35% motion, so footage delivered at one flat
level gives the Director nothing to choose between.

**Check what is on your screen before you record it.** A clip of your own editor will happily
capture `.env`. Source video in the repo root is gitignored for exactly this reason.

---

## Stack

FastAPI · SQLite · Server-Sent Events · faster-whisper · OpenCV · FFmpeg · edge-tts ·
React + Vite + TypeScript
