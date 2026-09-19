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
```

FFmpeg must be on your PATH (`ffmpeg -version`). Nothing else is required — the app runs
with a completely empty `.env`, just in offline heuristic mode.

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

`mock` needs no network anywhere: the Director falls back to an energy heuristic and the
narration becomes silence. **The app always produces a reel**, even with no keys and no internet.

Free API keys: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) ·
[console.groq.com/keys](https://console.groq.com/keys)

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
uv run python -m backend.pipeline.ingest      fixtures/synthetic.mp4
uv run python -m backend.pipeline.vision      fixtures/synthetic.mp4   # prints a motion bar chart
uv run python -m backend.pipeline.transcribe  fixtures/synthetic.mp4
uv run python -m backend.pipeline.timeline    fixtures/synthetic.mp4   # prints the LLM's prompt
uv run python -m backend.pipeline.director    fixtures/synthetic.mp4
uv run python -m backend.pipeline.render      fixtures/synthetic.mp4 --cut-only
```

`fixtures/synthetic.mp4` is a generated 12-second clip with three hard scene cuts and no speech —
useful as the "nothing works the way you expect" edge case.

---

## Assets

Drop three loops into `assets/music/` named for the moods the Director can pick:

```
assets/music/energetic.mp3
assets/music/chill.mp3
assets/music/dramatic.mp3
```

Free sources: [Pixabay Music](https://pixabay.com/music/), [Incompetech](https://incompetech.com/).
If a file is missing the reel renders without a music bed rather than failing.

---

## Stack

FastAPI · SQLite · Server-Sent Events · faster-whisper · OpenCV · FFmpeg · edge-tts ·
React + Vite + TypeScript
