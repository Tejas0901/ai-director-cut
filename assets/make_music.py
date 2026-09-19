"""Generate the three background music beds the Director can choose from.

The Director returns a `music_mood` of energetic / chill / dramatic, and
render.py looks for `assets/music/<mood>.mp3`. Rather than shipping borrowed
audio - which means licensing questions in a project meant to be handed to
judges - we synthesise all three here from scratch.

This is a small additive synth: sine partials shaped by an ADSR envelope,
arranged over a chord progression. It will not win a Grammy, but it is real
music, it loops seamlessly, and it is unambiguously ours.

Usage:  uv run python -m assets.make_music [energetic|chill|dramatic|all]
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from backend.config import MUSIC_DIR
from backend.pipeline.ffmpeg_util import run

RATE = 44100
BARS = 8
BEATS_PER_BAR = 4
TAIL_SECONDS = 2.0  # rendered past the end, then folded back for a clean loop


# --------------------------------------------------------------------------
# Arrangements. Chords are MIDI note numbers; each chord holds for two bars.
# --------------------------------------------------------------------------

MOODS: dict[str, dict] = {
    "energetic": {
        "bpm": 124,
        "chords": [[57, 60, 64], [53, 57, 60], [52, 55, 60], [50, 55, 59]],
        "arp_division": 2,     # eighth notes
        "drums": True,
        "pad_level": 0.16,
        "arp_level": 0.30,
        "bass_octave": -24,
    },
    "chill": {
        "bpm": 82,
        "chords": [[50, 54, 57, 61], [47, 50, 54, 57], [43, 47, 50, 54], [45, 49, 52, 55]],
        "arp_division": 1,     # quarter notes, sparse
        "drums": False,
        "pad_level": 0.34,
        "arp_level": 0.16,
        "bass_octave": -12,
    },
    "dramatic": {
        "bpm": 68,
        "chords": [[50, 53, 57], [46, 50, 53], [45, 48, 53], [43, 46, 50]],
        "arp_division": 1,
        "drums": False,
        "pad_level": 0.42,
        "arp_level": 0.10,
        "bass_octave": -24,
    },
}


def midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def adsr(length: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """Amplitude envelope. Without one, every note clicks on and off."""
    a = max(1, int(attack * RATE))
    d = max(1, int(decay * RATE))
    r = max(1, int(release * RATE))
    s = max(0, length - a - d - r)

    return np.concatenate([
        np.linspace(0.0, 1.0, a),
        np.linspace(1.0, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0.0, r),
    ])[:length]


def tone(freq: float, seconds: float, partials: list[float], env: np.ndarray) -> np.ndarray:
    """Additive synthesis: stack sine partials, shape with the envelope.

    Slight detune on the upper partials keeps it from sounding like a test
    signal - perfectly harmonic sines read as 'beep', not 'instrument'.
    """
    length = int(seconds * RATE)
    t = np.arange(length) / RATE
    out = np.zeros(length, dtype=np.float32)

    for i, level in enumerate(partials, start=1):
        detune = 1.0 + (i - 1) * 0.0008
        out += level * np.sin(2.0 * np.pi * freq * i * detune * t)

    return out * env[:length]


def pad(freq: float, seconds: float) -> np.ndarray:
    """Slow-attack sustained voice - the harmonic bed under everything."""
    env = adsr(int(seconds * RATE), attack=0.6, decay=0.3, sustain=0.75, release=0.9)
    return tone(freq, seconds, [1.0, 0.32, 0.14, 0.06], env)


def pluck(freq: float, seconds: float) -> np.ndarray:
    """Fast-decay voice for the arpeggio."""
    env = adsr(int(seconds * RATE), attack=0.004, decay=0.28, sustain=0.18, release=0.30)
    return tone(freq, seconds, [1.0, 0.5, 0.25, 0.12, 0.06], env)


def bass(freq: float, seconds: float) -> np.ndarray:
    env = adsr(int(seconds * RATE), attack=0.01, decay=0.25, sustain=0.6, release=0.35)
    return tone(freq, seconds, [1.0, 0.22, 0.08], env)


def kick(seconds: float = 0.28) -> np.ndarray:
    """Pitch-swept sine - the classic synthesised kick drum."""
    length = int(seconds * RATE)
    t = np.arange(length) / RATE
    sweep = 110.0 * np.exp(-t * 28.0) + 44.0
    env = np.exp(-t * 11.0)
    return (np.sin(2.0 * np.pi * np.cumsum(sweep) / RATE) * env).astype(np.float32)


def hat(seconds: float = 0.06) -> np.ndarray:
    """Filtered noise burst. Differencing white noise is a cheap high-pass."""
    length = int(seconds * RATE)
    noise = np.random.default_rng(7).standard_normal(length).astype(np.float32)
    return np.diff(noise, prepend=0.0) * np.exp(-np.arange(length) / RATE * 70.0) * 0.28


# --------------------------------------------------------------------------


def _place(buffer: np.ndarray, signal: np.ndarray, at: int, level: float) -> None:
    """Mix `signal` into `buffer` at a sample offset, clipping at the end."""
    end = min(len(buffer), at + len(signal))
    if end > at:
        buffer[at:end] += signal[:end - at] * level


def compose(spec: dict) -> np.ndarray:
    beat = 60.0 / spec["bpm"]
    bar = beat * BEATS_PER_BAR
    loop_seconds = BARS * bar
    total = int((loop_seconds + TAIL_SECONDS) * RATE)

    mono = np.zeros(total, dtype=np.float32)
    chords = spec["chords"]
    bars_per_chord = BARS // len(chords)

    for index, chord in enumerate(chords):
        start = index * bars_per_chord * bar
        hold = bars_per_chord * bar

        # Pad: the whole chord, sustained across its bars.
        for note in chord:
            _place(mono, pad(midi_to_freq(note), hold + 0.8),
                   int(start * RATE), spec["pad_level"])

        # Bass: root of the chord, one note per bar.
        root = midi_to_freq(chord[0] + spec["bass_octave"])
        for b in range(bars_per_chord):
            _place(mono, bass(root, bar * 0.9), int((start + b * bar) * RATE), 0.40)

        # Arpeggio: cycle up through the chord tones.
        step = beat / spec["arp_division"]
        steps = int(hold / step)
        for s in range(steps):
            note = chord[s % len(chord)] + (12 if (s // len(chord)) % 2 else 0)
            _place(mono, pluck(midi_to_freq(note), step * 1.8),
                   int((start + s * step) * RATE), spec["arp_level"])

    if spec["drums"]:
        for b in range(int(loop_seconds / beat)):
            _place(mono, kick(), int(b * beat * RATE), 0.55)
            _place(mono, hat(), int((b + 0.5) * beat * RATE), 0.35)

    # Fold the tail back over the head so the loop point is seamless: the
    # release of the last notes rings into the start of the next repetition,
    # exactly as it would if the track really were continuous.
    loop_samples = int(loop_seconds * RATE)
    tail = mono[loop_samples:]
    mono = mono[:loop_samples].copy()
    mono[:len(tail)] += tail

    # Gentle one-pole low-pass takes the edge off the sine stack.
    smoothed = np.copy(mono)
    for _ in range(2):
        smoothed[1:] = 0.72 * smoothed[1:] + 0.28 * smoothed[:-1]

    peak = float(np.max(np.abs(smoothed))) or 1.0
    smoothed = (smoothed / peak) * 0.9

    # Widen to stereo by nudging the channels a few milliseconds apart.
    offset = int(0.012 * RATE)
    left = smoothed
    right = np.concatenate([smoothed[offset:], smoothed[:offset]])
    return np.stack([left, right], axis=1)


def write_mood(name: str) -> Path:
    stereo = compose(MOODS[name])
    wav_path = MUSIC_DIR / f"{name}.wav"
    mp3_path = MUSIC_DIR / f"{name}.mp3"

    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes((stereo * 32767).astype("<i2").tobytes())

    run(["-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)])
    wav_path.unlink(missing_ok=True)
    return mp3_path


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    targets = list(MOODS) if which == "all" else [which]

    for mood in targets:
        path = write_mood(mood)
        seconds = BARS * BEATS_PER_BAR * 60.0 / MOODS[mood]["bpm"]
        print(f"{path.name:<16} {seconds:5.1f}s loop  "
              f"{MOODS[mood]['bpm']} bpm  {path.stat().st_size // 1024} KB")
