#!/usr/bin/env python3
"""Generate a synthetic isolated-guitar test clip via Karplus-Strong synthesis.

Why this exists: separation is currently a passthrough, so the pipeline expects
*already-isolated* guitar as input. Until you have real recordings, this gives
you a known, controllable test signal whose notes you already know -- so you can
check the transcriber output against ground truth. The plucked-string timbre is
far more realistic for the transcriber than a pure sine wave.

(Bonus: Karplus-Strong is the simplest physical-modelling synth -- the same idea
behind rendering audio from tab for synthetic training data.)

Usage:
    python scripts/make_test_audio.py --out data/raw/guitar_test.wav
"""
from __future__ import annotations

import argparse

import numpy as np


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def karplus_strong(freq: float, duration: float, sr: int, decay: float = 0.996) -> np.ndarray:
    """One plucked note. The delay-line length (sr/freq) sets the pitch; the
    running average acts as a low-pass filter so harmonics die off over time,
    which is what gives the characteristic plucked-string decay."""
    n = max(2, int(round(sr / freq)))
    buf = np.random.uniform(-1.0, 1.0, size=n)  # the 'pluck' = a burst of noise
    total = int(duration * sr)
    out = np.empty(total, dtype=np.float32)
    for i in range(total):
        out[i] = buf[i % n]
        nxt = buf[(i + 1) % n]
        buf[i % n] = decay * 0.5 * (buf[i % n] + nxt)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Make a synthetic guitar test clip")
    p.add_argument("--out", default="data/raw/guitar_test.wav")
    p.add_argument("--sr", type=int, default=22050)
    args = p.parse_args()

    import soundfile as sf

    sr = args.sr
    # An E-minor-pentatonic phrase: E3 G3 A3 B3 D4 E4 (MIDI numbers).
    phrase_midi = [52, 55, 57, 59, 62, 64]
    note_dur = 0.45
    audio = np.concatenate(
        [karplus_strong(midi_to_hz(m), note_dur, sr) for m in phrase_midi]
    )

    # Finish with an E3+B3 dyad so the transcriber sees a little polyphony.
    dyad = 0.5 * (
        karplus_strong(midi_to_hz(52), 0.9, sr) + karplus_strong(midi_to_hz(59), 0.9, sr)
    )
    audio = np.concatenate([audio, dyad]).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-9  # normalise to [-1, 1]

    sf.write(args.out, audio, sr)
    print(
        f"Wrote {len(audio) / sr:.1f}s -> {args.out}\n"
        f"Ground truth: notes {phrase_midi} then an E3+B3 dyad."
    )


if __name__ == "__main__":
    main()
