#!/usr/bin/env python3
"""Generate a synthetic mix for wiring-checking the separation stage.

Writes two files:
  - guitar_ref.wav : an isolated synthetic guitar phrase (the reference)
  - mix.wav        : that guitar summed with a synthetic drum+bass backing

This lets `eval_separation.py` measure guitar SDR end to end without any real
recordings. IMPORTANT: synthetic audio is out-of-distribution for Demucs, so the
absolute SDR will be modest -- this proves the wiring + stem routing are correct,
NOT real-world quality. Always confirm quality on real audio too (see the spec).

Everything is written at 44.1 kHz because Demucs is a 44.1 kHz model and we must
not downsample before separation.

Usage:
    python scripts/make_test_mix.py --out data/raw
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Reuse the Karplus-Strong guitar synth from the sibling script.
sys.path.insert(0, os.path.dirname(__file__))
from make_test_audio import karplus_strong, midi_to_hz  # noqa: E402

SR = 44100


def guitar_phrase() -> np.ndarray:
    """E-minor-pentatonic phrase then an E3+B3 dyad (same notes as make_test_audio)."""
    phrase_midi = [52, 55, 57, 59, 62, 64]
    note_dur = 0.45
    audio = np.concatenate(
        [karplus_strong(midi_to_hz(m), note_dur, SR) for m in phrase_midi]
    )
    dyad = 0.5 * (
        karplus_strong(midi_to_hz(52), 0.9, SR) + karplus_strong(midi_to_hz(59), 0.9, SR)
    )
    return np.concatenate([audio, dyad]).astype(np.float32)


def backing_track(n_samples: int) -> np.ndarray:
    """A crude drum+bass loop: a walking bass line plus periodic noise 'hits'."""
    t = np.arange(n_samples) / SR

    # Walking bass: E1/A1/B1-ish, one note per half second.
    bass = np.zeros(n_samples, dtype=np.float32)
    bass_midi = [28, 33, 35, 33]
    seg = int(0.5 * SR)
    for i, m in enumerate(bass_midi * (n_samples // (seg * len(bass_midi)) + 1)):
        start = i * seg
        if start >= n_samples:
            break
        end = min(start + seg, n_samples)
        tt = t[start:end] - t[start]
        env = np.exp(-3.0 * tt)
        bass[start:end] += (np.sin(2 * np.pi * midi_to_hz(m) * tt) * env).astype(
            np.float32
        )

    # Drums: short noise bursts every 0.25 s (a steady backbeat).
    drums = np.zeros(n_samples, dtype=np.float32)
    hit = int(0.25 * SR)
    rng = np.random.default_rng(0)
    for start in range(0, n_samples, hit):
        end = min(start + int(0.05 * SR), n_samples)
        tt = np.arange(end - start) / SR
        burst = rng.uniform(-1, 1, end - start) * np.exp(-60.0 * tt)
        drums[start:end] += burst.astype(np.float32)

    backing = 0.6 * bass + 0.4 * drums
    return backing.astype(np.float32)


def normalise(x: np.ndarray) -> np.ndarray:
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Make a synthetic guitar+backing mix")
    p.add_argument("--out", default="data/raw", help="Output directory")
    args = p.parse_args()

    import soundfile as sf

    os.makedirs(args.out, exist_ok=True)

    guitar = normalise(guitar_phrase())
    backing = normalise(backing_track(len(guitar)))  # comparable levels

    # Apply ONE scale factor to both the mix and the reference, so the reference
    # is the guitar *as it appears in the mix*. Global SDR is scale-sensitive, so
    # the reference and the estimate must share the same scale to be comparable.
    combined = guitar + backing
    scale = 1.0 / (np.max(np.abs(combined)) + 1e-9)
    mix = (combined * scale).astype(np.float32)
    guitar_ref = (guitar * scale).astype(np.float32)

    ref_path = os.path.join(args.out, "guitar_ref.wav")
    mix_path = os.path.join(args.out, "mix.wav")
    sf.write(ref_path, guitar_ref, SR)
    sf.write(mix_path, mix, SR)

    dur = len(mix) / SR
    print(f"Wrote {dur:.1f}s @ {SR} Hz:")
    print(f"  reference guitar -> {ref_path}")
    print(f"  full mix         -> {mix_path}")
    print("Ground truth: E-minor-pentatonic phrase + E3+B3 dyad over drum/bass backing.")


if __name__ == "__main__":
    main()
