#!/usr/bin/env python3
"""Measure guitar-separation quality: run a separator on a mix and report SDR.

Loads the mix, runs the chosen `Separator`, and compares the estimated guitar
stem to a reference guitar recording via global SDR (higher = better). Optionally
exports the separated stems so you can listen.

Audio is loaded at 44.1 kHz (Demucs' native rate); do not downsample first.

Usage:
    python scripts/make_test_mix.py --out data/raw
    python scripts/eval_separation.py --mix data/raw/mix.wav --ref data/raw/guitar_ref.wav --impl demucs
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gtab.config import SEPARATORS  # noqa: E402
from gtab.eval.separation import sdr_buffers  # noqa: E402
from gtab.io.export import save_stems  # noqa: E402
from gtab.stages.ingest import load_audio  # noqa: E402
from gtab.types import PipelineOutput, TranscriptionResult  # noqa: E402

SR = 44100


def build_separator(impl: str, device: str, shifts: int):
    cls = SEPARATORS[impl]
    if impl == "demucs":
        return cls(device=device, shifts=shifts)
    return cls()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a separator's guitar SDR")
    p.add_argument("--mix", required=True, help="Path to the full-mix audio")
    p.add_argument("--ref", required=True, help="Path to the reference guitar audio")
    p.add_argument(
        "--impl", default="demucs", choices=sorted(SEPARATORS), help="Separator impl"
    )
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (demucs)")
    p.add_argument("--shifts", type=int, default=1, help="Demucs test-time shifts")
    p.add_argument("--out", default=None, help="Export stems to this dir")
    p.add_argument(
        "--save-stems",
        action="store_true",
        help="Dump guitar.wav + backing.wav for listening (uses --out, or a "
        "default dir derived from the mix name)",
    )
    args = p.parse_args()

    mix = load_audio(args.mix, target_sr=SR)
    ref = load_audio(args.ref, target_sr=SR)

    separator = build_separator(args.impl, args.device, args.shifts)
    print(f"Separating with impl='{args.impl}' ...")
    stems = separator.separate(mix)

    sdr = sdr_buffers(stems.guitar, ref)
    print(f"\nGuitar SDR ({args.impl}): {sdr:+.2f} dB")
    if stems.extras:
        print(f"Non-guitar stems: {sorted(stems.extras)}")

    if args.out or args.save_stems:
        out_dir = args.out
        if out_dir is None:
            mix_stem = os.path.splitext(os.path.basename(args.mix))[0]
            out_dir = os.path.join("data", "interim", f"{mix_stem}_{args.impl}")
        output = PipelineOutput(
            stems=stems,
            transcription=TranscriptionResult(notes=[]),
            source_path=args.mix,
            sample_rate=stems.guitar.sample_rate,
        )
        save_stems(output, out_dir)
        print(f"Wrote guitar.wav + backing.wav -> {out_dir}")


if __name__ == "__main__":
    main()
