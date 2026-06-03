#!/usr/bin/env python3
"""CLI: run the transcription pipeline on an audio file.

Usage:
    python scripts/run_pipeline.py path/to/song.mp3 --out data/interim/song --stems
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gtab.io.export import save_json, save_stems  # noqa: E402
from gtab.pipeline import build_default_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio -> guitar note/technique signal pipeline"
    )
    parser.add_argument("audio", help="Path to an input audio file")
    parser.add_argument("--out", default="data/interim/run", help="Output directory")
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate")
    parser.add_argument(
        "--stems", action="store_true", help="Also export WAV stems (guitar + backing)"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pipeline = build_default_pipeline()
    output = pipeline.run(args.audio, target_sr=args.sr)

    notes_path = os.path.join(args.out, "notes.json")
    save_json(output.transcription, notes_path)
    if args.stems:
        save_stems(output, args.out)

    print(f"Done. Transcribed {len(output.transcription.notes)} notes -> {notes_path}")


if __name__ == "__main__":
    main()
