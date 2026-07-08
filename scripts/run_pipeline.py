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

from gtab.io.export import save_json, save_midi, save_stems, synthesize_wav  # noqa: E402
from gtab.pipeline import build_default_pipeline  # noqa: E402
from gtab.types import TranscriptionResult  # noqa: E402

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(pitch_midi: float) -> str:
    """MIDI note number -> scientific pitch name, e.g. 52 -> 'E3'."""
    m = int(round(pitch_midi))
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


def print_notes(result: TranscriptionResult) -> None:
    """Print a readable, time-ordered table of recognised notes."""
    notes = sorted(result.notes, key=lambda n: (n.note.onset, n.note.pitch_midi))
    if not notes:
        print("No notes recognised.")
        return

    if result.tempo_bpm is not None:
        print(f"Tempo: {result.tempo_bpm:.1f} BPM")
    print(f"Recognised {len(notes)} notes:\n")

    header = f"  {'#':>3}  {'onset':>7}  {'offset':>7}  {'dur':>6}  {'pitch':<7}  {'conf':>5}  techniques"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, an in enumerate(notes, 1):
        n = an.note
        pitch = f"{midi_to_name(n.pitch_midi)}({int(round(n.pitch_midi))})"
        techs = ", ".join(t.value for t in an.techniques)
        print(
            f"  {i:>3}  {n.onset:>7.3f}  {n.offset:>7.3f}  {n.duration:>6.3f}  "
            f"{pitch:<7}  {n.confidence:>5.2f}  {techs}"
        )


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
    parser.add_argument(
        "--plot",
        default=None,
        metavar="PATH",
        help="Save a piano-roll PNG (notes over the guitar stem's CQT) to PATH",
    )
    parser.add_argument(
        "--midi", default=None, metavar="PATH", help="Export the notes as a MIDI file"
    )
    parser.add_argument(
        "--synth",
        default=None,
        metavar="PATH",
        help="Render the notes to a playable WAV (sine synth, native playback)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pipeline = build_default_pipeline()
    output = pipeline.run(args.audio, target_sr=args.sr)

    notes_path = os.path.join(args.out, "notes.json")
    save_json(output.transcription, notes_path)
    if args.stems:
        save_stems(output, args.out)

    print_notes(output.transcription)
    print(f"\nWrote {len(output.transcription.notes)} notes -> {notes_path}")

    if args.midi:
        os.makedirs(os.path.dirname(os.path.abspath(args.midi)), exist_ok=True)
        save_midi(output.transcription, args.midi)
        print(f"Wrote MIDI -> {args.midi}")

    if args.synth:
        os.makedirs(os.path.dirname(os.path.abspath(args.synth)), exist_ok=True)
        synthesize_wav(output.transcription, args.synth)
        print(f"Wrote synthesized audio -> {args.synth}  (play: afplay {args.synth})")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")  # headless: render straight to file
        from gtab.viz.pianoroll import plot_transcription

        os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
        plot_transcription(
            output.transcription, audio=output.stems.guitar, save_path=args.plot
        )
        print(f"Wrote piano-roll plot -> {args.plot}")


if __name__ == "__main__":
    main()
