#!/usr/bin/env python3
"""Measure transcription quality on GuitarSet: note-F1, clean vs Demucs-separated.

Mirrors the SDR harness (`scripts/eval_separation.py`). Loads GuitarSet ground
truth via `mirdata` (JAMS note annotations), runs a `Transcriber` on each clip,
and reports note-F1 at three strictnesses (onset / onset+offset / onset+offset+
pitch). Evaluates TWO inputs so the gap between them is the diagnostic:

  - clean      : the GuitarSet mic recording (upper bound -- the transcriber alone)
  - separated  : Demucs run on that recording (the cost of separation artifacts)

A reference-vs-estimate piano-roll PNG is saved per clip per input.

GuitarSet is large, but this only needs the mic audio + annotations (not the
multi-GB hex-pickup files). `--download` fetches just those parts, or do it once:
`python -c "import mirdata; mirdata.initialize('guitarset').download(partial_download=['annotations','audio_mic'])"`.
The `--separated` pass also needs the separation extra (`pip install -e ".[separation]"`).

Usage:
    pip install -e ".[transcription,viz,eval]"
    python scripts/eval_transcription.py --num-clips 5 --separated
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gtab.config import TRANSCRIBERS  # noqa: E402
from gtab.eval.transcription import note_f1  # noqa: E402
from gtab.stages.ingest import load_audio  # noqa: E402
from gtab.types import AnnotatedNote, NoteEvent, TranscriptionResult  # noqa: E402

STRICTNESSES = ("onset", "onset_offset", "onset_offset_pitch")


def notedata_to_transcription(note_data) -> TranscriptionResult:
    """mirdata NoteData -> TranscriptionResult.

    Respects `note_data.pitch_unit`: GuitarSet stores pitches in MIDI (its JAMS
    uses the note_midi namespace); other datasets may use Hz.
    """
    unit = (note_data.pitch_unit or "hz").lower()
    notes = []
    for (onset, offset), pitch in zip(note_data.intervals, note_data.pitches):
        if unit == "midi":
            pitch_midi = float(pitch)
        elif unit == "hz":
            if pitch <= 0:
                continue
            pitch_midi = 69.0 + 12.0 * math.log2(pitch / 440.0)
        else:
            raise ValueError(f"Unsupported pitch unit: {note_data.pitch_unit!r}")
        notes.append(
            AnnotatedNote(
                note=NoteEvent(onset=float(onset), offset=float(offset), pitch_midi=pitch_midi)
            )
        )
    notes.sort(key=lambda n: n.note.onset)
    return TranscriptionResult(notes=notes)


def load_guitarset(data_home: str | None, download: bool):
    try:
        import mirdata
    except ImportError as e:
        raise ImportError(
            "mirdata is required for GuitarSet eval. Install with: "
            "pip install -e '.[eval]' (or pip install mirdata)"
        ) from e

    ds = mirdata.initialize("guitarset", data_home=data_home)
    if download:
        # Only the mic recordings + note annotations are used here; skip the
        # multi-GB hex-pickup audio.
        ds.download(partial_download=["annotations", "audio_mic"])
    tracks = ds.load_tracks()
    # Fail clearly if the audio isn't on disk yet.
    sample = next(iter(tracks.values()))
    if not os.path.exists(sample.audio_mic_path):
        raise FileNotFoundError(
            f"GuitarSet audio not found under {ds.data_home}. "
            "Download it once with --download."
        )
    return tracks


def mean_f1(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: sum(r[k] for r in rows) / len(rows) for k in STRICTNESSES} if rows else {}


def print_table(label: str, per_clip: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {label} (note-F1) ===")
    print(f"  {'clip':<22} {'onset':>7} {'on+off':>7} {'on+off+pitch':>13}")
    for tid, f1 in per_clip.items():
        print(
            f"  {tid:<22} {f1['onset']:>7.3f} {f1['onset_offset']:>7.3f} "
            f"{f1['onset_offset_pitch']:>13.3f}"
        )
    m = mean_f1(list(per_clip.values()))
    if m:
        print(
            f"  {'MEAN':<22} {m['onset']:>7.3f} {m['onset_offset']:>7.3f} "
            f"{m['onset_offset_pitch']:>13.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate transcriber note-F1 on GuitarSet")
    p.add_argument("--num-clips", type=int, default=5, help="How many clips to evaluate")
    p.add_argument("--impl", default="basic_pitch", choices=sorted(TRANSCRIBERS))
    p.add_argument("--data-home", default=None, help="GuitarSet location (mirdata)")
    p.add_argument("--download", action="store_true", help="Download GuitarSet first")
    p.add_argument(
        "--separated", action="store_true", help="Also eval Demucs-separated guitar"
    )
    p.add_argument("--device", default="auto", help="Demucs device (--separated)")
    p.add_argument("--sr", type=int, default=22050, help="Target sample rate")
    p.add_argument("--out", default="data/interim/transcription_eval", help="PNG dir")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")  # headless: render straight to file
    from gtab.viz.pianoroll import plot_comparison

    tracks = load_guitarset(args.data_home, args.download)
    track_ids = sorted(tracks)[: args.num_clips]
    os.makedirs(args.out, exist_ok=True)

    transcriber = TRANSCRIBERS[args.impl]()
    separator = None
    if args.separated:
        from gtab.stages.separation import DemucsSeparator

        separator = DemucsSeparator(device=args.device)

    clean_rows: dict[str, dict[str, float]] = {}
    separated_rows: dict[str, dict[str, float]] = {}

    for tid in track_ids:
        track = tracks[tid]
        reference = notedata_to_transcription(track.notes_all)

        clean_audio = load_audio(track.audio_mic_path, target_sr=args.sr)
        print(f"[{tid}] transcribing clean ({len(reference.notes)} ref notes) ...")
        est_clean = transcriber.transcribe(clean_audio)
        clean_rows[tid] = note_f1(reference, est_clean)
        plot_comparison(
            reference, est_clean,
            save_path=os.path.join(args.out, f"{tid}_clean.png"),
            title=f"{tid} clean: reference (green) vs {args.impl} (red)",
        )

        if separator is not None:
            print(f"[{tid}] separating + transcribing ...")
            guitar = separator.separate(clean_audio).guitar
            est_sep = transcriber.transcribe(guitar)
            separated_rows[tid] = note_f1(reference, est_sep)
            plot_comparison(
                reference, est_sep,
                save_path=os.path.join(args.out, f"{tid}_separated.png"),
                title=f"{tid} separated: reference (green) vs {args.impl} (red)",
            )

    print_table(f"CLEAN GuitarSet / {args.impl}", clean_rows)
    if separated_rows:
        print_table(f"DEMUCS-SEPARATED / {args.impl}", separated_rows)
        cm, sm = mean_f1(list(clean_rows.values())), mean_f1(list(separated_rows.values()))
        print("\n=== gap (clean - separated) = cost of separation ===")
        for k in STRICTNESSES:
            print(f"  {k:<20} {cm[k] - sm[k]:+.3f}")
    print(f"\nWrote comparison PNGs -> {args.out}")


if __name__ == "__main__":
    main()
