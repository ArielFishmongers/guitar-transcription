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

from gtab.config import TRANSCRIBERS, _build_transcriber  # noqa: E402
from gtab.eval.transcription import note_f1, tab_disambiguation_rate  # noqa: E402
from gtab.stages.ingest import load_audio  # noqa: E402
from gtab.stages.transcription import OPEN_STRING_MIDI  # noqa: E402
from gtab.types import AnnotatedNote, NoteEvent, TranscriptionResult  # noqa: E402

STRICTNESSES = ("onset", "onset_offset", "onset_offset_pitch")

# GuitarSet per-string annotation order == gtab string index (0 = low E ... 5 = high e).
_GSET_STRINGS = ["E", "A", "D", "G", "B", "e"]


def _pitch_to_midi(pitch: float, unit: str) -> float | None:
    """Convert a mirdata pitch to MIDI, honouring the annotation's unit.

    Returns None for non-positive Hz (unvoiced) so callers can skip it.
    """
    if unit == "midi":
        return float(pitch)
    if unit == "hz":
        return 69.0 + 12.0 * math.log2(pitch / 440.0) if pitch > 0 else None
    raise ValueError(f"Unsupported pitch unit: {unit!r}")


def notedata_to_transcription(note_data) -> TranscriptionResult:
    """mirdata NoteData -> TranscriptionResult (pitch only, no string/fret).

    Respects `note_data.pitch_unit`: GuitarSet stores pitches in MIDI (its JAMS
    uses the note_midi namespace); other datasets may use Hz.
    """
    unit = (note_data.pitch_unit or "hz").lower()
    notes = []
    for (onset, offset), pitch in zip(note_data.intervals, note_data.pitches):
        midi = _pitch_to_midi(pitch, unit)
        if midi is None:
            continue
        notes.append(
            AnnotatedNote(
                note=NoteEvent(onset=float(onset), offset=float(offset), pitch_midi=midi)
            )
        )
    notes.sort(key=lambda n: n.note.onset)
    return TranscriptionResult(notes=notes)


def guitarset_reference_with_strings(track) -> TranscriptionResult:
    """GuitarSet per-string ground truth -> TranscriptionResult WITH string/fret.

    `track.notes` is a dict of 6 per-string NoteData keyed by string name; the
    name's position in `_GSET_STRINGS` is the gtab string index (0 = low E).
    Fret = round(pitch) - open-string MIDI. A reversed string convention would
    produce large negative frets, so we guard on the plausible range.
    """
    notes = []
    for s_idx, name in enumerate(_GSET_STRINGS):
        nd = track.notes.get(name)
        if nd is None:
            continue
        unit = (nd.pitch_unit or "hz").lower()
        for (onset, offset), pitch in zip(nd.intervals, nd.pitches):
            midi = _pitch_to_midi(pitch, unit)
            if midi is None:
                continue
            fret = int(round(midi)) - OPEN_STRING_MIDI[s_idx]
            if not (0 <= fret <= 24):
                raise ValueError(
                    f"GuitarSet string-convention check failed: note {midi:.1f} on "
                    f"string {s_idx} ({name!r}) -> fret {fret} (expected 0-24). The "
                    "per-string annotation order may not match gtab's low-E-first index."
                )
            notes.append(
                AnnotatedNote(
                    note=NoteEvent(
                        onset=float(onset),
                        offset=float(offset),
                        pitch_midi=midi,
                        string=s_idx,
                        fret=fret,
                    )
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


def print_tdr_table(label: str, per_clip: dict[str, dict[str, float]]) -> None:
    """TDR per clip + a coverage-weighted mean (of correctly-detected pitches,
    the fraction assigned the right string)."""
    print(f"\n=== {label} ===")
    print(f"  {'clip':<22} {'TDR':>6} {'matched':>8} {'w/string':>9}")
    tot_correct = tot_with = 0
    for tid, r in per_clip.items():
        with_s = int(r["n_with_string"])
        tot_with += with_s
        tot_correct += round(r["tdr"] * with_s)
        print(
            f"  {tid:<22} {r['tdr']:>6.3f} {int(r['n_matched']):>8} {with_s:>9}"
        )
    micro = tot_correct / tot_with if tot_with else 0.0
    print(f"  {'MEAN (micro)':<22} {micro:>6.3f} {'':>8} {tot_with:>9}")
    print("  TDR = of correctly-detected pitches, fraction on the right string.")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate transcriber note-F1 on GuitarSet")
    p.add_argument("--num-clips", type=int, default=5, help="How many clips to evaluate")
    p.add_argument("--impl", default="basic_pitch", choices=sorted(TRANSCRIBERS))
    p.add_argument("--checkpoint", default=None, help="FretNet .pt (for fretnet/fusion)")
    p.add_argument(
        "--tdr",
        action="store_true",
        help="Also report tablature disambiguation rate (needs a string-predicting "
        "impl: fretnet or fusion)",
    )
    p.add_argument(
        "--player",
        type=int,
        default=None,
        help="Evaluate only this GuitarSet player (e.g. 0). For a fold-N checkpoint, "
        "player N was held out -- the honest generalization split.",
    )
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

    if args.tdr and args.impl not in ("fretnet", "fusion"):
        p.error("--tdr needs a string-predicting impl: --impl fretnet or fusion")
    if args.impl in ("fretnet", "fusion") and not args.checkpoint:
        p.error(f"--impl {args.impl} requires --checkpoint (the trained FretNet .pt)")

    tracks = load_guitarset(args.data_home, args.download)
    ids = sorted(tracks)
    if args.player is not None:
        ids = [t for t in ids if t.startswith(f"{args.player:02d}_")]
    track_ids = ids[: args.num_clips]
    os.makedirs(args.out, exist_ok=True)

    transcriber = _build_transcriber(
        {
            "transcription": {
                "impl": args.impl,
                "checkpoint": args.checkpoint,
                "onset_threshold": 0.5,
                "frame_threshold": 0.3,
            }
        }
    )
    separator = None
    if args.separated:
        from gtab.stages.separation import DemucsSeparator

        separator = DemucsSeparator(device=args.device)

    clean_rows: dict[str, dict[str, float]] = {}
    separated_rows: dict[str, dict[str, float]] = {}
    tdr_rows: dict[str, dict[str, float]] = {}

    for tid in track_ids:
        track = tracks[tid]
        reference = notedata_to_transcription(track.notes_all)

        clean_audio = load_audio(track.audio_mic_path, target_sr=args.sr)
        print(f"[{tid}] transcribing clean ({len(reference.notes)} ref notes) ...")
        est_clean = transcriber.transcribe(clean_audio)
        clean_rows[tid] = note_f1(reference, est_clean)
        if args.tdr:
            ref_strings = guitarset_reference_with_strings(track)
            tdr_rows[tid] = tab_disambiguation_rate(ref_strings, est_clean)
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
    if tdr_rows:
        print_tdr_table(f"TDR / {args.impl}", tdr_rows)
    print(f"\nWrote comparison PNGs -> {args.out}")


if __name__ == "__main__":
    main()
