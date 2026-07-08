#!/usr/bin/env python3
"""Out-of-domain (ELECTRIC guitar) TDR on IDMT-SMT-GUITAR.

GuitarSet (our in-domain eval) is acoustic; this measures whether the fused /
FretNet string predictions hold up on real ELECTRIC guitar. IDMT-SMT-GUITAR's
per-note XML carries `stringNumber` (1-indexed, low-E-first) + `fretNumber`, so
it is a ready-made tablature ground truth. We use dataset2 (electric phrases,
standard tuning EADGBE) by default.

    python scripts/eval_idmt_tdr.py --impl fusion --checkpoint <fretnet.pt> \
        --idmt-root data/raw/idmt/IDMT-SMT-GUITAR_V2 --num-clips 40

Reports note-F1 (3 strictnesses) + micro-TDR with coverage, mirroring
scripts/eval_transcription.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gtab.config import _build_transcriber  # noqa: E402
from gtab.eval.transcription import note_f1, tab_disambiguation_rate  # noqa: E402
from gtab.stages.ingest import load_audio  # noqa: E402
from gtab.stages.transcription import OPEN_STRING_MIDI  # noqa: E402
from gtab.types import AnnotatedNote, NoteEvent, TranscriptionResult  # noqa: E402


def idmt_reference(xml_path: str, max_fret: int = 19) -> TranscriptionResult:
    """IDMT-SMT-GUITAR annotation XML -> TranscriptionResult with string/fret.

    `stringNumber` is 1-indexed low-E-first -> gtab index = stringNumber - 1.
    Notes above `max_fret` (beyond FretNet's representable range) and notes whose
    pitch is inconsistent with open+fret (bends/label noise) are dropped so the
    number reflects string disambiguation, not an architectural fret-range cap.
    """
    root = ET.parse(xml_path).getroot()
    notes: list[AnnotatedNote] = []
    dropped = 0
    for ev in root.iter("event"):
        def txt(tag):
            e = ev.find(tag)
            return e.text if e is not None else None

        pitch, onset, offset = txt("pitch"), txt("onsetSec"), txt("offsetSec")
        s, fr = txt("stringNumber"), txt("fretNumber")
        if None in (pitch, onset, offset, s, fr):
            continue
        s_idx, fret, midi = int(s) - 1, int(fr), float(pitch)
        if not (0 <= s_idx <= 5) or not (0 <= fret <= max_fret):
            continue
        if abs(round(midi) - (OPEN_STRING_MIDI[s_idx] + fret)) > 1:
            dropped += 1  # convention mismatch / heavy bend -> skip
            continue
        notes.append(
            AnnotatedNote(note=NoteEvent(
                onset=float(onset), offset=float(offset), pitch_midi=midi,
                string=s_idx, fret=fret))
        )
    if notes and dropped > len(notes):
        raise ValueError(
            f"{xml_path}: more notes dropped ({dropped}) than kept -- check the "
            "stringNumber convention / tuning."
        )
    notes.sort(key=lambda n: n.note.onset)
    return TranscriptionResult(notes=notes)


def find_pairs(idmt_root: str, subset: str) -> list[tuple[str, str]]:
    """(xml, wav) pairs matched by basename within a subset."""
    ann = os.path.join(idmt_root, subset, "annotation")
    aud = os.path.join(idmt_root, subset, "audio")
    pairs = []
    for xml in sorted(glob.glob(os.path.join(ann, "*.xml"))):
        wav = os.path.join(aud, os.path.splitext(os.path.basename(xml))[0] + ".wav")
        if os.path.exists(wav):
            pairs.append((xml, wav))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Electric-guitar TDR on IDMT-SMT-GUITAR")
    p.add_argument("--idmt-root", required=True, help="IDMT-SMT-GUITAR_V2 root dir")
    p.add_argument("--subset", default="dataset2", help="dataset1|dataset2|dataset3")
    p.add_argument("--impl", default="fusion", choices=("fusion", "fretnet"))
    p.add_argument("--checkpoint", required=True, help="FretNet .pt")
    p.add_argument("--num-clips", type=int, default=40)
    p.add_argument("--max-fret", type=int, default=19)
    p.add_argument("--sr", type=int, default=22050)
    args = p.parse_args()

    transcriber = _build_transcriber(
        {"transcription": {"impl": args.impl, "checkpoint": args.checkpoint}}
    )
    pairs = find_pairs(args.idmt_root, args.subset)[: args.num_clips]
    print(f"{args.impl} on {len(pairs)} {args.subset} clips (electric, IDMT-SMT-GUITAR)")

    f1_keys = ("onset", "onset_offset", "onset_offset_pitch")
    f1_sum = {k: 0.0 for k in f1_keys}
    f1_n = 0
    tot_correct = tot_with = tot_matched = 0

    for xml, wav in pairs:
        ref = idmt_reference(xml, max_fret=args.max_fret)
        if not ref.notes:
            continue
        audio = load_audio(wav, target_sr=args.sr)
        est = transcriber.transcribe(audio)
        f1 = note_f1(ref, est)
        for k in f1_keys:
            f1_sum[k] += f1[k]
        f1_n += 1
        r = tab_disambiguation_rate(ref, est)
        tot_correct += round(r["tdr"] * r["n_with_string"])
        tot_with += r["n_with_string"]
        tot_matched += r["n_matched"]
        print(f"  {os.path.basename(xml):<32} refN={len(ref.notes):>3} "
              f"matched={r['n_matched']:>3} TDR={r['tdr']:.3f}", flush=True)

    print(f"\n=== {args.impl} / IDMT {args.subset} (electric, frets 0-{args.max_fret}) ===")
    if f1_n:
        print("note-F1  " + "  ".join(f"{k}={f1_sum[k]/f1_n:.3f}" for k in f1_keys))
    tdr = tot_correct / tot_with if tot_with else 0.0
    cov = tot_with / tot_matched if tot_matched else 0.0
    print(f"TDR (micro) = {tdr:.3f}   coverage = {cov:.3f}   "
          f"matched={tot_matched}  with_string={tot_with}")


if __name__ == "__main__":
    main()
