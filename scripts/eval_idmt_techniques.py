#!/usr/bin/env python3
"""Validate Stage-4 technique detection (vibrato + bend) on IDMT-SMT-GUITAR.

IDMT dataset2 carries per-note expressionStyle labels (BE/VI/SL/HA/...). We
transcribe each electric clip, run the technique detector, and report note-level
precision/recall/F1 per technique against the IDMT labels -- the build->validate
loop used for TDR. Technique detection estimates its own per-note F0 (pyin) from
the audio, so a lightweight transcriber (basic_pitch) that just supplies the
notes is enough; no FretNet subprocess needed.

    python scripts/eval_idmt_techniques.py --idmt-root data/raw/idmt/IDMT-SMT-GUITAR_V2 \
        --num-clips 80
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_idmt_tdr import find_pairs, idmt_reference  # noqa: E402

from gtab.config import _build_technique, _build_transcriber  # noqa: E402
from gtab.eval.transcription import technique_prf  # noqa: E402
from gtab.stages.ingest import load_audio  # noqa: E402
from gtab.types import Technique  # noqa: E402

TECHNIQUES = [Technique.VIBRATO, Technique.BEND, Technique.SLIDE,
              Technique.HARMONIC, Technique.PALM_MUTE, Technique.DEAD_NOTE]


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-4 technique P/R/F1 on IDMT-SMT-GUITAR")
    p.add_argument("--idmt-root", required=True, help="IDMT-SMT-GUITAR_V2 root dir")
    p.add_argument("--subset", default="dataset2")
    p.add_argument("--impl", default="basic_pitch", help="transcriber impl (notes source)")
    p.add_argument("--checkpoint", default=None, help="FretNet .pt (fusion/fretnet only)")
    p.add_argument("--technique", default="contour", help="technique detector: contour | learned")
    p.add_argument("--model-path", default=None, help="learned technique model (.joblib)")
    p.add_argument("--num-clips", type=int, default=80)
    p.add_argument("--sr", type=int, default=22050)
    args = p.parse_args()

    transcriber = _build_transcriber({
        "transcription": {"impl": args.impl, "checkpoint": args.checkpoint,
                          "track_beats": False},
    })
    detector = _build_technique({
        "techniques": {"impl": args.technique, "model_path": args.model_path,
                       "checkpoint": args.checkpoint},
    })
    pairs = find_pairs(args.idmt_root, args.subset)[: args.num_clips]
    print(f"technique eval: {args.impl} + {args.technique} detector on {len(pairs)} "
          f"{args.subset} clips (electric IDMT)")

    agg = {t: {"tp": 0, "n_ref": 0, "n_est": 0} for t in TECHNIQUES}
    for i, (xml, wav) in enumerate(pairs):
        ref = idmt_reference(xml, with_techniques=True)
        if not ref.notes:
            continue
        audio = load_audio(wav, target_sr=args.sr)
        est = detector.detect(audio, transcriber.transcribe(audio))
        for t in TECHNIQUES:
            r = technique_prf(ref, est, t)
            agg[t]["tp"] += r["tp"]
            agg[t]["n_ref"] += r["n_ref"]
            agg[t]["n_est"] += r["n_est"]
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)} clips", flush=True)

    print(f"\n=== technique P/R/F1 (micro) / {args.impl} / IDMT {args.subset} ===")
    print(f"  {'technique':<10} {'P':>6} {'R':>6} {'F1':>6}   {'tp':>4} {'ref':>4} {'est':>4}")
    for t in TECHNIQUES:
        a = agg[t]
        prec = a["tp"] / a["n_est"] if a["n_est"] else 0.0
        rec = a["tp"] / a["n_ref"] if a["n_ref"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {t.value:<10} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}   "
              f"{a['tp']:>4} {a['n_ref']:>4} {a['n_est']:>4}")


if __name__ == "__main__":
    main()
