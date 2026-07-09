"""Transcription quality metric: note-level precision/recall/F1.

Mirrors the SDR harness in `eval/separation.py`. Uses
`mir_eval.transcription` to score an estimated `TranscriptionResult` against a
reference one at three strictnesses (loosest to strictest):

    onset                -- onset within +/-50 ms, pitch & offset ignored
    onset_offset         -- onset + offset, pitch ignored
    onset_offset_pitch   -- onset + offset + pitch (within 50 cents)

Offset/pitch strictnesses score lower than onset-only by construction; reporting
all three shows *which* part of a transcription is failing.

NOTE: full Stage-3 eval (GuitarSet ground truth via mirdata) is still TODO; this
module is the metric the viz comparison plot visualises. See docs/SPEC_transcription.md.
"""
from __future__ import annotations

import numpy as np

from gtab.types import TranscriptionResult

ONSET_TOLERANCE = 0.05  # seconds, the mir_eval / MIREX default
_IGNORE_PITCH_CENTS = 1.0e9  # huge pitch tolerance == "ignore pitch"


def _to_intervals_pitches(result: TranscriptionResult) -> tuple[np.ndarray, np.ndarray]:
    """TranscriptionResult -> (intervals[N,2] in seconds, pitches[N] in Hz)."""
    if not result.notes:
        return np.zeros((0, 2)), np.zeros((0,))
    intervals = np.array(
        [[n.note.onset, max(n.note.offset, n.note.onset + 1e-3)] for n in result.notes]
    )
    pitches_hz = np.array(
        [440.0 * 2.0 ** ((n.note.pitch_midi - 69) / 12.0) for n in result.notes]
    )
    return intervals, pitches_hz


def note_f1(
    ref: TranscriptionResult,
    est: TranscriptionResult,
    onset_tolerance: float = ONSET_TOLERANCE,
) -> dict[str, float]:
    """Note-level F1 at three strictnesses (see module docstring)."""
    import mir_eval

    ref_i, ref_p = _to_intervals_pitches(ref)
    est_i, est_p = _to_intervals_pitches(est)

    # Both empty -> a vacuously perfect match; one empty -> nothing matches.
    if len(ref_i) == 0 and len(est_i) == 0:
        return {k: 1.0 for k in ("onset", "onset_offset", "onset_offset_pitch")}
    if len(ref_i) == 0 or len(est_i) == 0:
        return {k: 0.0 for k in ("onset", "onset_offset", "onset_offset_pitch")}

    _, _, onset_f1 = mir_eval.transcription.onset_precision_recall_f1(
        ref_i, est_i, onset_tolerance=onset_tolerance
    )
    _, _, onset_offset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=_IGNORE_PITCH_CENTS,  # ignore pitch, keep onset+offset
        offset_ratio=0.2,
    )
    _, _, full_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p,
        onset_tolerance=onset_tolerance,
        offset_ratio=0.2,  # default pitch_tolerance=50 cents
    )
    return {
        "onset": float(onset_f1),
        "onset_offset": float(onset_offset_f1),
        "onset_offset_pitch": float(full_f1),
    }


def tab_disambiguation_rate(
    ref: TranscriptionResult,
    est: TranscriptionResult,
    onset_tolerance: float = ONSET_TOLERANCE,
    pitch_tolerance: float = 50.0,
) -> dict[str, float]:
    """Note-level tablature disambiguation rate (TDR).

    Of the correctly-detected pitches (est notes matched to ref notes on onset +
    pitch), the fraction assigned to the CORRECT string. This is the metric that
    decides whether string output is trustworthy downstream.

    Both `ref` and `est` must carry `string`. Est notes with `string is None`
    (withheld/unassigned) are excluded from the rate but surfaced via coverage
    counts. When every matched est note carries a string, `tdr` equals the
    standard FretNet/Wiggins TDR (num_correct_tablature / num_correct_pitch).

    Returns {tdr, n_matched, n_with_string, n_ref, n_est}.
    """
    import mir_eval

    ref_i, ref_p = _to_intervals_pitches(ref)
    est_i, est_p = _to_intervals_pitches(est)
    n_ref, n_est = len(ref_i), len(est_i)
    if n_ref == 0 or n_est == 0:
        return {"tdr": 0.0, "n_matched": 0, "n_with_string": 0, "n_ref": n_ref, "n_est": n_est}

    ref_strings = [n.note.string for n in ref.notes]
    est_strings = [n.note.string for n in est.notes]

    # onset + pitch match (offset_ratio=None) == the "correctly-detected pitches".
    matches = mir_eval.transcription.match_notes(
        ref_i, ref_p, est_i, est_p,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=None,
    )
    with_string = [
        (ri, ei)
        for ri, ei in matches
        if est_strings[ei] is not None and ref_strings[ri] is not None
    ]
    correct = sum(1 for ri, ei in with_string if est_strings[ei] == ref_strings[ri])
    tdr = correct / len(with_string) if with_string else 0.0
    return {
        "tdr": float(tdr),
        "n_matched": int(len(matches)),
        "n_with_string": int(len(with_string)),
        "n_ref": int(n_ref),
        "n_est": int(n_est),
    }


def technique_prf(
    ref: TranscriptionResult,
    est: TranscriptionResult,
    technique,
    onset_tolerance: float = ONSET_TOLERANCE,
    pitch_tolerance: float = 50.0,
) -> dict[str, float]:
    """Note-level precision/recall/F1 for a single `Technique`.

    A true positive is an est note tagged with the technique that matches (onset +
    pitch) a ref note also tagged with it. Precision is over all est notes tagged
    with the technique, recall over all ref notes tagged with it. Both `ref` and
    `est` must carry techniques on their `AnnotatedNote`s.

    Returns {precision, recall, f1, tp, n_ref, n_est, n_matched}.
    """
    import mir_eval

    ref_i, ref_p = _to_intervals_pitches(ref)
    est_i, est_p = _to_intervals_pitches(est)
    ref_has = [technique in an.techniques for an in ref.notes]
    est_has = [technique in an.techniques for an in est.notes]
    n_ref, n_est = int(sum(ref_has)), int(sum(est_has))

    if len(ref_i) == 0 or len(est_i) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0,
                "n_ref": n_ref, "n_est": n_est, "n_matched": 0}

    matches = mir_eval.transcription.match_notes(
        ref_i, ref_p, est_i, est_p,
        onset_tolerance=onset_tolerance, pitch_tolerance=pitch_tolerance,
        offset_ratio=None,
    )
    tp = sum(1 for ri, ei in matches if ref_has[ri] and est_has[ei])
    precision = tp / n_est if n_est else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": int(tp), "n_ref": n_ref, "n_est": n_est, "n_matched": int(len(matches))}
