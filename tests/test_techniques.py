"""Stage 4 technique-detection tests.

`detect_contour_techniques` is pure numpy and always runs. The detector
integration test synthesizes vibrato/bend tones and runs the real pyin path
(librosa is a core dep). `technique_prf` needs mir_eval (skipped if absent).
"""
import importlib.util

import numpy as np
import pytest

from gtab.stages.techniques import ContourTechniqueDetector, detect_contour_techniques
from gtab.types import AnnotatedNote, AudioBuffer, NoteEvent, Technique, TranscriptionResult

_FRAME_HOP = 256 / 22050  # matches the detector's pyin hop


def _t(n):
    return np.arange(n) * _FRAME_HOP


# --------------------------------------------------------------------------- #
# detect_contour_techniques -- pure numpy
# --------------------------------------------------------------------------- #

def test_detects_vibrato():
    t = np.arange(0, 0.6, _FRAME_HOP)
    f0 = 60 + 0.4 * np.sin(2 * np.pi * 6 * t)  # 6 Hz, ±0.4 semitone
    assert detect_contour_techniques(f0, t) == [Technique.VIBRATO]


def test_detects_bend():
    t = np.arange(0, 0.3, _FRAME_HOP)
    f0 = 60 + 2.0 * (t / t[-1])  # smooth +2 semitone glide
    assert detect_contour_techniques(f0, t) == [Technique.BEND]


def test_flat_contour_is_no_technique():
    t = np.arange(0, 0.4, _FRAME_HOP)
    rng = np.random.default_rng(0)
    f0 = 60 + 0.05 * rng.standard_normal(t.size)
    assert detect_contour_techniques(f0, t) == []


def test_sub_semitone_drift_is_not_a_bend():
    t = np.arange(0, 0.3, _FRAME_HOP)
    f0 = 60 + 0.5 * (t / t[-1])  # 0.5 semitone -> below the >=1 gate
    assert detect_contour_techniques(f0, t) == []


def test_too_few_frames_is_empty():
    assert detect_contour_techniques(np.array([60.0, 60.5]), _t(2)) == []


# --------------------------------------------------------------------------- #
# ContourTechniqueDetector -- real pyin path on synthesized tones (librosa core)
# --------------------------------------------------------------------------- #

def _tone(midi_fn, dur=0.6, sr=22050):
    """Synthesize a tone whose instantaneous pitch is midi_fn(t) (MIDI)."""
    t = np.arange(int(sr * dur)) / sr
    f0 = 440.0 * 2.0 ** ((midi_fn(t) - 69) / 12.0)
    y = 0.5 * np.sin(2 * np.pi * np.cumsum(f0) / sr)
    return AudioBuffer(samples=y.astype(np.float32), sample_rate=sr), t[-1]


def _one_note_result(offset, pitch_midi=57.0):
    return TranscriptionResult(
        notes=[AnnotatedNote(note=NoteEvent(onset=0.0, offset=offset, pitch_midi=pitch_midi))]
    )


def test_detector_tags_vibrato_tone():
    audio, end = _tone(lambda t: 57.0 + 0.4 * np.sin(2 * np.pi * 6 * t))
    out = ContourTechniqueDetector().detect(audio, _one_note_result(end))
    assert Technique.VIBRATO in out.notes[0].techniques


def test_detector_tags_bend_tone():
    audio, end = _tone(lambda t: 57.0 + 2.0 * (t / t[-1]), dur=0.5)
    out = ContourTechniqueDetector().detect(audio, _one_note_result(end))
    assert Technique.BEND in out.notes[0].techniques


def test_detector_preserves_notes_and_metadata():
    audio, end = _tone(lambda t: 57.0 + 0.0 * t)  # steady tone -> no technique
    res = _one_note_result(end)
    res.tempo_bpm, res.beats = 120.0, [0.0, 0.5]
    out = ContourTechniqueDetector().detect(audio, res)
    assert len(out.notes) == 1 and out.tempo_bpm == 120.0 and out.beats == [0.0, 0.5]
    assert out.notes[0].techniques == []  # steady tone: nothing tagged


# --------------------------------------------------------------------------- #
# technique_prf metric (needs mir_eval)
# --------------------------------------------------------------------------- #

_mir_eval = importlib.util.find_spec("mir_eval") is not None


def _note(onset, pitch, techs):
    return AnnotatedNote(
        note=NoteEvent(onset=onset, offset=onset + 0.2, pitch_midi=pitch),
        techniques=list(techs),
    )


@pytest.mark.skipif(not _mir_eval, reason="mir_eval not installed")
def test_technique_prf_perfect_match():
    from gtab.eval.transcription import technique_prf

    ref = TranscriptionResult(notes=[_note(0.0, 60, [Technique.BEND]), _note(0.5, 64, [])])
    r = technique_prf(ref, ref, Technique.BEND)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0
    assert r["tp"] == 1 and r["n_ref"] == 1 and r["n_est"] == 1


@pytest.mark.skipif(not _mir_eval, reason="mir_eval not installed")
def test_technique_prf_missed_detection_drops_recall():
    from gtab.eval.transcription import technique_prf

    ref = TranscriptionResult(notes=[_note(0.0, 60, [Technique.BEND])])
    est = TranscriptionResult(notes=[_note(0.0, 60, [])])  # matched note, technique missed
    r = technique_prf(ref, est, Technique.BEND)
    assert r["recall"] == 0.0 and r["tp"] == 0 and r["n_ref"] == 1
