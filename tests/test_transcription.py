"""Stage 3 transcription tests.

The `assign_string_fret` unit tests and the fusion plumbing test are numpy+pytest
only -- they exercise the fusion string-lookup and wiring with a fake FretNet
client, so they always run. The heavy integration tests (real FretNet subprocess
in the `fretnet-repro` env on a GuitarSet clip) are skipped unless that env, a
checkpoint, and a clip are all present.
"""
import importlib.util
import os

import numpy as np
import pytest

from gtab.stages.fretnet_client import FretNetClient, FretNetPrediction
from gtab.stages.transcription import (
    OPEN_STRING_MIDI,
    FusionTranscriber,
    _BP_FRAME_HOP_S,
    assign_string_fret,
    pitch_bends_to_contour,
)
from gtab.types import AnnotatedNote, AudioBuffer, NoteEvent, TranscriptionResult


# --------------------------------------------------------------------------- #
# assign_string_fret -- pure numpy, no heavy deps
# --------------------------------------------------------------------------- #

def _cube(bin_idx: int, per_string: dict[int, float], n_frames: int = 3) -> np.ndarray:
    """Activation cube (6, 44, n_frames) with `per_string` values at `bin_idx`."""
    mp = np.zeros((6, 44, n_frames), dtype=np.float32)
    for s, v in per_string.items():
        mp[s, bin_idx, :] = v
    return mp


def test_assign_picks_max_activation_valid_string():
    # A2 (45) is playable on string0/fret5 and string1/fret0. String1 is hottest.
    times = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    mp = _cube(45 - 40, {0: 0.2, 1: 0.9})
    assert assign_string_fret(45, 0.0, 0.2, mp, times) == (1, 0)


def test_assign_pitch_below_range_returns_none():
    times = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    mp = _cube(0, {0: 0.9})
    assert assign_string_fret(30, 0.0, 0.2, mp, times) == (None, None)


def test_assign_pitch_above_range_returns_none():
    times = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    mp = _cube(0, {0: 0.9})
    assert assign_string_fret(90, 0.0, 0.2, mp, times) == (None, None)


def test_assign_physically_invalid_string_excluded():
    # Low E (40) is only playable on string0 (fret0). Even though string5 is the
    # hottest, it cannot produce 40, so string0 must be chosen.
    times = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    mp = _cube(0, {5: 0.9, 0: 0.1})
    assert assign_string_fret(40, 0.0, 0.2, mp, times) == (0, 0)


def test_assign_no_valid_string_returns_none():
    # With num_frets=0 only open-string pitches are reachable; 41 is not one.
    times = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    mp = _cube(41 - 40, {0: 0.9, 1: 0.9})
    assert assign_string_fret(41, 0.0, 0.2, mp, times, num_frets=0) == (None, None)


def test_assign_gate_withholds_low_confidence():
    times = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    mp = _cube(45 - 40, {1: 0.3})
    assert assign_string_fret(45, 0.0, 0.2, mp, times, gate=0.5) == (None, None)
    # Below the gate threshold it is withheld; at/above it is emitted.
    assert assign_string_fret(45, 0.0, 0.2, mp, times, gate=0.2) == (1, 0)


def test_assign_frame_window_and_nearest_fallback():
    # frame0: string0 hot; frame2: string1 hot; frame1: nothing.
    times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    mp = np.zeros((6, 44, 3), dtype=np.float32)
    mp[0, 45 - 40, 0] = 0.9  # frame 0 -> string0
    mp[1, 45 - 40, 2] = 0.9  # frame 2 -> string1
    # window catches only frame 0
    assert assign_string_fret(45, -0.1, 0.2, mp, times) == (0, 5)
    # window catches only frame 2
    assert assign_string_fret(45, 0.9, 1.1, mp, times) == (1, 0)
    # window past the last frame -> nearest-frame fallback picks frame 2
    assert assign_string_fret(45, 5.0, 5.1, mp, times) == (1, 0)


# --------------------------------------------------------------------------- #
# Pitch-contour helper + beat tracking (numpy/librosa only -- librosa is core)
# --------------------------------------------------------------------------- #

def test_pitch_bends_to_contour_folds_bins_into_midi():
    # 3 contour bins == 1 semitone: 0->+0, 3->+1, -3->-1.
    contour = pitch_bends_to_contour(1.0, 52.0, [0, 3, -3])
    assert contour is not None and len(contour) == 3
    pitches = [p for _, p in contour]
    times = [t for t, _ in contour]
    assert pitches == [52.0, 53.0, 51.0]
    assert abs(times[0] - 1.0) < 1e-9
    assert abs(times[1] - (1.0 + _BP_FRAME_HOP_S)) < 1e-9


def test_pitch_bends_to_contour_empty_is_none():
    assert pitch_bends_to_contour(0.0, 40.0, None) is None
    assert pitch_bends_to_contour(0.0, 40.0, []) is None


def test_estimate_beats_on_click_track():
    import librosa

    from gtab.rhythm import estimate_beats

    sr, bpm, dur = 22050, 120.0, 8.0
    period = 60.0 / bpm
    y = librosa.clicks(
        times=np.arange(0.0, dur, period), sr=sr, length=int(sr * dur)
    ).astype(np.float32)
    tempo, beats = estimate_beats(AudioBuffer(samples=y, sample_rate=sr))
    assert tempo is not None and tempo > 0
    assert len(beats) > 3 and beats == sorted(beats)


def test_estimate_beats_short_audio_returns_empty():
    from gtab.rhythm import estimate_beats

    y = np.zeros(1000, dtype=np.float32)  # < 0.5 s -> not enough for a tempo
    tempo, beats = estimate_beats(AudioBuffer(samples=y, sample_rate=22050))
    assert tempo is None and beats == []


# --------------------------------------------------------------------------- #
# FusionTranscriber plumbing -- fake client + fake Basic Pitch, no subprocess
# --------------------------------------------------------------------------- #

class _FakeBP:
    """Stand-in for BasicPitchTranscriber returning fixed notes."""

    def __init__(self, notes):
        self._result = TranscriptionResult(notes=notes)

    def transcribe(self, guitar):
        return self._result


class _FakeClient:
    def __init__(self, pred):
        self._pred = pred

    def run(self, guitar):
        return self._pred


def test_fusion_attaches_string_fret_and_preserves_notes():
    # Two Basic Pitch notes: A2 (45) and E3 (52).
    bp_notes = [
        AnnotatedNote(note=NoteEvent(onset=0.0, offset=0.2, pitch_midi=45.0, confidence=0.8)),
        AnnotatedNote(note=NoteEvent(onset=0.3, offset=0.5, pitch_midi=52.0, confidence=0.6)),
    ]
    mp = np.zeros((6, 44, 3), dtype=np.float32)
    mp[1, 45 - 40, :] = 0.9  # 45 -> string1 fret0
    mp[2, 52 - 40, :] = 0.9  # 52 -> string2 fret2
    pred = FretNetPrediction(
        sr=22050, hop=512, open_string_midi=list(OPEN_STRING_MIDI), profile_low=40,
        num_pitch_bins=44, notes=[],
        multi_pitch=mp, times=np.array([0.0, 0.25, 0.5], dtype=np.float32),
    )
    fus = FusionTranscriber(client=_FakeClient(pred))
    fus.bp = _FakeBP(bp_notes)

    result = fus.transcribe(AudioBuffer(samples=np.zeros(10, np.float32), sample_rate=22050))

    assert len(result.notes) == 2
    n0, n1 = result.notes[0].note, result.notes[1].note
    # Basic Pitch's pitch/onset/confidence preserved verbatim
    assert (n0.pitch_midi, n0.confidence) == (45.0, 0.8)
    # FretNet string/fret attached, consistent with pitch
    assert (n0.string, n0.fret) == (1, 0)
    assert (n1.string, n1.fret) == (2, 2)
    assert n1.pitch_midi == 52.0


def test_fusion_empty_notes_skips_fretnet():
    # No Basic Pitch notes -> return early without ever calling the client.
    class _BoomClient:
        def run(self, guitar):
            raise AssertionError("client should not run when there are no notes")

    fus = FusionTranscriber(client=_BoomClient())
    fus.bp = _FakeBP([])
    out = fus.transcribe(AudioBuffer(samples=np.zeros(10, np.float32), sample_rate=22050))
    assert out.notes == []


# --------------------------------------------------------------------------- #
# Note-level TDR metric (needs mir_eval)
# --------------------------------------------------------------------------- #

_mir_eval = importlib.util.find_spec("mir_eval") is not None


def _note(onset, pitch, string):
    return AnnotatedNote(
        note=NoteEvent(onset=onset, offset=onset + 0.2, pitch_midi=pitch, string=string)
    )


@pytest.mark.skipif(not _mir_eval, reason="mir_eval not installed")
def test_tdr_perfect_match_is_one():
    from gtab.eval.transcription import tab_disambiguation_rate

    ref = TranscriptionResult(notes=[_note(0.0, 45, 1), _note(0.5, 52, 2)])
    r = tab_disambiguation_rate(ref, ref)
    assert r["tdr"] == 1.0
    assert r["n_matched"] == 2 and r["n_with_string"] == 2


@pytest.mark.skipif(not _mir_eval, reason="mir_eval not installed")
def test_tdr_wrong_string_drops_below_one():
    from gtab.eval.transcription import tab_disambiguation_rate

    ref = TranscriptionResult(notes=[_note(0.0, 45, 1), _note(0.5, 52, 2)])
    # Same pitches/onsets (so both match), but the second note's string is wrong.
    est = TranscriptionResult(notes=[_note(0.0, 45, 1), _note(0.5, 52, 3)])
    r = tab_disambiguation_rate(ref, est)
    assert r["n_matched"] == 2 and r["n_with_string"] == 2
    assert r["tdr"] == 0.5


# --------------------------------------------------------------------------- #
# Integration -- real FretNet subprocess (skipped without the env/checkpoint)
# --------------------------------------------------------------------------- #

_CKPT = (
    "/Users/ariel/Documents/Personal Projects/fretnet-repro/"
    "guitar-transcription-continuous/generated/experiments/"
    "FretNet_GuitarSetPlus_HCQT_X/models/fold-0/model-2500.pt"
)
_FRETNET_PY = "/opt/miniconda3/envs/fretnet-repro/bin/python"
_GSET_CLIP = (
    "/Users/ariel/Documents/Personal Projects/fretnet-repro/"
    "mir_datasets/guitarset/audio_mono-mic/00_BN1-129-Eb_comp_mic.wav"
)
_fretnet_ready = all(os.path.exists(p) for p in (_CKPT, _FRETNET_PY, _GSET_CLIP))


def _load_clip(seconds: float = 5.0) -> AudioBuffer:
    import soundfile as sf

    y, sr = sf.read(_GSET_CLIP)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return AudioBuffer(samples=y[: int(sr * seconds)].astype(np.float32), sample_rate=sr)


@pytest.mark.skipif(not _fretnet_ready, reason="fretnet-repro env / checkpoint / clip not available")
def test_fretnet_client_roundtrip():
    pred = FretNetClient(checkpoint=_CKPT).run(_load_clip())
    assert pred.multi_pitch.ndim == 3 and pred.multi_pitch.shape[0] == 6
    assert pred.multi_pitch.shape[2] == pred.times.shape[0]
    assert pred.notes, "expected FretNet to detect notes on an in-domain GuitarSet clip"
    for n in pred.notes:
        assert 0 <= n["string"] <= 5
        assert 0 <= n["fret"] <= 19


@pytest.mark.skipif(not _fretnet_ready, reason="fretnet-repro env / checkpoint / clip not available")
def test_fusion_end_to_end_string_fret_consistent():
    result = FusionTranscriber(checkpoint=_CKPT).transcribe(_load_clip())
    assert result.notes
    assigned = [n.note for n in result.notes if n.note.string is not None]
    assert assigned, "expected at least some notes to get a string/fret"
    for n in assigned:
        assert 0 <= n.string <= 5
        assert 0 <= n.fret <= 19
        # pitch must equal the open-string MIDI plus the fret (rounding to semitone)
        assert round(n.pitch_midi) == OPEN_STRING_MIDI[n.string] + n.fret
