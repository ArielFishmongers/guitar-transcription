"""Learned-technique feature extractor + detector tests.

`technique_features` runs on librosa (a core dep) so it always runs. The
`LearnedTechniqueDetector` integration test is skipped unless a trained model
artifact exists.
"""
import os

import numpy as np
import pytest

from gtab.stages.technique_features import (
    FEATURE_NAMES,
    N_FEATURES,
    technique_features,
)
from gtab.types import AnnotatedNote, AudioBuffer, NoteEvent, TranscriptionResult


def _tone(midi=57.0, dur=0.5, sr=22050):
    t = np.arange(int(sr * dur)) / sr
    hz = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    return (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32), sr


def test_feature_vector_shape_and_finite():
    y, sr = _tone()
    f = technique_features(y, sr, 57.0)
    assert f is not None
    assert len(f) == N_FEATURES == len(FEATURE_NAMES)
    assert np.all(np.isfinite(f))


def test_feature_extractor_rejects_too_short():
    assert technique_features(np.zeros(100, np.float32), 22050, 57.0) is None


def test_feature_extractor_handles_out_of_range_pitch():
    # sub-guitar pitch -> F0 features degrade gracefully, still a full finite vector
    y, sr = _tone(midi=57.0)
    f = technique_features(y, sr, 20.0)  # pyin skipped for out-of-range pitch
    assert f is not None and len(f) == N_FEATURES and np.all(np.isfinite(f))


_MODEL = "models/technique_clf.joblib"


@pytest.mark.skipif(not os.path.exists(_MODEL), reason="trained technique model not present")
def test_learned_detector_runs_and_tags():
    from gtab.stages.techniques import LearnedTechniqueDetector

    y, sr = _tone(midi=57.0, dur=0.6)
    result = TranscriptionResult(
        notes=[AnnotatedNote(note=NoteEvent(onset=0.0, offset=0.6, pitch_midi=57.0))]
    )
    out = LearnedTechniqueDetector(_MODEL).detect(
        AudioBuffer(samples=y, sample_rate=sr), result
    )
    assert len(out.notes) == 1
    # techniques is a (possibly empty) list of valid Technique values
    from gtab.types import Technique

    assert all(isinstance(t, Technique) for t in out.notes[0].techniques)


def test_learned_detector_missing_model_errors():
    from gtab.stages.techniques import LearnedTechniqueDetector

    det = LearnedTechniqueDetector("/nonexistent/model.joblib")
    y, sr = _tone()
    res = TranscriptionResult(notes=[AnnotatedNote(note=NoteEvent(0.0, 0.5, 57.0))])
    with pytest.raises((FileNotFoundError, ImportError)):
        det.detect(AudioBuffer(samples=y, sample_rate=sr), res)
