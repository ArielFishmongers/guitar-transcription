"""Stage 2 separation tests.

The DemucsSeparator test is skipped when demucs isn't installed so the core
suite stays dependency-light (numpy + pytest). It downloads model weights on
first run and is slow; it asserts wiring and stem routing, not quality.
"""
import importlib.util

import numpy as np
import pytest

from gtab.eval.separation import global_sdr, sdr_buffers
from gtab.types import AudioBuffer

demucs_available = importlib.util.find_spec("demucs") is not None


def test_global_sdr_perfect_match_is_high():
    ref = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    assert global_sdr(ref, ref) > 60.0


def test_sdr_buffers_resamples_to_reference_rate():
    ref = AudioBuffer(samples=np.ones(2000, dtype=np.float32), sample_rate=44100)
    est = AudioBuffer(samples=np.ones(1000, dtype=np.float32), sample_rate=22050)
    # Constant signals at different rates -> still a near-perfect match after resample.
    assert sdr_buffers(est, ref) > 30.0


@pytest.mark.skipif(not demucs_available, reason="demucs not installed")
def test_demucs_separates_short_clip():
    from gtab.stages.separation import DemucsSeparator

    sr = 44100
    rng = np.random.default_rng(0)
    samples = (0.1 * rng.standard_normal(sr)).astype(np.float32)  # ~1 s
    audio = AudioBuffer(samples=samples, sample_rate=sr)

    stems = DemucsSeparator(device="cpu", shifts=1).separate(audio)

    assert stems.guitar.sample_rate == 44100
    assert stems.backing.sample_rate == 44100
    assert len(stems.guitar.samples) > 0
    # htdemucs_6s exposes guitar + 5 non-guitar stems.
    assert "guitar" not in stems.extras
    assert {"drums", "bass", "other", "vocals", "piano"} <= set(stems.extras)
