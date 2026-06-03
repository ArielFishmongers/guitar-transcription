"""Stage 1: decode any audio file to a normalised mono AudioBuffer.

This stage is effectively *solved* -- it needs no research. ffmpeg (via
librosa/audioread) decodes essentially any format; we downmix to mono and
resample to a target rate so every downstream stage sees a consistent signal.
"""
from __future__ import annotations

import numpy as np

from gtab.types import AudioBuffer

DEFAULT_SAMPLE_RATE = 22050


def load_audio(
    path: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> AudioBuffer:
    """Load an audio file as a mono float32 AudioBuffer resampled to target_sr."""
    try:
        import librosa
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "librosa is required for audio loading. Install with: pip install librosa"
        ) from e

    samples, sr = librosa.load(path, sr=target_sr, mono=mono)
    samples = np.asarray(samples, dtype=np.float32)
    return AudioBuffer(samples=samples, sample_rate=sr)
