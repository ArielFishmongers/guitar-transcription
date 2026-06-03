"""Separation quality metric: signal-to-distortion ratio (SDR).

We use the simple "global SDR" from the SiSEC/MDX challenges -- the energy ratio
between the reference and the residual, in decibels:

    SDR = 10 * log10( sum(ref^2) / sum((ref - est)^2) )

Higher is better. It is scale-sensitive (no projection/filtering), which is the
point: a separator that gets the guitar's level wrong is penalised. Use it to
rank separators on the SAME mix, not as an absolute quality number.
"""
from __future__ import annotations

import numpy as np

from gtab.types import AudioBuffer


def global_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    """Global SDR in dB between estimate and reference (1-D arrays, same length)."""
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    numerator = float(np.sum(ref**2))
    denominator = float(np.sum((ref - est) ** 2))
    return 10.0 * np.log10((numerator + eps) / (denominator + eps))


def sdr_buffers(est: AudioBuffer, ref: AudioBuffer) -> float:
    """Global SDR between two AudioBuffers.

    Resamples the estimate to the reference's rate if they differ, then compares
    over their common length.
    """
    e = np.asarray(est.samples, dtype=np.float64)
    r = np.asarray(ref.samples, dtype=np.float64)

    if est.sample_rate != ref.sample_rate:
        import librosa

        e = librosa.resample(
            e, orig_sr=est.sample_rate, target_sr=ref.sample_rate
        )

    n = min(len(e), len(r))
    if n == 0:
        raise ValueError("Cannot compute SDR on an empty signal.")
    return global_sdr(e[:n], r[:n])
