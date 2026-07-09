"""Per-note feature extraction for learned technique classification.

Shared by the training script (`scripts/train_technique_classifier.py`) and the
`LearnedTechniqueDetector` at inference, so the model always sees identical
features. Features follow the Kehling DAFx-14 family (spectral + temporal + F0
shape) that reaches ~83% expression-style accuracy on IDMT-SMT-GUITAR.

librosa is a core gtab dependency and imported lazily inside the function to
keep module import cheap. Returns a fixed-length float vector, or None when the
note is too short / features are non-finite (caller skips it).
"""
from __future__ import annotations

import numpy as np

# Bump when the feature layout changes so a stale model artifact is rejected.
FEATURE_VERSION = 1

FEATURE_NAMES = [
    "duration", "attack_rel", "decay_slope", "rms_mean", "rms_std",
    "centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean",
    "flatness_mean", "zcr_mean", "flux_mean", "contrast_mean",
    *[f"mfcc{i}_mean" for i in range(13)],
    "f0_excursion_cents", "f0_oscillation", "f0_monotonic_frac", "f0_abs_slope_cents",
]
N_FEATURES = len(FEATURE_NAMES)

_N_FFT = 1024
_HOP = 256


def technique_features(samples: np.ndarray, sr: int, pitch_midi: float):
    """Extract the fixed-length feature vector for one note, or None if unusable."""
    import librosa

    seg = np.ascontiguousarray(samples, dtype=np.float32)
    if seg.size < _N_FFT:
        return None

    S = np.abs(librosa.stft(seg, n_fft=_N_FFT, hop_length=_HOP)) + 1e-9
    rms = librosa.feature.rms(y=seg, frame_length=_N_FFT, hop_length=_HOP).ravel()
    flux = np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=0)) if S.shape[1] > 1 else np.zeros(1)

    feats = [
        seg.size / sr,
        float(np.argmax(rms)) / max(1, rms.size),
        float(np.log(rms[-1] + 1e-9) - np.log(rms.max() + 1e-9)),
        float(rms.mean()),
        float(rms.std()),
        float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
        float(librosa.feature.spectral_centroid(S=S, sr=sr).std()),
        float(librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()),
        float(librosa.feature.spectral_rolloff(S=S, sr=sr).mean()),
        float(librosa.feature.spectral_flatness(S=S).mean()),
        float(librosa.feature.zero_crossing_rate(seg, frame_length=_N_FFT, hop_length=_HOP).mean()),
        float(flux.mean()),
        float(librosa.feature.spectral_contrast(S=S, sr=sr).mean()),
    ]
    feats += [float(v) for v in librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13).mean(axis=1)]
    feats += list(_f0_shape(seg, sr, pitch_midi))

    vec = np.asarray(feats, dtype=np.float64)
    if vec.size != N_FEATURES or not np.all(np.isfinite(vec)):
        return None
    return vec


def _f0_shape(seg: np.ndarray, sr: int, pitch_midi: float):
    """(excursion_cents, oscillation, monotonic_frac, abs_slope_cents) from pyin."""
    import librosa

    if not (40.0 <= pitch_midi <= 88.0):
        return 0.0, 0.0, 0.0, 0.0
    try:
        f0, _, _ = librosa.pyin(
            seg, fmin=float(librosa.midi_to_hz(pitch_midi - 3)),
            fmax=float(librosa.midi_to_hz(pitch_midi + 9)), sr=sr, frame_length=_N_FFT,
        )
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0, 0.0
    m = librosa.hz_to_midi(f0)
    m = m[~np.isnan(m)]
    if m.size < 5:
        return 0.0, 0.0, 0.0, 0.0
    d = np.diff(m) * 100.0
    nz = d[np.abs(d) > 1e-6]
    big = d[np.abs(d) >= 15.0]
    excursion = float((m.max() - m.min()) * 100.0)
    oscillation = float(np.sum(np.diff(np.sign(big)) != 0)) if big.size >= 2 else 0.0
    monotonic = float(abs(np.sign(nz).sum()) / nz.size) if nz.size else 0.0
    abs_slope = float(np.mean(np.abs(d))) if d.size else 0.0
    return excursion, oscillation, monotonic, abs_slope
