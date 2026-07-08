"""Beat / tempo estimation -- the 'rhythm' half of Stage 3.

A `TranscriptionResult` carries `tempo_bpm` + `beats`; transcribers call this to
fill them so the pipeline emits *notes with rhythm* (not just notes). Kept as a
standalone helper so every transcriber shares one beat tracker instead of each
reimplementing it. librosa is a core gtab dependency, imported lazily here to
keep module import cheap.
"""
from __future__ import annotations

from gtab.types import AudioBuffer


def estimate_beats(audio: AudioBuffer) -> tuple[float | None, list[float]]:
    """Estimate (tempo_bpm, beat_times_seconds) from a guitar buffer.

    Returns (None, []) for audio too short/silent for beat tracking rather than
    raising, so a transcriber can always call it safely.
    """
    import numpy as np
    import librosa

    y = np.ascontiguousarray(audio.samples, dtype=np.float32)
    if y.size < audio.sample_rate // 2:  # < ~0.5 s -> not enough for a tempo
        return None, []

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=audio.sample_rate)
    # librosa >=0.10 returns tempo as a 0-d/1-d ndarray; coerce to a scalar.
    tempo_val = float(np.asarray(tempo).ravel()[0]) if np.size(tempo) else None
    if tempo_val is not None and tempo_val <= 0:
        tempo_val = None
    beat_times = librosa.frames_to_time(beat_frames, sr=audio.sample_rate)
    return tempo_val, [float(t) for t in beat_times]
