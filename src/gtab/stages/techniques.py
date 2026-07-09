"""Stage 4 implementations: detect expressive techniques.

`NoOpTechniqueDetector` passes notes through unchanged so the pipeline runs end
to end. `ContourTechniqueDetector` does rule-based detection of the two
pitch-contour techniques the literature finds reliably rule-detectable -- vibrato
and bend (Chen/Su/Yang ISMIR-2015; Kehling DAFx-14) -- from a per-note F0 the
stage estimates itself with pyin. (Basic Pitch's own contour is too coarse to
track bend glides; see the Stage-4 research notes.) Timbre/attack techniques
(harmonic, palm-mute) and the brittle legato family (slide, hammer-on/pull-off)
are deliberately out of scope for this tier.
"""
from __future__ import annotations

import numpy as np

from gtab.stages.base import TechniqueDetector
from gtab.stages.transcription import _build_fretnet_client
from gtab.types import AnnotatedNote, AudioBuffer, Technique, TranscriptionResult


class NoOpTechniqueDetector(TechniqueDetector):
    """Identity stage: returns notes unchanged."""

    def detect(
        self, guitar: AudioBuffer, result: TranscriptionResult
    ) -> TranscriptionResult:
        return result


# --- Rule thresholds (defaults from the Stage-4 research; see memory) --------
# Vibrato: periodic F0 oscillation (Chen/Su/Yang ISMIR-2015).
_VIB_MIN_EXTREMA = 4          # >= 4 alternating pitch extrema
_VIB_MIN_RATE_HZ = 1.25       # inter-extrema spacing <= 400 ms
_VIB_MAX_RATE_HZ = 16.7       # inter-extrema spacing >= 30 ms
_VIB_MAX_STEP_CENTS = 225.0   # neighbouring extrema differ by < 225 cents
_VIB_MIN_EXTENT_CENTS = 20.0  # peak-to-peak must clear the F0 noise floor
# Bend: smooth, mostly-monotonic glide of >= 1 semitone (Kehling gate).
_BEND_MIN_DUR_S = 0.08
_BEND_MAX_STEP_CENTS = 50.0   # median frame-to-frame step < 50 cents (smooth)
_BEND_MIN_CENTS = 100.0       # total excursion >= 1 semitone
_BEND_MIN_MONOTONIC = 0.6     # fraction of steps sharing one sign


def _extrema_indices(x: np.ndarray) -> np.ndarray:
    """Indices of interior local maxima+minima (where the slope flips sign)."""
    if x.size < 3:
        return np.array([], dtype=int)
    d = np.diff(x)
    return np.where(d[:-1] * d[1:] < 0)[0] + 1


def detect_contour_techniques(
    f0_midi,
    times,
    *,
    vib_min_extrema: int = _VIB_MIN_EXTREMA,
    vib_min_rate_hz: float = _VIB_MIN_RATE_HZ,
    vib_max_rate_hz: float = _VIB_MAX_RATE_HZ,
    vib_max_step_cents: float = _VIB_MAX_STEP_CENTS,
    vib_min_extent_cents: float = _VIB_MIN_EXTENT_CENTS,
    bend_min_dur_s: float = _BEND_MIN_DUR_S,
    bend_max_step_cents: float = _BEND_MAX_STEP_CENTS,
    bend_min_cents: float = _BEND_MIN_CENTS,
    bend_min_monotonic: float = _BEND_MIN_MONOTONIC,
) -> list[Technique]:
    """Tag one note's F0 trajectory with vibrato and/or bend.

    Pure numpy (no audio deps) so it is unit-testable with synthetic F0 arrays.
    `f0_midi`/`times` are the note's (smoothed) per-frame pitch in MIDI and frame
    times in seconds. Vibrato and bend are treated as mutually exclusive (a note
    is a vibrato candidate first, per the literature).
    """
    f0 = np.asarray(f0_midi, dtype=float)
    t = np.asarray(times, dtype=float)
    if f0.size < 3 or t.size != f0.size:
        return []

    # --- vibrato: >=4 extrema at a plausible rate with real extent ---
    ext = _extrema_indices(f0)
    if ext.size >= vib_min_extrema:
        spacing = np.diff(t[ext])
        med = float(np.median(spacing)) if spacing.size else 0.0
        rate = 1.0 / (2.0 * med) if med > 0 else 0.0  # extrema are half-cycles
        step_cents = np.abs(np.diff(f0[ext])) * 100.0
        extent_cents = (f0[ext].max() - f0[ext].min()) * 100.0
        if (
            vib_min_rate_hz <= rate <= vib_max_rate_hz
            and extent_cents >= vib_min_extent_cents
            and bool(np.all(step_cents < vib_max_step_cents))
        ):
            return [Technique.VIBRATO]

    # --- bend: smooth, mostly-monotonic glide >= 1 semitone ---
    if (t[-1] - t[0]) >= bend_min_dur_s:
        steps_cents = np.abs(np.diff(f0)) * 100.0
        excursion_cents = (f0.max() - f0.min()) * 100.0
        dsign = np.sign(np.diff(f0))
        dsign = dsign[dsign != 0]
        monotonic = abs(dsign.sum()) / dsign.size if dsign.size else 0.0
        if (
            excursion_cents >= bend_min_cents
            and float(np.median(steps_cents)) < bend_max_step_cents
            and monotonic >= bend_min_monotonic
        ):
            return [Technique.BEND]

    return []


def _note_f0(samples: np.ndarray, sr: int, pitch_midi: float, frame_length: int, edge_trim: int):
    """Per-note F0 (MIDI) + frame times via pyin, voiced-only, edge-trimmed and
    lightly smoothed. Returns (None, None) when the note is too short, out of the
    guitar range, or when pyin can't handle the window (skipped -> no technique).
    """
    import librosa

    # Skip sub-/super-guitar pitches (Basic Pitch emits occasional sub-bass /
    # octave-error artefacts) -- a search band around them either inverts or is
    # too narrow for pyin, and they carry no real technique.
    if samples.size < frame_length or not (40.0 <= pitch_midi <= 88.0):
        return None, None
    fmin = float(librosa.midi_to_hz(pitch_midi - 3))
    fmax = float(librosa.midi_to_hz(pitch_midi + 9))
    hop = frame_length // 4
    try:
        f0, _, _ = librosa.pyin(
            np.ascontiguousarray(samples, dtype=np.float32),
            fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop,
        )
    except Exception:  # noqa: BLE001 -- pathological window -> skip this note
        return None, None
    midi = librosa.hz_to_midi(f0)
    t = np.arange(midi.size) * (hop / sr)
    voiced = ~np.isnan(midi)
    if int(voiced.sum()) < 5:
        return None, None
    midi, t = midi[voiced], t[voiced]
    if midi.size >= 3:  # edge-safe moving average (mode='same' would zero-pad the
        # ends and fabricate a huge excursion); pad with edge values instead.
        midi = np.convolve(np.pad(midi, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    if midi.size > 2 * edge_trim + 3:  # drop attack/decay edge artefacts
        midi, t = midi[edge_trim:-edge_trim], t[edge_trim:-edge_trim]
    return midi, t


class ContourTechniqueDetector(TechniqueDetector):
    """Rule-based vibrato + bend detection from a per-note pyin F0.

    For each note it estimates a clean F0 trajectory with pyin over the note's
    audio window (Basic Pitch's contour is too coarse for bend glides), then
    applies `detect_contour_techniques`. Notes too short/unvoiced for a reliable
    F0 are left unchanged. Thresholds are constructor params so they can be tuned
    from config.
    """

    def __init__(
        self,
        frame_length: int = 1024,
        edge_trim: int = 2,
        vib_min_extent_cents: float = _VIB_MIN_EXTENT_CENTS,
        bend_min_cents: float = _BEND_MIN_CENTS,
        bend_min_monotonic: float = _BEND_MIN_MONOTONIC,
    ) -> None:
        self.frame_length = frame_length
        self.edge_trim = edge_trim
        self.vib_min_extent_cents = vib_min_extent_cents
        self.bend_min_cents = bend_min_cents
        self.bend_min_monotonic = bend_min_monotonic

    def detect(
        self, guitar: AudioBuffer, result: TranscriptionResult
    ) -> TranscriptionResult:
        y = np.asarray(guitar.samples)
        sr = guitar.sample_rate
        out: list[AnnotatedNote] = []
        for an in result.notes:
            n = an.note
            techs = list(an.techniques)
            i0 = max(0, int(n.onset * sr))
            i1 = min(y.size, int(n.offset * sr))
            f0, t = _note_f0(y[i0:i1], sr, n.pitch_midi, self.frame_length, self.edge_trim)
            if f0 is not None:
                for tech in detect_contour_techniques(
                    f0, t,
                    vib_min_extent_cents=self.vib_min_extent_cents,
                    bend_min_cents=self.bend_min_cents,
                    bend_min_monotonic=self.bend_min_monotonic,
                ):
                    if tech not in techs:
                        techs.append(tech)
            out.append(AnnotatedNote(note=n, techniques=techs))
        return TranscriptionResult(
            notes=out, tempo_bpm=result.tempo_bpm, beats=result.beats
        )


_EXPR_TECH = {
    "bend": Technique.BEND,
    "vibrato": Technique.VIBRATO,
    "slide": Technique.SLIDE,
    "harmonic": Technique.HARMONIC,
    "dead_note": Technique.DEAD_NOTE,
}


class LearnedTechniqueDetector(TechniqueDetector):
    """Per-note technique tagging with a trained classifier (Kehling-style).

    Rule-based detection can't separate techniques on this pipeline's coarse
    contour; a learned feature-based classifier can (see the Stage-4 research
    notes). Loads a model trained by scripts/train_technique_classifier.py: an
    expression head (bend/vibrato/slide/harmonic/dead_note/normal) + a muted head
    (-> PALM_MUTE). Features come from the shared `technique_features` extractor
    so training and inference match.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import joblib
        except ImportError as e:
            raise ImportError(
                "scikit-learn + joblib are required for LearnedTechniqueDetector. "
                "Install with: pip install 'gtab[techniques]'"
            ) from e
        import os

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Technique model not found: {self.model_path!r}. Train one with "
                "scripts/train_technique_classifier.py (or set techniques.model_path)."
            )
        from gtab.stages.technique_features import FEATURE_VERSION

        model = joblib.load(self.model_path)
        if model.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Technique model feature_version {model.get('feature_version')} != "
                f"{FEATURE_VERSION}; retrain with the current features."
            )
        self._model = model
        return model

    def detect(
        self, guitar: AudioBuffer, result: TranscriptionResult
    ) -> TranscriptionResult:
        from gtab.stages.technique_features import technique_features

        model = self._load()
        y = np.asarray(guitar.samples)
        sr = guitar.sample_rate

        feats, idxs = [], []
        for i, an in enumerate(result.notes):
            n = an.note
            i0 = max(0, int(n.onset * sr))
            i1 = min(y.size, int(n.offset * sr))
            f = technique_features(y[i0:i1], sr, n.pitch_midi)
            if f is not None:
                feats.append(f)
                idxs.append(i)

        techs = [list(an.techniques) for an in result.notes]
        if feats:
            X = np.asarray(feats)
            expr = model["expr_clf"].predict(X)
            muted = model["mute_clf"].predict(X)
            for k, i in enumerate(idxs):
                t = _EXPR_TECH.get(expr[k])
                if t is not None and t not in techs[i]:
                    techs[i].append(t)
                if int(muted[k]) == 1 and Technique.PALM_MUTE not in techs[i]:
                    techs[i].append(Technique.PALM_MUTE)

        out = [AnnotatedNote(note=an.note, techniques=techs[i])
               for i, an in enumerate(result.notes)]
        return TranscriptionResult(
            notes=out, tempo_bpm=result.tempo_bpm, beats=result.beats
        )


# --- Route A: per-string F0 glide detection (bend/slide) ---------------------
_GLIDE_MIN_CENTS = 70.0      # min per-string F0 excursion to be a glide (NO floor ~18c;
# tuned on IDMT: bend P/R/F1 ~0.39/0.36/0.38 given correct notes/strings)
_GLIDE_SLIDE_NET_CENTS = 90.0  # net start->end change to call it a slide vs bend
_GLIDE_MIN_MONOTONIC = 0.6


def detect_glide(
    f0_midi,
    *,
    min_glide_cents: float = _GLIDE_MIN_CENTS,
    slide_net_cents: float = _GLIDE_SLIDE_NET_CENTS,
    min_monotonic: float = _GLIDE_MIN_MONOTONIC,
):
    """Tag a bend or slide from one note's per-string F0 (MIDI, NaN-gapped).

    A per-string F0 is monophonic, so the note's window contains only that note --
    excursion reflects the intra-note pitch gesture, not note-to-note transitions.
    Returns Technique.SLIDE (large, mostly-monotonic net glide to a new pitch),
    Technique.BEND (a smaller/returning deviation), or None (no glide). Pure numpy.
    """
    m = np.asarray(f0_midi, dtype=float)
    m = m[~np.isnan(m)]
    if m.size < 3:
        return None
    excursion = (m.max() - m.min()) * 100.0
    if excursion < min_glide_cents:
        return None
    net = abs(m[-1] - m[0]) * 100.0
    dsign = np.sign(np.diff(m))
    dsign = dsign[dsign != 0]
    monotonic = abs(dsign.sum()) / dsign.size if dsign.size else 0.0
    if net >= slide_net_cents and monotonic >= min_monotonic:
        return Technique.SLIDE
    return Technique.BEND


class PerStringGlideDetector(TechniqueDetector):
    """Detect bend/slide from FretNet's per-string continuous F0 (Route A).

    FretNet decomposes the (polyphonic) guitar into 6 monophonic per-string F0
    streams; a note's window on its own string contains only that note, so the
    pitch glide of a bend/slide is visible (unlike a polyphonic contour, where
    note-to-note transitions swamp it). Requires the transcription result's notes
    to carry a `string` (i.e. run with the fusion/fretnet transcriber). Runs its
    own FretNet inference via the subprocess client; VIBRATO is out of scope here
    (FretNet's F0 is too smooth for oscillation).
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        min_glide_cents: float = _GLIDE_MIN_CENTS,
        slide_net_cents: float = _GLIDE_SLIDE_NET_CENTS,
        min_monotonic: float = _GLIDE_MIN_MONOTONIC,
        fretnet_python: str | None = None,
        worker_script: str | None = None,
        muda_stub: str | None = None,
        timeout_s: int = 600,
        client=None,
    ) -> None:
        self.min_glide_cents = min_glide_cents
        self.slide_net_cents = slide_net_cents
        self.min_monotonic = min_monotonic
        self._client = _build_fretnet_client(
            checkpoint, client=client, fretnet_python=fretnet_python,
            worker_script=worker_script, muda_stub=muda_stub, timeout_s=timeout_s,
        )

    def detect(
        self, guitar: AudioBuffer, result: TranscriptionResult
    ) -> TranscriptionResult:
        pred = self._client.run(guitar)
        f0 = pred.perstring_f0
        if f0 is None:  # worker didn't export per-string F0 -> nothing to add
            return result
        times = pred.times
        out: list[AnnotatedNote] = []
        for an in result.notes:
            n = an.note
            techs = list(an.techniques)
            if n.string is not None and 0 <= n.string < f0.shape[0]:
                seg = f0[n.string][(times >= n.onset) & (times <= n.offset)]
                tech = detect_glide(
                    seg, min_glide_cents=self.min_glide_cents,
                    slide_net_cents=self.slide_net_cents, min_monotonic=self.min_monotonic,
                )
                if tech is not None and tech not in techs:
                    techs.append(tech)
            out.append(AnnotatedNote(note=n, techniques=techs))
        return TranscriptionResult(
            notes=out, tempo_bpm=result.tempo_bpm, beats=result.beats
        )
