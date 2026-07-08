"""Stage 3 implementations: notes + rhythm from guitar audio.

`StubTranscriber` returns nothing (zero heavy deps) so you can exercise the
plumbing. `BasicPitchTranscriber` is a runnable BASELINE using Spotify's Basic
Pitch (Apache-2.0, polyphonic, pitch-bend aware). Treat it as a starting point
to benchmark alternatives against during the research phase, not a final choice.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gtab.stages.base import Transcriber
from gtab.types import AnnotatedNote, AudioBuffer, NoteEvent, TranscriptionResult


class StubTranscriber(Transcriber):
    """Returns no notes. Lets the pipeline run with zero heavy dependencies."""

    def transcribe(self, guitar: AudioBuffer) -> TranscriptionResult:
        return TranscriptionResult(notes=[])


# Basic Pitch contour geometry: per-frame bend is an integer offset (in contour
# bins) from the nominal pitch; 3 bins == 1 semitone. Frames are spaced by the
# model's FFT hop over its 22050 Hz analysis rate.
_BP_FRAME_HOP_S = 256 / 22050
_BP_CONTOUR_BINS_PER_SEMITONE = 3


def pitch_bends_to_contour(
    onset: float, pitch_midi: float, bends
) -> list[tuple[float, float]] | None:
    """Basic Pitch per-frame pitch bends -> a pitch_contour for a note.

    `bends` is Basic Pitch's per-frame bend (contour-bin offset from the nominal
    pitch). Returns [(time_seconds, midi_pitch), ...] with the bend folded into
    absolute MIDI, or None when no bend data is available.
    """
    if not bends:
        return None
    return [
        (
            onset + i * _BP_FRAME_HOP_S,
            pitch_midi + b / _BP_CONTOUR_BINS_PER_SEMITONE,
        )
        for i, b in enumerate(bends)
    ]


def _maybe_beats(guitar: AudioBuffer, enabled: bool):
    """(tempo_bpm, beats|None) from a shared beat tracker, or (None, None)."""
    if not enabled:
        return None, None
    from gtab.rhythm import estimate_beats

    tempo_bpm, beat_times = estimate_beats(guitar)
    return tempo_bpm, (beat_times or None)


class BasicPitchTranscriber(Transcriber):
    """Baseline transcription using Spotify's Basic Pitch.

    TODO(research:transcription):
      - benchmark against alternatives: pYIN/CREPE (monophonic), NMF or a
        fine-tuned CNN (polyphony) -- see the per-stage research notes
      - tune onset_threshold / frame_threshold on held-out guitar audio
      - add beat/tempo tracking (librosa.beat / madmom) to populate
        tempo_bpm + beats
      - carry Basic Pitch's pitch-bend output into NoteEvent.pitch_contour so
        Stage 4 can use it
    """

    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        track_beats: bool = True,
    ) -> None:
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.track_beats = track_beats

    def transcribe(self, guitar: AudioBuffer) -> TranscriptionResult:
        try:
            import os
            import tempfile

            import soundfile as sf
            from basic_pitch.inference import predict
        except ImportError as e:
            raise ImportError(
                "basic-pitch + soundfile are required for this transcriber. "
                "Install with: pip install 'gtab[transcription]' (or pip install basic-pitch soundfile)"
            ) from e

        # Basic Pitch's predict() takes a file path; write a temp WAV.
        # TODO: verify exact predict() signature/return for your installed version.
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, guitar.samples, guitar.sample_rate)
            _model_output, _midi_data, note_events = predict(
                tmp_path,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        notes: list[AnnotatedNote] = []
        for ev in note_events:
            # note_events tuple: (start_s, end_s, pitch_midi, amplitude, pitch_bends)
            start_s, end_s, pitch, amplitude = ev[0], ev[1], ev[2], ev[3]
            bends = ev[4] if len(ev) > 4 else None
            notes.append(
                AnnotatedNote(
                    note=NoteEvent(
                        onset=float(start_s),
                        offset=float(end_s),
                        pitch_midi=float(pitch),
                        confidence=float(amplitude),
                        pitch_contour=pitch_bends_to_contour(
                            float(start_s), float(pitch), bends
                        ),
                    )
                )
            )
        notes.sort(key=lambda n: n.note.onset)
        tempo_bpm, beats = _maybe_beats(guitar, self.track_beats)
        return TranscriptionResult(notes=notes, tempo_bpm=tempo_bpm, beats=beats)


# Standard-tuning open-string MIDI by string index (0 = low E ... 5 = high E).
OPEN_STRING_MIDI: tuple[int, ...] = (40, 45, 50, 55, 59, 64)


def assign_string_fret(
    pitch_midi: float,
    onset: float,
    offset: float,
    multi_pitch: np.ndarray,
    times: np.ndarray,
    open_string_midi: Sequence[int] = OPEN_STRING_MIDI,
    profile_low: int = 40,
    num_frets: int = 19,
    gate: float = 0.0,
) -> tuple[int | None, int | None]:
    """Assign a (string, fret) to one note from FretNet's per-frame activation.

    This is the core of the fusion transcriber, kept as a pure numpy function so
    it is unit-testable without any heavy deps. It reads FretNet's per-string,
    per-pitch activation over the note's time window and picks the string whose
    activation is strongest -- among only the strings that can *physically* play
    the pitch in standard tuning.

    Args:
        pitch_midi: the note's pitch (rounded to nearest semitone for the lookup).
        onset, offset: note bounds in seconds (same clock as ``times``).
        multi_pitch: FretNet activation, shape (S strings, B pitch-bins, F frames);
            bin ``b`` corresponds to MIDI ``profile_low + b``.
        times: (F,) frame times in seconds.
        open_string_midi: open-string MIDI per string index (0 = low E).
        profile_low: MIDI of pitch-bin 0.
        num_frets: highest fret the model represents (FretNet: 19).
        gate: withhold (return None) if the winning string's activation is below
            this. Default 0.0 = always assign when a valid string exists.

    Returns:
        ``(string, fret)`` or ``(None, None)`` when the pitch is outside the
        model's bin range, no string can physically play it, or the best
        activation is below ``gate``.
    """
    p = int(round(pitch_midi))
    b = p - profile_low
    if b < 0 or b >= multi_pitch.shape[1]:
        return None, None

    frames = np.nonzero((times >= onset) & (times <= offset))[0]
    if frames.size == 0:  # note falls between frames -> use the nearest one
        frames = np.array([int(np.argmin(np.abs(times - onset)))])

    scores = multi_pitch[:, b, frames].mean(axis=1)  # (S,)

    # A string can play pitch p iff the implied fret is on the fretboard.
    valid = np.array(
        [0 <= (p - m) <= num_frets for m in open_string_midi], dtype=bool
    )
    if not valid.any():
        return None, None

    string = int(np.argmax(np.where(valid, scores, -np.inf)))
    if scores[string] < gate:
        return None, None
    return string, p - int(open_string_midi[string])


def _build_fretnet_client(
    checkpoint: str | None,
    *,
    client=None,
    fretnet_python: str | None = None,
    worker_script: str | None = None,
    muda_stub: str | None = None,
    timeout_s: int = 600,
):
    """Construct a FretNetClient, forwarding only the params the caller set so
    the client keeps its own defaults. `client` (an already-built client or a
    test double) short-circuits construction."""
    if client is not None:
        return client
    if not checkpoint:
        raise ValueError(
            "A FretNet `checkpoint` path is required "
            "(set `checkpoint:` under transcription in the config)."
        )
    from gtab.stages.fretnet_client import FretNetClient

    kwargs = {"checkpoint": checkpoint, "timeout_s": timeout_s}
    if fretnet_python is not None:
        kwargs["fretnet_python"] = fretnet_python
    if worker_script is not None:
        kwargs["worker_script"] = worker_script
    if muda_stub is not None:
        kwargs["muda_stub"] = muda_stub
    return FretNetClient(**kwargs)


class FretNetTranscriber(Transcriber):
    """Guitar-specific transcription via a trained FretNet checkpoint.

    FretNet emits notes WITH (string, fret) and is strong on acoustic,
    standard-tuning material (TDR ~0.90). Its research deps conflict with gtab's
    env, so inference runs in the separate `fretnet-repro` conda env via a
    subprocess (see `fretnet_client`); this class just adapts the returned notes.

    NOTE: FretNet under-detects note onsets on out-of-domain audio (electric /
    effected / separated), so its note *count* can collapse there even though its
    string/fret head stays reliable. For robust note detection use
    `FusionTranscriber`, which keeps Basic Pitch's notes and borrows only
    FretNet's string/fret.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        track_beats: bool = True,
        fretnet_python: str | None = None,
        worker_script: str | None = None,
        muda_stub: str | None = None,
        timeout_s: int = 600,
        client=None,
    ) -> None:
        self.checkpoint = checkpoint
        self.track_beats = track_beats
        self._client = _build_fretnet_client(
            checkpoint,
            client=client,
            fretnet_python=fretnet_python,
            worker_script=worker_script,
            muda_stub=muda_stub,
            timeout_s=timeout_s,
        )

    def transcribe(self, guitar: AudioBuffer) -> TranscriptionResult:
        pred = self._client.run(guitar)
        notes = [
            AnnotatedNote(
                note=NoteEvent(
                    onset=float(n["onset"]),
                    offset=float(n["offset"]),
                    pitch_midi=float(n["pitch_midi"]),
                    confidence=1.0,
                    string=n["string"],
                    fret=n["fret"],
                )
            )
            for n in pred.notes
        ]
        notes.sort(key=lambda n: n.note.onset)
        tempo_bpm, beats = _maybe_beats(guitar, self.track_beats)
        return TranscriptionResult(notes=notes, tempo_bpm=tempo_bpm, beats=beats)


class FusionTranscriber(Transcriber):
    """Basic Pitch notes + FretNet string/fret.

    Basic Pitch is the stronger, domain-robust note detector; FretNet uniquely
    predicts the fretboard. This transcriber keeps Basic Pitch's notes verbatim
    and, for each one, looks up a (string, fret) from FretNet's per-frame
    tablature activation (`assign_string_fret`) -- pairing each model to its
    strength and sidestepping FretNet's fragile onset head.

    String/fret out of FretNet's acoustic-GuitarSet domain are not yet validated
    (see the plan's follow-up); `gate` can withhold low-confidence assignments.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        gate: float = 0.05,
        track_beats: bool = True,
        fretnet_python: str | None = None,
        worker_script: str | None = None,
        muda_stub: str | None = None,
        timeout_s: int = 600,
        client=None,
    ) -> None:
        self.bp = BasicPitchTranscriber(onset_threshold, frame_threshold, track_beats)
        self.gate = gate
        self._client = _build_fretnet_client(
            checkpoint,
            client=client,
            fretnet_python=fretnet_python,
            worker_script=worker_script,
            muda_stub=muda_stub,
            timeout_s=timeout_s,
        )

    def transcribe(self, guitar: AudioBuffer) -> TranscriptionResult:
        bp_result = self.bp.transcribe(guitar)
        if not bp_result.notes:
            return bp_result  # no notes -> nothing to localise, skip the subprocess

        pred = self._client.run(guitar)
        notes: list[AnnotatedNote] = []
        for an in bp_result.notes:
            n = an.note
            string, fret = assign_string_fret(
                n.pitch_midi,
                n.onset,
                n.offset,
                pred.multi_pitch,
                pred.times,
                pred.open_string_midi,
                pred.profile_low,
                gate=self.gate,
            )
            notes.append(
                AnnotatedNote(
                    note=NoteEvent(
                        onset=n.onset,
                        offset=n.offset,
                        pitch_midi=n.pitch_midi,  # keep Basic Pitch's float (bend info)
                        confidence=n.confidence,
                        pitch_contour=n.pitch_contour,
                        string=string,
                        fret=fret,
                    ),
                    techniques=an.techniques,
                )
            )
        return TranscriptionResult(
            notes=notes, tempo_bpm=bp_result.tempo_bpm, beats=bp_result.beats
        )
