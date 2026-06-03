"""Stage 3 implementations: notes + rhythm from guitar audio.

`StubTranscriber` returns nothing (zero heavy deps) so you can exercise the
plumbing. `BasicPitchTranscriber` is a runnable BASELINE using Spotify's Basic
Pitch (Apache-2.0, polyphonic, pitch-bend aware). Treat it as a starting point
to benchmark alternatives against during the research phase, not a final choice.
"""
from __future__ import annotations

from gtab.stages.base import Transcriber
from gtab.types import AnnotatedNote, AudioBuffer, NoteEvent, TranscriptionResult


class StubTranscriber(Transcriber):
    """Returns no notes. Lets the pipeline run with zero heavy dependencies."""

    def transcribe(self, guitar: AudioBuffer) -> TranscriptionResult:
        return TranscriptionResult(notes=[])


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
        self, onset_threshold: float = 0.5, frame_threshold: float = 0.3
    ) -> None:
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold

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
            _model_output, _midi_data, note_events = predict(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        notes: list[AnnotatedNote] = []
        for ev in note_events:
            # note_events tuple: (start_s, end_s, pitch_midi, amplitude, pitch_bends)
            start_s, end_s, pitch, amplitude = ev[0], ev[1], ev[2], ev[3]
            notes.append(
                AnnotatedNote(
                    note=NoteEvent(
                        onset=float(start_s),
                        offset=float(end_s),
                        pitch_midi=float(pitch),
                        confidence=float(amplitude),
                    )
                )
            )
        notes.sort(key=lambda n: n.note.onset)
        return TranscriptionResult(notes=notes)
