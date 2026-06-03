"""Pipeline plumbing tests using fake stages (numpy + pytest only).

These verify the *contracts and wiring*, not any real DSP/ML. They should always
pass regardless of which heavy optional dependencies are installed.
"""
import numpy as np

from gtab.pipeline import Pipeline
from gtab.stages.base import Transcriber
from gtab.stages.separation import PassthroughSeparator
from gtab.stages.techniques import NoOpTechniqueDetector
from gtab.types import (
    AnnotatedNote,
    AudioBuffer,
    NoteEvent,
    Stems,
    TranscriptionResult,
)


class FakeTranscriber(Transcriber):
    def transcribe(self, guitar: AudioBuffer) -> TranscriptionResult:
        return TranscriptionResult(
            notes=[AnnotatedNote(note=NoteEvent(onset=0.0, offset=0.5, pitch_midi=64.0))]
        )


def test_audiobuffer_duration():
    buf = AudioBuffer(samples=np.zeros(22050, dtype=np.float32), sample_rate=22050)
    assert abs(buf.duration - 1.0) < 1e-9


def test_passthrough_separator_returns_input_as_guitar():
    audio = AudioBuffer(samples=np.ones(100, dtype=np.float32), sample_rate=22050)
    stems = PassthroughSeparator().separate(audio)
    assert isinstance(stems, Stems)
    assert np.allclose(stems.guitar.samples, audio.samples)
    assert np.allclose(stems.backing.samples, 0.0)


def test_pipeline_stages_chain_end_to_end():
    pipeline = Pipeline(
        separator=PassthroughSeparator(),
        transcriber=FakeTranscriber(),
        technique_detector=NoOpTechniqueDetector(),
    )
    audio = AudioBuffer(samples=np.zeros(2205, dtype=np.float32), sample_rate=22050)
    stems = pipeline.separator.separate(audio)
    result = pipeline.transcriber.transcribe(stems.guitar)
    result = pipeline.technique_detector.detect(stems.guitar, result)
    assert len(result.notes) == 1
    assert result.notes[0].note.pitch_midi == 64.0
