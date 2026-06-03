"""The pipeline orchestrator: wires the four stages together.

The whole point of this file is that it is short and boring. All the
interesting, swappable logic lives in the stages; this just sequences them.
"""
from __future__ import annotations

from dataclasses import dataclass

from gtab.stages.base import Separator, TechniqueDetector, Transcriber
from gtab.stages.ingest import load_audio
from gtab.types import PipelineOutput


@dataclass
class Pipeline:
    separator: Separator
    transcriber: Transcriber
    technique_detector: TechniqueDetector

    def run(self, audio_path: str, target_sr: int = 22050) -> PipelineOutput:
        audio = load_audio(audio_path, target_sr=target_sr)          # Stage 1: ingest
        stems = self.separator.separate(audio)                        # Stage 2: separation
        result = self.transcriber.transcribe(stems.guitar)           # Stage 3: notes + rhythm
        result = self.technique_detector.detect(stems.guitar, result)  # Stage 4: techniques
        return PipelineOutput(
            stems=stems,
            transcription=result,
            source_path=audio_path,
            sample_rate=audio.sample_rate,
        )


def build_default_pipeline() -> Pipeline:
    """A runnable pipeline using baseline / passthrough stages.

    Swap any stage here (or via config.build_pipeline_from_config) as research
    lands a better method for that link in the chain.
    """
    from gtab.stages.separation import PassthroughSeparator
    from gtab.stages.techniques import NoOpTechniqueDetector
    from gtab.stages.transcription import BasicPitchTranscriber

    return Pipeline(
        separator=PassthroughSeparator(),
        transcriber=BasicPitchTranscriber(),
        technique_detector=NoOpTechniqueDetector(),
    )
