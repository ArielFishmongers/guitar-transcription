"""Stage 4 implementations: detect expressive techniques.

`NoOpTechniqueDetector` passes notes through unchanged so the pipeline runs end
to end. `ContourTechniqueDetector` is where real, mostly-classical detection
goes -- it reasons over each note's pitch contour and onset envelope.
"""
from __future__ import annotations

from gtab.stages.base import TechniqueDetector
from gtab.types import AudioBuffer, TranscriptionResult


class NoOpTechniqueDetector(TechniqueDetector):
    """Identity stage: returns notes unchanged."""

    def detect(
        self, guitar: AudioBuffer, result: TranscriptionResult
    ) -> TranscriptionResult:
        return result


class ContourTechniqueDetector(TechniqueDetector):
    """Rule-based detection over pitch contour + onset envelope.

    Much of this is achievable with classical DSP (no training):
      - bend:     monotonic pitch rise (and optional return) within one note
      - slide:    step glide between two stable pitches
      - vibrato:  periodic oscillation about the centre pitch
      - hammer-on/pull-off: a pitch change with an onset that LACKS a sharp
                  pick-attack transient (low spectral-flux spike)

    TODO(research:techniques):
      - requires NoteEvent.pitch_contour populated by Stage 3
      - decide thresholds empirically; consider a small learned classifier on
        contour + onset features if rules prove too brittle
    """

    def detect(
        self, guitar: AudioBuffer, result: TranscriptionResult
    ) -> TranscriptionResult:
        raise NotImplementedError(
            "ContourTechniqueDetector is a stub. Implement during the techniques research stage."
        )
