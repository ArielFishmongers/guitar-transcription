"""Abstract interfaces for each pipeline stage.

Every stage is a *strategy*: the rest of the codebase programs against these
interfaces, never against a concrete implementation. That is what makes a
classical DSP method and an ML method interchangeable -- when we research a
stage, we add a new subclass and swap it into the config. Nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from gtab.types import AudioBuffer, Stems, TranscriptionResult


class Separator(ABC):
    """Stage 2: full mix -> isolated guitar + backing."""

    @abstractmethod
    def separate(self, audio: AudioBuffer) -> Stems: ...


class Transcriber(ABC):
    """Stage 3: guitar audio -> notes with rhythm."""

    @abstractmethod
    def transcribe(self, guitar: AudioBuffer) -> TranscriptionResult: ...


class TechniqueDetector(ABC):
    """Stage 4: refine notes with expressive techniques.

    Receives BOTH the audio and the Stage-3 result so it can analyse the
    waveform / pitch contour around each note, then returns an updated result.
    """

    @abstractmethod
    def detect(
        self, guitar: AudioBuffer, result: TranscriptionResult
    ) -> TranscriptionResult: ...
