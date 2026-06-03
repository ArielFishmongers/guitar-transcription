"""Stage 2 implementations: isolate the guitar.

`PassthroughSeparator` lets the whole pipeline run before we've chosen a real
separator -- it treats the input as already-guitar, so you can test Stages 3-4
on an isolated guitar recording immediately. `DemucsSeparator` is a stub to be
implemented during the separation research stage.
"""
from __future__ import annotations

import numpy as np

from gtab.stages.base import Separator
from gtab.types import AudioBuffer, Stems


class PassthroughSeparator(Separator):
    """No separation: input is treated as the guitar; backing is silence.

    Useful for (a) developing the rest of the pipeline and (b) inputs that are
    already isolated guitar.
    """

    def separate(self, audio: AudioBuffer) -> Stems:
        silence = AudioBuffer(
            samples=np.zeros_like(audio.samples), sample_rate=audio.sample_rate
        )
        return Stems(guitar=audio, backing=silence)


class DemucsSeparator(Separator):
    """Source separation via Demucs (htdemucs_6s has a dedicated guitar stem).

    TODO(research:separation): wire up inference. Open questions to resolve then:
      - model variant: htdemucs_6s vs bs-roformer -- evaluate guitar SDR on real mixes
      - device (cpu/cuda) and chunking strategy for long files
      - build `backing` as the sum of all non-guitar stems
      - expose other stems (drums/bass/vocals/other) via Stems.extras
    """

    def __init__(self, model_name: str = "htdemucs_6s", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def separate(self, audio: AudioBuffer) -> Stems:
        raise NotImplementedError(
            "DemucsSeparator is a stub. Implement during the separation research stage."
        )
