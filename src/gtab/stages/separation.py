"""Stage 2 implementations: isolate the guitar.

`PassthroughSeparator` lets the whole pipeline run before we've chosen a real
separator -- it treats the input as already-guitar, so you can test Stages 3-4
on an isolated guitar recording immediately. `DemucsSeparator` runs Demucs
(`htdemucs_6s`, the only widely-deployed model with a dedicated guitar stem).
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
    """Source separation via Demucs.

    Uses `htdemucs_6s`, the only widely-deployed open model with a dedicated
    `guitar` stem (4-stem models fold guitar into "other"). The guitar stem is
    returned as `Stems.guitar`; `Stems.backing` is the sum of every non-guitar
    stem (so it doubles as a guitar-free play-along track); the individual
    non-guitar stems are also exposed via `Stems.extras`.

    Demucs is a 44.1 kHz stereo model. We resample/upmix the input to the model's
    rate and channels, so any input rate works, but the output stems are always
    44100 Hz. We downmix each stem back to mono to match the project's mono
    `AudioBuffer` contract.

    The model (and its weights) are loaded lazily on first `separate()` call and
    cached on the instance.

    Upgrade path (kept behind this same interface) if guitar quality is
    insufficient: community RoFormer guitar models via `audio-separator`.
    """

    GUITAR_STEM = "guitar"

    def __init__(
        self,
        model_name: str = "htdemucs_6s",
        device: str = "auto",
        shifts: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.shifts = shifts
        self._model = None  # demucs BagOfModels, built lazily
        self._device: str = "cpu"  # resolved when the model is built

    def _resolve_device(self) -> str:
        """Honour an explicit device; otherwise auto-select cuda -> mps -> cpu."""
        if self.device not in (None, "", "auto"):
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from demucs.pretrained import get_model
        except ImportError as e:
            raise ImportError(
                "demucs + torch are required for DemucsSeparator. "
                "Install with: pip install 'gtab[separation]' "
                "(or pip install demucs torch)"
            ) from e

        model = get_model(self.model_name)
        if self.GUITAR_STEM not in model.sources:
            raise ValueError(
                f"Model '{self.model_name}' has no '{self.GUITAR_STEM}' stem "
                f"(stems: {list(model.sources)}). Use a 6-source model such as "
                f"'htdemucs_6s' -- 4-stem models fold guitar into 'other'."
            )
        model.eval()
        device = self._resolve_device()
        try:
            model.to(device)
        except (RuntimeError, AssertionError):
            # e.g. mps/cuda requested but unusable -> fall back to cpu once.
            if device == "cpu":
                raise
            device = "cpu"
            model.to(device)
        self._model = model
        self._device = device
        return model

    def separate(self, audio: AudioBuffer) -> Stems:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        model = self._get_model()

        # Mono numpy -> (1, samples) tensor, then up to the model's rate/channels.
        wav = torch.from_numpy(np.ascontiguousarray(audio.samples)).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        wav = convert_audio(
            wav, audio.sample_rate, model.samplerate, model.audio_channels
        )

        # Demucs is trained on per-mix normalised audio; normalise then undo.
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav_n = (wav - mean) / std

        with torch.no_grad():
            sources = apply_model(
                model,
                wav_n[None],
                device=self._device,
                shifts=self.shifts,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        sources = sources * std + mean  # (n_sources, channels, samples)

        stems = {name: sources[i] for i, name in enumerate(model.sources)}
        sr = int(model.samplerate)
        guitar = self._tensor_to_buffer(stems[self.GUITAR_STEM], sr)

        backing_sum = None
        extras: dict[str, AudioBuffer] = {}
        for name, tensor in stems.items():
            if name == self.GUITAR_STEM:
                continue
            extras[name] = self._tensor_to_buffer(tensor, sr)
            backing_sum = tensor if backing_sum is None else backing_sum + tensor

        if backing_sum is None:  # single-stem model: no backing to build
            backing = AudioBuffer(
                samples=np.zeros_like(guitar.samples), sample_rate=sr
            )
        else:
            backing = self._tensor_to_buffer(backing_sum, sr)

        return Stems(guitar=guitar, backing=backing, extras=extras)

    @staticmethod
    def _tensor_to_buffer(tensor, sample_rate: int) -> AudioBuffer:
        """Demucs stem tensor (channels, samples) -> mono float32 AudioBuffer."""
        arr = tensor.detach().to("cpu").numpy()
        if arr.ndim == 2:  # (channels, samples) -> mono downmix
            arr = arr.mean(axis=0)
        return AudioBuffer(samples=arr.astype(np.float32), sample_rate=sample_rate)
