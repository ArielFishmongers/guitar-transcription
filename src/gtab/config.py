"""Load YAML config and assemble a Pipeline from named stage implementations.

A registry maps the `impl` strings in config/default.yaml to classes, so you can
change which method runs at each stage by editing YAML -- no code edits needed.
(Per-impl parameters are wired with defaults for now; pass them through here as
the stages mature.)
"""
from __future__ import annotations

from typing import Any

from gtab.pipeline import Pipeline
from gtab.stages.separation import DemucsSeparator, PassthroughSeparator
from gtab.stages.techniques import ContourTechniqueDetector, NoOpTechniqueDetector
from gtab.stages.transcription import (
    BasicPitchTranscriber,
    FretNetTranscriber,
    FusionTranscriber,
    StubTranscriber,
)

SEPARATORS = {"passthrough": PassthroughSeparator, "demucs": DemucsSeparator}
TRANSCRIBERS = {
    "stub": StubTranscriber,
    "basic_pitch": BasicPitchTranscriber,
    "fretnet": FretNetTranscriber,
    "fusion": FusionTranscriber,
}
TECHNIQUES = {"noop": NoOpTechniqueDetector, "contour": ContourTechniqueDetector}


def load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _build_separator(cfg: dict[str, Any]):
    sep_cfg = cfg.get("separation", {})
    impl = sep_cfg.get("impl", "passthrough")
    cls = SEPARATORS[impl]
    if impl == "demucs":
        return cls(
            model_name=sep_cfg.get("model_name", "htdemucs_6s"),
            device=sep_cfg.get("device", "auto"),
            shifts=int(sep_cfg.get("shifts", 1)),
        )
    return cls()


def _build_transcriber(cfg: dict[str, Any]):
    tr_cfg = cfg.get("transcription", {})
    impl = tr_cfg.get("impl", "basic_pitch")
    cls = TRANSCRIBERS[impl]
    if impl == "basic_pitch":
        return cls(
            onset_threshold=float(tr_cfg.get("onset_threshold", 0.5)),
            frame_threshold=float(tr_cfg.get("frame_threshold", 0.3)),
        )
    if impl in ("fretnet", "fusion"):
        if "checkpoint" not in tr_cfg:
            raise KeyError(
                f"transcription.impl={impl!r} requires a 'checkpoint' path "
                "(the trained FretNet .pt) under transcription in the config."
            )
        common = dict(
            checkpoint=tr_cfg["checkpoint"],
            fretnet_python=tr_cfg.get("fretnet_python"),
            worker_script=tr_cfg.get("worker_script"),
            muda_stub=tr_cfg.get("muda_stub"),
            timeout_s=int(tr_cfg.get("timeout_s", 600)),
        )
        if impl == "fusion":
            return cls(
                onset_threshold=float(tr_cfg.get("onset_threshold", 0.5)),
                frame_threshold=float(tr_cfg.get("frame_threshold", 0.3)),
                gate=float(tr_cfg.get("gate", 0.05)),
                **common,
            )
        return cls(**common)
    return cls()


def build_pipeline_from_config(cfg: dict[str, Any]) -> Pipeline:
    tech_impl = cfg.get("techniques", {}).get("impl", "noop")
    return Pipeline(
        separator=_build_separator(cfg),
        transcriber=_build_transcriber(cfg),
        technique_detector=TECHNIQUES[tech_impl](),
    )
