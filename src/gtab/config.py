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
from gtab.stages.transcription import BasicPitchTranscriber, StubTranscriber

SEPARATORS = {"passthrough": PassthroughSeparator, "demucs": DemucsSeparator}
TRANSCRIBERS = {"stub": StubTranscriber, "basic_pitch": BasicPitchTranscriber}
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


def build_pipeline_from_config(cfg: dict[str, Any]) -> Pipeline:
    tr_impl = cfg.get("transcription", {}).get("impl", "basic_pitch")
    tech_impl = cfg.get("techniques", {}).get("impl", "noop")
    return Pipeline(
        separator=_build_separator(cfg),
        transcriber=TRANSCRIBERS[tr_impl](),
        technique_detector=TECHNIQUES[tech_impl](),
    )
