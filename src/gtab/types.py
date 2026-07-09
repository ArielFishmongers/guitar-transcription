"""Core data contracts shared across all pipeline stages.

These types are the *stable backbone* of the project. Stage implementations
(separation, transcription, technique detection) can be swapped freely as long
as they consume and produce these types. Change them deliberately -- a change
here ripples through every stage.

The chain:
    bytes on disk
      -> AudioBuffer          (Stage 1: ingest)
      -> Stems                (Stage 2: separation)   guitar + backing
      -> TranscriptionResult  (Stage 3: transcription) notes with rhythm
      -> TranscriptionResult  (Stage 4: techniques)    notes + effects
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class AudioBuffer:
    """A mono audio signal plus its sample rate.

    samples: 1-D float32 array, nominally in [-1.0, 1.0].
    sample_rate: samples per second (Hz).
    """

    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass
class Stems:
    """Output of Stage 2 (source separation).

    guitar:  the isolated guitar signal (the input to transcription).
    backing: everything-except-guitar (doubles as the play-along backing track).
    extras:  optional named stems, e.g. 'drums', 'bass', 'vocals', 'other'.
    """

    guitar: AudioBuffer
    backing: AudioBuffer
    extras: dict[str, AudioBuffer] = field(default_factory=dict)


class Technique(str, Enum):
    """Expressive techniques attached to a note in Stage 4.

    Subclassing `str` makes these trivially JSON-serialisable.
    """

    HAMMER_ON = "hammer_on"
    PULL_OFF = "pull_off"
    SLIDE = "slide"
    BEND = "bend"
    VIBRATO = "vibrato"
    PALM_MUTE = "palm_mute"
    TAP = "tap"
    HARMONIC = "harmonic"
    DEAD_NOTE = "dead_note"


@dataclass
class NoteEvent:
    """A single transcribed note (Stage 3 output).

    onset/offset:  start and end time in seconds.
    pitch_midi:    MIDI note number. Float so a bend's centre pitch is expressible.
    confidence:    model confidence in [0, 1].
    pitch_contour: optional list of (time_seconds, midi_pitch) samples. Stage 4
                   reads this to detect bends/slides/vibrato from the F0 trajectory,
                   so it is worth populating in Stage 3 where the method allows.
    string:        guitar string index, 0=low E ... 5=high E, or None if the
                   transcriber doesn't predict it. Pitch is recoverable as
                   open_string_midi[string] + fret (standard tuning open strings:
                   E2=40, A2=45, D3=50, G3=55, B3=59, E4=64).
    fret:          fret number (0=open) on `string`, or None if not predicted.
    """

    onset: float
    offset: float
    pitch_midi: float
    confidence: float = 1.0
    pitch_contour: list[tuple[float, float]] | None = None
    string: int | None = None
    fret: int | None = None

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class AnnotatedNote:
    """A note enriched with detected techniques (Stage 4 output).

    NOTE: string/fret assignment is intentionally OUT OF SCOPE for this pipeline.
    A downstream tab-assignment stage would consume these notes and add it.
    """

    note: NoteEvent
    techniques: list[Technique] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """The symbolic result of the transcription + technique stages.

    notes:      transcribed (and possibly annotated) notes, time-ordered.
    tempo_bpm:  estimated tempo, if a beat tracker ran.
    beats:      beat times in seconds, used for quantisation / display.
    """

    notes: list[AnnotatedNote]
    tempo_bpm: float | None = None
    beats: list[float] | None = None


@dataclass
class PipelineOutput:
    """Everything the pipeline produces for one input file."""

    stems: Stems
    transcription: TranscriptionResult
    source_path: str
    sample_rate: int
