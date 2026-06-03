"""Export a TranscriptionResult / PipelineOutput to inspectable formats.

JSON is the always-available, dependency-free way to eyeball what each stage
produced. MIDI and WAV exports help you *listen* to the result while debugging.
"""
from __future__ import annotations

import json

from gtab.types import PipelineOutput, TranscriptionResult


def transcription_to_dict(result: TranscriptionResult) -> dict:
    return {
        "tempo_bpm": result.tempo_bpm,
        "beats": result.beats,
        "notes": [
            {
                "onset": n.note.onset,
                "offset": n.note.offset,
                "pitch_midi": n.note.pitch_midi,
                "confidence": n.note.confidence,
                "techniques": [t.value for t in n.techniques],
            }
            for n in result.notes
        ],
    }


def save_json(result: TranscriptionResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump(transcription_to_dict(result), f, indent=2)


def save_midi(result: TranscriptionResult, path: str) -> None:
    """Export notes to a MIDI file for quick listening / DAW import."""
    try:
        import pretty_midi
    except ImportError as e:
        raise ImportError(
            "pretty_midi is required for MIDI export. Install with: pip install pretty_midi"
        ) from e

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=25)  # acoustic steel guitar
    for n in result.notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=max(1, min(127, int(n.note.confidence * 127))),
                pitch=int(round(n.note.pitch_midi)),
                start=n.note.onset,
                end=max(n.note.onset + 0.01, n.note.offset),
            )
        )
    pm.instruments.append(inst)
    pm.write(path)


def save_stems(output: PipelineOutput, out_dir: str) -> None:
    """Write guitar + backing stems as WAV for inspection / play-along."""
    import os

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required for stem export. Install with: pip install soundfile"
        ) from e

    os.makedirs(out_dir, exist_ok=True)
    sf.write(
        os.path.join(out_dir, "guitar.wav"),
        output.stems.guitar.samples,
        output.stems.guitar.sample_rate,
    )
    sf.write(
        os.path.join(out_dir, "backing.wav"),
        output.stems.backing.samples,
        output.stems.backing.sample_rate,
    )
