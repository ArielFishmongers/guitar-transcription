#!/usr/bin/env python3
"""Transcribe a clip with a trained FretNet checkpoint -> gtab outputs.

Produces, in --out: notes.json (gtab TranscriptionResult with string/fret),
a piano-roll PNG, a MIDI file, and a playable WAV.

FretNet's research deps (amt_tools + the cwitkowitz repos) conflict with gtab's
core stack, so this lives behind the separate `fretnet-repro` env. Run it as:

  PYTHONPATH="src:/tmp/fnstub" \
  /opt/miniconda3/envs/fretnet-repro/bin/python scripts/fretnet_infer.py \
      --audio <clip.wav> --out data/interim/fretnet

(`src` puts gtab on the path; `/tmp/fnstub` provides the muda stub.)
"""
from __future__ import annotations

import argparse
import glob
import os

import torch

# torch>=2.6 loads the pickled model object -> restore old default
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{"weights_only": False, **k})

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import amt_tools.tools as tools  # noqa: E402
from amt_tools.features import HCQT  # noqa: E402
from amt_tools.transcribe import (  # noqa: E402
    ComboEstimator,
    TablatureWrapper,
    StackedOffsetsWrapper,
    StackedNoteTranscriber,
)
from amt_tools.inference import run_offline  # noqa: E402
from guitar_transcription_continuous.estimators import (  # noqa: E402
    StackedPitchListTablatureWrapper,
)
import guitar_transcription_continuous.utils as gtc_utils  # noqa: E402

from gtab.types import AnnotatedNote, AudioBuffer, NoteEvent, TranscriptionResult  # noqa: E402
from gtab.io.export import save_json, save_midi, synthesize_wav  # noqa: E402
from gtab.viz.pianoroll import plot_transcription  # noqa: E402

# Standard-tuning open-string MIDI by string index (0 = low E ... 5 = high E)
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]
_DEFAULT_CKPT = (
    "/Users/ariel/Documents/Personal Projects/fretnet-repro/"
    "guitar-transcription-continuous/generated/experiments/"
    "FretNet_GuitarSetPlus_HCQT_X/models/fold-0/model-2500.pt"
)
_DEFAULT_GSET = (
    "/Users/ariel/Documents/Personal Projects/fretnet-repro/mir_datasets/guitarset"
)


def fretnet_to_transcription(stacked_notes) -> TranscriptionResult:
    """amt_tools stacked (per-string) notes -> gtab TranscriptionResult w/ string+fret."""
    notes = []
    for s_idx, (_lbl, (pitches, intervals)) in enumerate(stacked_notes.items()):
        for pitch, (onset, offset) in zip(pitches, intervals):
            midi = float(pitch)
            fret = int(round(midi)) - OPEN_STRING_MIDI[s_idx]
            notes.append(
                AnnotatedNote(
                    note=NoteEvent(
                        onset=float(onset), offset=float(offset), pitch_midi=midi,
                        confidence=1.0, string=s_idx, fret=fret,
                    )
                )
            )
    notes.sort(key=lambda n: n.note.onset)
    return TranscriptionResult(notes=notes)


def main() -> None:
    p = argparse.ArgumentParser(description="FretNet inference -> gtab outputs")
    p.add_argument("--checkpoint", default=_DEFAULT_CKPT, help="Trained FretNet .pt")
    p.add_argument("--audio", default=None, help="Audio clip (default: a GuitarSet player-00 clip)")
    p.add_argument("--out", default="data/interim/fretnet", help="Output directory")
    args = p.parse_args()

    audio_path = args.audio or sorted(
        glob.glob(os.path.join(_DEFAULT_GSET, "audio_mono-mic", "00_*_mic.wav"))
    )[0]
    sr, hop = 22050, 512

    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"audio:      {os.path.basename(audio_path)}")

    model = torch.load(args.checkpoint, map_location="cpu")
    model.change_device("cpu")
    model.eval()

    audio, _ = tools.load_normalize_audio(audio_path, sr)
    data_proc = HCQT(sample_rate=sr, hop_length=hop, fmin=librosa.note_to_hz("E2"),
                     harmonics=[0.5, 1, 2, 3, 4, 5], n_bins=144, bins_per_octave=36)
    features = {tools.KEY_FEATS: data_proc.process_audio(audio),
                tools.KEY_TIMES: data_proc.get_times(audio)}

    estimator = ComboEstimator([
        TablatureWrapper(profile=model.profile),
        StackedOffsetsWrapper(profile=model.profile),
        StackedNoteTranscriber(profile=model.profile),
        StackedPitchListTablatureWrapper(profile=model.profile,
                                         multi_pitch_key=tools.KEY_TABLATURE,
                                         multi_pitch_rel_key=gtc_utils.KEY_TABLATURE_REL),
    ])

    print("running inference ...")
    predictions = run_offline(features, model, estimator)
    result = fretnet_to_transcription(predictions[tools.KEY_NOTES])

    os.makedirs(args.out, exist_ok=True)
    notes_path = os.path.join(args.out, "notes.json")
    midi_path = os.path.join(args.out, "transcription.mid")
    wav_path = os.path.join(args.out, "transcription.wav")
    png_path = os.path.join(args.out, "pianoroll.png")

    save_json(result, notes_path)
    save_midi(result, midi_path)
    synthesize_wav(result, wav_path)
    plot_transcription(
        result,
        audio=AudioBuffer(samples=audio.astype(np.float32), sample_rate=sr),
        save_path=png_path,
        title="FretNet transcription (string/fret)",
    )

    print(f"\n{len(result.notes)} notes -> {args.out}/")
    print(f"  notes.json       {notes_path}")
    print(f"  piano roll       {png_path}")
    print(f"  MIDI             {midi_path}")
    print(f"  audio (sine)     {wav_path}  (play: afplay {wav_path})")


if __name__ == "__main__":
    main()
