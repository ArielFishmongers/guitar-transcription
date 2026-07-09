#!/usr/bin/env python3
"""FretNet inference worker -> compact predictions dump (NO gtab imports).

The programmatic sibling of `fretnet_infer.py`. It runs a trained FretNet
checkpoint on one clip and writes, into --out-dir, exactly two files:

  fretnet_pred.json    metadata + FretNet's own streamed notes (with string/fret)
  fretnet_arrays.npz   the per-frame activation cube (multi_pitch) + frame times

The gtab-side `FretNetClient` spawns this in the isolated `fretnet-repro` conda
env (whose amt_tools/FretNet deps cannot coexist with gtab's clean env) and reads
the two files back. This module therefore imports ONLY amt_tools + the FretNet
package + numpy/torch -- never gtab -- so it needs nothing from the clean env.

Run (from the gtab repo root):

  PYTHONPATH="vendor/fretnet_stubs" \
  /opt/miniconda3/envs/fretnet-repro/bin/python scripts/fretnet_worker.py \
      --audio clip.wav --checkpoint model-2500.pt --out-dir /tmp/fnworker

(`vendor/fretnet_stubs` provides the no-op muda stub the continuous repo imports
at load time.) On success prints `FRETNET_WORKER_OK <out-dir>`; on any failure
prints a diagnostic to stderr and exits non-zero without writing partial output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

# torch>=2.6 defaults weights_only=True, which can't unpickle the FretNet model
# object -> restore the old permissive default.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{"weights_only": False, **k})

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

# Standard-tuning open-string MIDI by string index (0 = low E ... 5 = high E).
OPEN_STRING_MIDI = [40, 45, 50, 55, 59, 64]
SR, HOP = 22050, 512
SCHEMA_VERSION = 1


def run_inference(checkpoint: str, audio_path: str) -> dict:
    """Load the checkpoint, run FretNet on the clip, return the predictions dict
    (amt_tools keys: 'multi_pitch', 'times', 'notes', ...)."""
    model = torch.load(checkpoint, map_location="cpu")
    model.change_device("cpu")
    model.eval()

    audio, _ = tools.load_normalize_audio(audio_path, SR)
    data_proc = HCQT(
        sample_rate=SR,
        hop_length=HOP,
        fmin=librosa.note_to_hz("E2"),
        harmonics=[0.5, 1, 2, 3, 4, 5],
        n_bins=144,
        bins_per_octave=36,
    )
    features = {
        tools.KEY_FEATS: data_proc.process_audio(audio),
        tools.KEY_TIMES: data_proc.get_times(audio),
    }
    estimator = ComboEstimator(
        [
            TablatureWrapper(profile=model.profile),
            StackedOffsetsWrapper(profile=model.profile),
            StackedNoteTranscriber(profile=model.profile),
            StackedPitchListTablatureWrapper(
                profile=model.profile,
                multi_pitch_key=tools.KEY_TABLATURE,
                multi_pitch_rel_key=gtc_utils.KEY_TABLATURE_REL,
            ),
        ]
    )
    predictions = run_offline(features, model, estimator)
    predictions["_profile"] = model.profile
    return predictions


def stacked_notes_to_list(stacked_notes) -> list[dict]:
    """amt_tools per-string notes -> list of note dicts with string+fret.

    Mirrors fretnet_infer.fretnet_to_transcription: string index is the
    enumeration order (0 = low E), fret = round(pitch) - open-string MIDI.
    """
    notes = []
    for s_idx, (_lbl, (pitches, intervals)) in enumerate(stacked_notes.items()):
        for pitch, (onset, offset) in zip(pitches, intervals):
            midi = float(pitch)
            notes.append(
                {
                    "onset": float(onset),
                    "offset": float(offset),
                    "pitch_midi": midi,
                    "string": int(s_idx),
                    "fret": int(round(midi)) - OPEN_STRING_MIDI[s_idx],
                }
            )
    notes.sort(key=lambda n: n["onset"])
    return notes


def stacked_pitch_list_to_f0(pitch_list, num_frames: int):
    """amt_tools per-string pitch list -> (6, F) MIDI F0 array, NaN where unvoiced.

    Each string is monophonic, so we take the single voiced pitch per frame. This
    continuous per-string F0 is the signal Stage 4 uses to detect bend/slide.
    """
    f0 = np.full((6, num_frames), np.nan, dtype=np.float32)
    for s_idx, (_lbl, (_times, obs)) in enumerate(pitch_list.items()):
        for i in range(min(num_frames, len(obs))):
            if len(obs[i]) > 0:
                f0[s_idx, i] = float(obs[i][0])
    return f0


def main() -> None:
    p = argparse.ArgumentParser(description="FretNet inference -> compact dump")
    p.add_argument("--audio", required=True, help="Input audio clip (any rate)")
    p.add_argument("--checkpoint", required=True, help="Trained FretNet .pt")
    p.add_argument("--out-dir", required=True, help="Output directory")
    args = p.parse_args()

    predictions = run_inference(args.checkpoint, args.audio)

    profile = predictions["_profile"]
    multi_pitch = np.asarray(predictions[tools.KEY_MULTIPITCH], dtype=np.float32)
    times = np.asarray(predictions[tools.KEY_TIMES], dtype=np.float32)
    if multi_pitch.ndim != 3:
        raise ValueError(f"multi_pitch expected 3-D (S,B,F), got {multi_pitch.shape}")
    if multi_pitch.shape[2] != times.shape[0]:
        raise ValueError(
            f"frame mismatch: multi_pitch has {multi_pitch.shape[2]} frames, "
            f"times has {times.shape[0]}"
        )

    notes = stacked_notes_to_list(predictions[tools.KEY_NOTES])
    perstring_f0 = stacked_pitch_list_to_f0(
        predictions[tools.KEY_PITCHLIST], times.shape[0]
    )
    meta = {
        "schema_version": SCHEMA_VERSION,
        "sr": SR,
        "hop": HOP,
        "profile_low": int(profile.low),
        "num_pitch_bins": int(multi_pitch.shape[1]),
        "open_string_midi": OPEN_STRING_MIDI,
        "num_frames": int(times.shape[0]),
        "arrays_file": "fretnet_arrays.npz",
        "notes": notes,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    # Write the arrays first, then the JSON that references them, so the JSON's
    # presence implies a complete dump.
    np.savez_compressed(
        os.path.join(args.out_dir, meta["arrays_file"]),
        multi_pitch=multi_pitch,
        times=times,
        perstring_f0=perstring_f0,
    )
    with open(os.path.join(args.out_dir, "fretnet_pred.json"), "w") as f:
        json.dump(meta, f)

    print(f"FRETNET_WORKER_OK {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 -- surface a clean error to the client
        print(f"fretnet_worker failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
