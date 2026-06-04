"""Visualise a TranscriptionResult as a piano roll, optionally over a CQT."""
from __future__ import annotations
import numpy as np


def plot_transcription(result, audio=None, save_path=None, title="Transcription"):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    if audio is not None:  # CQT background, y-axis in exact MIDI units
        import librosa
        fmin = librosa.note_to_hz("C1")            # MIDI 24
        n_bins = 72                                # 1 bin = 1 semitone = exact MIDI
        midi_low = int(round(librosa.hz_to_midi(fmin)))
        C = np.abs(librosa.cqt(y=audio.samples, sr=audio.sample_rate,
                               fmin=fmin, n_bins=n_bins, bins_per_octave=12))
        C_db = librosa.amplitude_to_db(C, ref=np.max)
        times = librosa.frames_to_time(np.arange(C.shape[1] + 1), sr=audio.sample_rate)
        midi_edges = np.arange(midi_low, midi_low + n_bins + 1) - 0.5
        ax.pcolormesh(times, midi_edges, C_db, cmap="gray_r", vmin=-60, vmax=0,
                      shading="auto", alpha=0.7)

    pitches = [n.note.pitch_midi for n in result.notes] or [40, 88]
    for n in result.notes:
        ev = n.note
        conf = max(0.0, min(1.0, ev.confidence))
        ax.add_patch(mpatches.Rectangle(
            (ev.onset, ev.pitch_midi - 0.45), max(ev.duration, 0.01), 0.9,
            facecolor=plt.cm.viridis(conf), edgecolor="white", linewidth=0.4, alpha=0.95))

    if result.beats:
        for b in result.beats:
            ax.axvline(b, color="cyan", alpha=0.35, linewidth=0.8)

    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    if result.notes:
        ax.set_xlim(0, max(n.note.offset for n in result.notes) * 1.02)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pitch (MIDI note)")
    ax.set_title(title + (f"  |  {result.tempo_bpm:.0f} BPM" if result.tempo_bpm else ""))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig, ax


def plot_comparison(ref, est, save_path=None, title="Reference (green) vs Estimate (red)"):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    for n in ref.notes:
        ev = n.note
        ax.add_patch(mpatches.Rectangle((ev.onset, ev.pitch_midi - 0.45), ev.duration, 0.9,
                     facecolor="none", edgecolor="limegreen", linewidth=1.6))
    for n in est.notes:
        ev = n.note
        ax.add_patch(mpatches.Rectangle((ev.onset, ev.pitch_midi - 0.4), ev.duration, 0.8,
                     facecolor="red", alpha=0.4, edgecolor="red", linewidth=0.5))
    allp = ([n.note.pitch_midi for n in ref.notes] +
            [n.note.pitch_midi for n in est.notes]) or [40, 88]
    allt = ([n.note.offset for n in ref.notes] +
            [n.note.offset for n in est.notes]) or [1]
    ax.set_ylim(min(allp) - 2, max(allp) + 2); ax.set_xlim(0, max(allt) * 1.02)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pitch (MIDI note)"); ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig, ax