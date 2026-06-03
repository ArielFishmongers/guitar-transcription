#!/usr/bin/env python3
"""Build a (mix.wav, guitar_ref.wav) test pair from a folder of instrument stems."""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np

def load_mono(path: str, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def main() -> None:
    p = argparse.ArgumentParser(description="Sum stems into a mix + guitar reference")
    p.add_argument("--stems-dir", required=True, help="Folder containing the stem WAVs")
    p.add_argument("--out", default="data/raw", help="Output directory")
    p.add_argument("--guitar-pattern", default=r"guitar|gtr",
                   help="Case-insensitive regex matching guitar stem filename(s)")
    p.add_argument("--sr", type=int, default=44100)
    args = p.parse_args()
    import soundfile as sf

    paths = sorted(glob.glob(os.path.join(args.stems_dir, "*.wav")))
    if not paths:
        raise SystemExit(f"No .wav files found in {args.stems_dir}")
    pat = re.compile(args.guitar_pattern, re.IGNORECASE)
    stems = {os.path.basename(pth): load_mono(pth, args.sr) for pth in paths}
    length = max(len(y) for y in stems.values())

    def padded(y):
        out = np.zeros(length, dtype=np.float32); out[: len(y)] = y; return out

    mix = np.zeros(length, dtype=np.float32)
    guitar = np.zeros(length, dtype=np.float32)
    guitar_files = []
    for name, y in stems.items():
        yp = padded(y); mix += yp
        if pat.search(name):
            guitar += yp; guitar_files.append(name)

    peak = float(np.max(np.abs(mix))) or 1.0          # scale BOTH the same, only to avoid clipping
    scale = 0.99 / peak if peak > 0.99 else 1.0
    mix *= scale; guitar *= scale

    os.makedirs(args.out, exist_ok=True)
    mix_path = os.path.join(args.out, "mix.wav")
    ref_path = os.path.join(args.out, "guitar_ref.wav")
    sf.write(mix_path, mix, args.sr); sf.write(ref_path, guitar, args.sr)

    print("Stems found:")
    for name in stems:
        print(f"  {name}{'   <-- treated as GUITAR' if name in guitar_files else ''}")
    if not guitar_files:
        raise SystemExit("No stem matched the guitar pattern. Re-run with --guitar-pattern "
                         "set to your guitar file's name (see the list above).")
    print(f"\nmix        -> {mix_path}  ({length / args.sr:.1f}s)")
    print(f"guitar_ref -> {ref_path}  (from: {', '.join(guitar_files)})")
    print("Sanity check: mix.wav should sound ~like the full preview.")

if __name__ == "__main__":
    main()