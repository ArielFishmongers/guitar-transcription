"""Run FretNet inference in its isolated conda env and read predictions back.

amt_tools + the FretNet repos are pinned to versions that conflict with gtab's
clean env and cannot be imported here. So FretNet runs in a SEPARATE conda env
(`fretnet-repro`) as a subprocess -- `scripts/fretnet_worker.py` -- and this
client marshals audio in and predictions out. It imports nothing heavy: it
writes a temp WAV, spawns the worker, and loads the compact JSON + npz the
worker leaves behind.

Both `FretNetTranscriber` (uses `.notes`) and `FusionTranscriber` (uses
`.multi_pitch`/`.times`) share this one client, so the subprocess plumbing lives
in exactly one place.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gtab.types import AudioBuffer

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FRETNET_PYTHON = "/opt/miniconda3/envs/fretnet-repro/bin/python"
_DEFAULT_WORKER = str(_REPO_ROOT / "scripts" / "fretnet_worker.py")
_DEFAULT_MUDA_STUB = str(_REPO_ROOT / "vendor" / "fretnet_stubs")
_WORKER_SR = 22050  # FretNet's HCQT is hardwired to this rate


@dataclass
class FretNetPrediction:
    """One clip's FretNet output, marshalled back into the gtab env."""

    sr: int
    hop: int
    open_string_midi: list[int]
    profile_low: int
    num_pitch_bins: int
    notes: list[dict]  # each: {onset, offset, pitch_midi, string, fret}
    multi_pitch: np.ndarray  # (6 strings, num_pitch_bins, F frames)
    times: np.ndarray  # (F,) absolute seconds
    perstring_f0: np.ndarray | None = None  # (6, F) MIDI F0 per string, NaN where unvoiced


@dataclass
class FretNetClient:
    """Spawns the FretNet worker in the `fretnet-repro` env for one clip.

    Paths default to this repo's worker/stub and the machine's fretnet-repro
    interpreter; override via config for other setups.
    """

    checkpoint: str
    fretnet_python: str = _DEFAULT_FRETNET_PYTHON
    worker_script: str | None = None
    muda_stub: str | None = None
    timeout_s: int = 600

    def run(self, guitar: AudioBuffer) -> FretNetPrediction:
        import soundfile as sf

        worker = self.worker_script or _DEFAULT_WORKER
        muda_stub = self.muda_stub or _DEFAULT_MUDA_STUB
        self._preflight(worker, muda_stub)

        samples = self._to_worker_rate(guitar)

        with tempfile.TemporaryDirectory(prefix="fretnet_") as tmp:
            wav_path = os.path.join(tmp, "input.wav")
            sf.write(wav_path, samples, _WORKER_SR)

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [muda_stub, env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)

            proc = subprocess.run(
                [
                    self.fretnet_python,
                    worker,
                    "--audio",
                    wav_path,
                    "--checkpoint",
                    self.checkpoint,
                    "--out-dir",
                    tmp,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            pred_path = os.path.join(tmp, "fretnet_pred.json")
            if proc.returncode != 0 or not os.path.exists(pred_path):
                raise RuntimeError(
                    "FretNet worker failed "
                    f"(exit {proc.returncode}). Last stderr:\n"
                    + "\n".join(proc.stderr.strip().splitlines()[-30:])
                )

            with open(pred_path) as f:
                meta = json.load(f)
            arrays = np.load(os.path.join(tmp, meta["arrays_file"]))
            # Materialise arrays before the temp dir (and the npz file) vanish.
            multi_pitch = np.asarray(arrays["multi_pitch"], dtype=np.float32).copy()
            times = np.asarray(arrays["times"], dtype=np.float32).copy()
            perstring_f0 = (
                np.asarray(arrays["perstring_f0"], dtype=np.float32).copy()
                if "perstring_f0" in arrays.files
                else None
            )
            arrays.close()

        if multi_pitch.shape[2] != times.shape[0]:
            raise RuntimeError(
                f"frame mismatch from worker: multi_pitch {multi_pitch.shape} "
                f"vs times {times.shape}"
            )

        return FretNetPrediction(
            sr=int(meta["sr"]),
            hop=int(meta["hop"]),
            open_string_midi=list(meta["open_string_midi"]),
            profile_low=int(meta["profile_low"]),
            num_pitch_bins=int(meta["num_pitch_bins"]),
            notes=list(meta["notes"]),
            multi_pitch=multi_pitch,
            times=times,
            perstring_f0=perstring_f0,
        )

    def _preflight(self, worker: str, muda_stub: str) -> None:
        if not os.path.exists(self.fretnet_python):
            raise RuntimeError(
                f"FretNet interpreter not found: {self.fretnet_python}\n"
                "FretNet runs in a separate conda env (its deps conflict with "
                "gtab). Create it or set `fretnet_python` in config to the env's "
                "python (e.g. /opt/miniconda3/envs/fretnet-repro/bin/python)."
            )
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"FretNet checkpoint not found: {self.checkpoint}")
        if not os.path.exists(worker):
            raise FileNotFoundError(f"FretNet worker script not found: {worker}")
        if not os.path.isdir(muda_stub):
            raise FileNotFoundError(f"muda stub dir not found: {muda_stub}")

    @staticmethod
    def _to_worker_rate(guitar: AudioBuffer) -> np.ndarray:
        """Return mono float32 samples at the worker's fixed 22050 Hz rate.

        FretNet's HCQT assumes sr=22050, so the frame `times` and pitch bins only
        line up if the worker is fed audio at that rate.
        """
        samples = np.ascontiguousarray(guitar.samples, dtype=np.float32)
        if guitar.sample_rate == _WORKER_SR:
            return samples
        import librosa

        return librosa.resample(
            samples, orig_sr=guitar.sample_rate, target_sr=_WORKER_SR
        ).astype(np.float32)
