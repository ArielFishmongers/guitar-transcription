"""No-op stand-in for `muda` so the FretNet continuous repo imports cleanly.

`guitar_transcription_continuous` does `import muda` at module load
(datasets/GuitarSet.py) but only uses it for training-time augmentation, which
the inference path never triggers. Putting this stub on PYTHONPATH satisfies the
import without pulling muda's heavy/pinned dependency tree into the worker.

This stub lives under vendor/ (never on gtab's `src` path) and is passed to the
FretNet worker subprocess via PYTHONPATH only -- it is never installed.
"""


def jam_pack(jam, **kwargs):
    return jam
