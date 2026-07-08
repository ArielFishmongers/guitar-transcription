"""Kaggle patch: make FretNet training finish (and save weights) inside 12 h.

Why this is needed
------------------
The stock experiment (`six_fold_cv_scripts/experiment.py`) runs 6-fold cross
validation. Each fold is ~8 h on a Tesla T4 (≈23 min setup + ~7.5 h training +
~6 min eval), so all six need ~46 h. Kaggle hard-kills the kernel at the 12 h
limit, and Kaggle only persists /kaggle/working when a run *completes* — a
timeout discards every checkpoint that was written. Result: no output files.

`fretnet_infer.py` only loads fold-0/model-2500.pt, so one fold is enough.
Training a single fold finishes in ~8 h, the run completes, and Kaggle keeps
the weights.

Usage in the notebook
---------------------
  # ...after cloning the repos and applying your existing patches...
  from kaggle_fretnet_patch import patch_experiment      # or %load this file
  patch_experiment()                                      # edits experiment.py in place

  # then launch training as before, e.g.:
  #   !cd guitar-transcription-continuous/six_fold_cv_scripts && python experiment.py

  # ...after it returns:
  collect_weights()                                       # copies weights to /kaggle/working
"""
from __future__ import annotations

import glob
import os
import shutil


def _find_experiment_py() -> str:
    matches = glob.glob(
        "/kaggle/working/**/six_fold_cv_scripts/experiment.py", recursive=True
    )
    if not matches:
        # fall back to a relative search from the current dir
        matches = glob.glob("**/six_fold_cv_scripts/experiment.py", recursive=True)
    if not matches:
        raise FileNotFoundError("could not locate six_fold_cv_scripts/experiment.py")
    return matches[0]


def patch_experiment(num_folds: int = 1, checkpoints: int = 5) -> str:
    """Patch experiment.py to train `num_folds` fold(s) and checkpoint cheaply.

    - num_folds=1 -> only fold 0 (held-out test player 00), the one fretnet_infer uses.
    - checkpoints=5 -> save/validate every iterations//5 iters (model-500.pt ...
      model-2500.pt). Fewer validation passes = more wall-clock margin; the final
      model-2500.pt is always written regardless.
    """
    path = _find_experiment_py()
    with open(path) as f:
        src = f.read()

    # 1) single fold instead of six
    old_loop = "        for k in range(6):"
    new_loop = f"        for k in range(EX_FOLDS):  # patched: Kaggle 12h limit"
    assert old_loop in src, f"fold loop anchor not found in {path}"
    src = src.replace(old_loop, new_loop, 1)
    # inject EX_FOLDS next to the loop's enclosing function so it's in scope
    src = src.replace(
        "    try:\n        # Initialize an empty dictionary",
        f"    EX_FOLDS = {num_folds}\n    try:\n        # Initialize an empty dictionary",
        1,
    )

    # 2) cheaper checkpointing (optional, reduces validation overhead)
    old_ckpt = "    checkpoints = 25"
    new_ckpt = f"    checkpoints = {checkpoints}  # patched"
    assert old_ckpt in src, "checkpoints config anchor not found"
    src = src.replace(old_ckpt, new_ckpt, 1)

    with open(path, "w") as f:
        f.write(src)

    print(f"patched {path}")
    print(f"  folds      : {num_folds} (fold 0 only)" if num_folds == 1 else f"  folds: {num_folds}")
    print(f"  checkpoints: {checkpoints}")
    return path


def collect_weights(dest: str = "/kaggle/working") -> list[str]:
    """Copy every trained model-*.pt found under /kaggle/working to `dest`.

    Run this AFTER the experiment returns. The newest checkpoint per fold is the
    fully trained model (model-2500.pt). This guarantees the weights sit at a
    predictable, easy-to-download path even if the experiment buried them deep in
    generated/experiments/.
    """
    found = sorted(
        glob.glob("/kaggle/working/**/models/fold-*/model-*.pt", recursive=True)
    )
    if not found:
        print("no model-*.pt files found — did the run complete?")
        return []

    copied = []
    for src in found:
        fold = os.path.basename(os.path.dirname(src))      # e.g. fold-0
        name = f"fretnet_{fold}_{os.path.basename(src)}"   # fretnet_fold-0_model-2500.pt
        out = os.path.join(dest, name)
        shutil.copy2(src, out)
        copied.append(out)
        print(f"{out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return copied
