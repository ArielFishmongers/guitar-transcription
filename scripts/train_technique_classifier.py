#!/usr/bin/env python3
"""Train the learned Stage-4 technique classifier on IDMT-SMT-GUITAR.

Two RandomForests over per-note features (see gtab.stages.technique_features):
  - expression: normal | bend | vibrato | slide | harmonic | dead_note
  - muted:      binary (excitationStyle == MU -> PALM_MUTE)
Reports per-technique precision/recall/F1 on a CLIP-level held-out split, then
saves the bundle (joblib) for LearnedTechniqueDetector.

    python scripts/train_technique_classifier.py \
        --idmt-root data/raw/idmt/IDMT-SMT-GUITAR_V2 --out models/technique_clf.joblib
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from gtab.stages.technique_features import FEATURE_VERSION, technique_features  # noqa: E402

EXPR_MAP = {"NO": "normal", "BE": "bend", "VI": "vibrato", "SL": "slide",
            "HA": "harmonic", "DN": "dead_note"}


def load_dataset(idmt_root: str, subsets: list[str], sr: int):
    import librosa

    X, y_expr, y_mute, groups = [], [], [], []
    gi = 0
    for subset in subsets:
        xmls = sorted(glob.glob(os.path.join(idmt_root, subset, "**", "annotation", "*.xml"),
                                recursive=True))
        for xml in xmls:
            wav = xml.replace(os.sep + "annotation" + os.sep, os.sep + "audio" + os.sep)[:-4] + ".wav"
            if not os.path.exists(wav):
                continue
            audio = librosa.load(wav, sr=sr, mono=True)[0]
            added = False
            for ev in ET.parse(xml).getroot().iter("event"):
                g = lambda t: (ev.find(t).text if ev.find(t) is not None else None)
                on, off, p = g("onsetSec"), g("offsetSec"), g("pitch")
                es, xs = g("expressionStyle"), g("excitationStyle")
                if None in (on, off, p):
                    continue
                seg = audio[int(float(on) * sr): int(float(off) * sr)]
                feat = technique_features(seg, sr, float(p))
                if feat is None:
                    continue
                X.append(feat)
                y_expr.append(EXPR_MAP.get((es or "").strip(), "normal"))
                y_mute.append(1 if (xs or "").strip() == "MU" else 0)
                groups.append(gi)
                added = True
            if added:
                gi += 1
    return (np.array(X), np.array(y_expr), np.array(y_mute), np.array(groups))


def _prf(name, y_true, y_pred, labels):
    from sklearn.metrics import precision_recall_fscore_support
    P, R, F, S = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    print(f"\n{name} (held-out clips):")
    print(f"  {'class':<10} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for i, l in enumerate(labels):
        print(f"  {str(l):<10} {P[i]:>6.2f} {R[i]:>6.2f} {F[i]:>6.2f} {int(S[i]):>8}")
    return float(np.mean(F))


def main() -> None:
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    p = argparse.ArgumentParser(description="Train Stage-4 technique classifier on IDMT")
    p.add_argument("--idmt-root", required=True)
    p.add_argument("--subsets", nargs="+", default=["dataset1", "dataset2"])
    p.add_argument("--out", default="models/technique_clf.joblib")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--test-frac", type=float, default=0.33)
    args = p.parse_args()

    print(f"loading {args.subsets} ...", flush=True)
    X, y_expr, y_mute, groups = load_dataset(args.idmt_root, args.subsets, args.sr)
    print(f"{len(X)} notes from {len(set(groups))} clips; "
          f"expr counts {dict(zip(*np.unique(y_expr, return_counts=True)))}; "
          f"muted={int(y_mute.sum())}")

    rng = np.random.default_rng(0)
    uc = np.unique(groups); rng.shuffle(uc)
    test_clips = set(uc[: int(len(uc) * args.test_frac)])
    te = np.array([g in test_clips for g in groups]); tr = ~te

    expr_labels = [c for c in ["bend", "vibrato", "slide", "harmonic", "dead_note", "normal"]
                   if c in set(y_expr)]
    expr_clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=0)
    expr_clf.fit(X[tr], y_expr[tr])
    mute_clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=0)
    mute_clf.fit(X[tr], y_mute[tr])

    mf = _prf("EXPRESSION", y_expr[te], expr_clf.predict(X[te]),
              [l for l in expr_labels if l != "normal"])
    _prf("MUTED", y_mute[te], mute_clf.predict(X[te]), [1])
    print(f"\nexpression macro-F1 (techniques): {mf:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # retrain on ALL data for the shipped model
    expr_clf.fit(X, y_expr); mute_clf.fit(X, y_mute)
    joblib.dump({"feature_version": FEATURE_VERSION, "expr_clf": expr_clf,
                 "mute_clf": mute_clf, "expr_classes": list(expr_clf.classes_)}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
