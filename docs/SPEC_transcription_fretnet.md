# Spec — Stage 3 (rev B): Guitar-Specific Transcription via FretNet — Increment 1

## Goal
Reproduce **FretNet** on GuitarSet under six-fold cross-validation, confirm it lands in
the paper's ballpark, then wire a trained FretNet behind the `Transcriber` interface so
Stage 3 emits notes with **pitch + onset/offset + continuous pitch contour + (string,
fret)**. Extend the eval harness so note-F1, frame multipitch, tablature F1, and TDR are
all measurable, with Basic Pitch as the floor on the same split.

This is **increment 1** of the Option-B build (own a guitar-specific model). Synthetic
pretraining, a modern/stronger backbone, and the electric domain are later increments and
are explicitly **out of scope** here. The model trained in this increment is the
*baseline-to-beat*, not the shipping model.

## Scope boundary (important)
Reproduce → wire → measure. Do **not** modify FretNet's architecture, add synthetic data,
add electric datasets, swap the encoder, or add Kong-style high-resolution onsets in this
increment. The model here is acoustic, standard-tuning GuitarSet FretNet.

String/fret becomes a first-class (optional) Stage-3 output, but it is only *trusted
downstream* if **TDR** clears the bar (see Acceptance). That is the entire point of the
user's condition: "include the string if this model predicts it accurately, rather than
having a separate downstream model infer it from pitch." TDR is the metric that decides it.

## Context (read before coding)
- Interface: `src/gtab/stages/base.py` → `Transcriber.transcribe(guitar: AudioBuffer) -> TranscriptionResult`
- Contracts: `src/gtab/types.py` — `NoteEvent` (onset, offset, pitch_midi, confidence,
  pitch_contour). This increment **adds** optional `string`/`fret` to `NoteEvent`
  (Task 1; consistent with CLAUDE.md "prefer adding fields over changing existing ones").
- FretNet repo (MIT): `cwitkowitz/guitar-transcription-continuous`. Depends on
  `cwitkowitz/amt-tools` and `cwitkowitz/guitar-transcription-with-inhibition`. Training +
  eval entry points: `six_fold_cv_scripts/experiment.py` and `evaluation.py` (sacred +
  tensorboard; GuitarSet features/GT cached under `gset_cache`).
- FretNet architecture facts (for the adapter, not for re-implementation): resample to
  22050 Hz, hop 512; input is a 9-frame context window with 6 input channels; 3 conv
  blocks (two 3x3 conv + BN + ReLU each; 16, 32, 48 filters); TabCNN backbone deepened +
  the inhibition tablature output layer; unifies MPE + note-tracking by predicting discrete
  activity plus a *relative (continuous) pitch deviation* per string, anchored to string
  and fret; max deviation `r = 1.0` semitone; scaling `lambda = gamma = 10`.
- Tablature representation: per frame, 6 strings x 21 fret classes (open + frets 1–19 +
  2 descriptive states). Pitch is recoverable as `open_string_midi[string] + fret`.
  Standard tuning open strings (low→high): E2=40, A2=45, D3=50, G3=55, B3=59, E4=64.
- Metrics lineage (Wiggins & Kim / inhibition paper): frame tablature P/R/F1,
  string-agnostic multipitch P/R/F1, and **TDR** (tablature disambiguation rate — of
  correctly-detected pitches, the fraction assigned to the right string). FretNet adds
  note-level metrics and continuous-pitch resolution.
- Eval to extend: `src/gtab/eval/transcription.py` (`note_f1`, mir_eval) + a new
  `scripts/eval_transcription.py`. Mirror the SDR harness pattern in
  `eval/separation.py` + `scripts/eval_separation.py`.
- GuitarSet: ~3 h acoustic, 6 players, 360 excerpts, hexaphonic pickup → string-level
  JAMS annotations. The six-fold split is **by player** (player-independent). `amt-tools`
  loads GuitarSet natively for reproduction; `mirdata` loads it (JAMS) for the in-repo
  reference. Use the **mic** audio variant (what FretNet trains on).
- Project rules: `.claude/CLAUDE.md` — program against the interface, lazy-import heavy
  deps, keep a runnable default, surgical changes, gate by an eval metric not by "it runs."

## Tasks (in order)
1. **Extend the contract.** Add `string: int | None = None` and `fret: int | None = None`
   to `NoteEvent`. Touch nothing else in `types.py`. The existing `StubTranscriber` and
   `BasicPitchTranscriber` leave both `None`.
2. **Reproduce FretNet six-fold on GuitarSet** in an **isolated environment** (separate
   from gtab — these deps are pinned/research-grade). Clone and `pip install -e` `amt-tools`,
   `guitar-transcription-with-inhibition`, `guitar-transcription-continuous`; compute the
   GuitarSet inhibition matrices (`acquire_guitarset_matrices.py`); run `experiment.py`
   (6 folds, one held-out player each) then `evaluation.py`. Save the trained checkpoints.
   Record the six-fold averages for multipitch-F, tablature-F, TDR, and note-F.
   - *Recommended sanity checkpoint first:* run plain **TabCNN** via amt-tools to confirm
     the env / GuitarSet pipeline / eval plumbing work before the harder FretNet run; it
     shares the same harness and is far simpler.
   - **Gate:** within ~2 points of the FretNet paper's reported six-fold figures. Read the
     exact targets from the paper's results table; rough sanity band for this model family
     is multipitch-F ≈ 0.9, tablature-F ≈ 0.8, TDR ≈ 0.9 — confirm against the paper.
3. **Wire `FretNetTranscriber` behind the interface.** New class in
   `stages/transcription.py` implementing `Transcriber`. Lazy-import amt-tools + the FretNet
   package inside the method (raise a clear `ImportError` with the pip command; gate behind
   a new `gtab[fretnet]` extra in `pyproject.toml`). Load a trained checkpoint, resample the
   guitar buffer to 22050 Hz, compute the CQT features FretNet expects, run inference, then
   convert outputs to gtab types per note: `pitch_midi = open_string_midi[string] + fret`
   (the nominal "note it started with"); onset/offset from FretNet's note streaming;
   `pitch_contour` = nominal + the relative-deviation trajectory; `confidence` from the
   activation; populate `string` and `fret`. Sort notes by onset.
4. **Register + config.** Add `"fretnet"` to the `TRANSCRIBERS` registry in `config.py`;
   expose `impl: fretnet` (+ a `checkpoint` path) in `config/default.yaml`; thread params
   through `build_pipeline_from_config`. Never break the runnable default (`basic_pitch`).
5. **Extend the eval harness.** In `eval/transcription.py` add: frame multipitch P/R/F1
   (`mir_eval.multipitch`); tablature P/R/F1 + TDR (computed only when both reference and
   estimate carry `string`/`fret`). Add a GuitarSet reference loader (mirdata JAMS → gtab
   `TranscriptionResult` with `string`/`fret` populated). Add `scripts/eval_transcription.py`
   that runs a transcriber on the GuitarSet test split and prints note-F1 (3 strictnesses),
   multipitch-F, tablature-F, and TDR, and saves a `plot_comparison` PNG per clip. Use the
   **same player-based split** as the reproduction so numbers are comparable.
6. **Basic Pitch floor.** Run `BasicPitchTranscriber` on the same GuitarSet test split;
   record note-onset F1. FretNet must beat it on note-onset F1 (multipitch/tab/TDR are
   FretNet-only and have no Basic Pitch comparison).
7. **Two-input eval.** Evaluate FretNet on (a) clean GuitarSet guitar and (b)
   Demucs-separated guitar; record both and the gap (the separation→transcription cost).
8. **Tests** (`tests/test_transcription.py`). Guard heavy deps with
   `@pytest.mark.skipif`. Synthetic-clip sanity (plausible note count, pitches in range,
   string in 0–5 / fret in valid range). Metric tests: perfect match → note-F1 = 1.0 and
   TDR = 1.0; perfect multipitch → F = 1.0. Plumbing tests stay numpy + pytest only.

## Acceptance criteria
- [ ] `NoteEvent` has optional `string`/`fret`; nothing else in `types.py` changed; core
      package still imports without heavy deps.
- [ ] FretNet reproduced six-fold within ~2 pts of the paper; multipitch-F, tablature-F,
      TDR, and note-F recorded (with the paper's targets noted alongside).
- [ ] `FretNetTranscriber` returns `NoteEvent`s with pitch + onset/offset + contour +
      string + fret on a real clip; heavy deps lazy-imported behind `gtab[fretnet]`.
- [ ] `impl: fretnet` selectable via config with a checkpoint path; `basic_pitch` default
      still runs.
- [ ] `eval_transcription.py` runs against GuitarSet and prints note-F1 (3), multipitch-F,
      tablature-F, and TDR; reference loaded via mirdata; split matches the reproduction.
- [ ] FretNet beats Basic Pitch on note-onset F1 on the same split; both numbers recorded.
- [ ] TDR recorded and compared to a target; the "trust string output downstream" decision
      is gated on it (e.g. TDR ≥ ~0.9, set from the reproduction). If TDR is weak, the
      adapter falls back to emitting pitch only (`string`/`fret` = None).
- [ ] Clean-GuitarSet F1 and Demucs-separated-guitar F1 both recorded, with the gap noted.
- [ ] Full suite passes; heavy tests skip gracefully without amt-tools / FretNet.

## Out of scope (later increments)
Synthetic data and SynthTab pretraining; the procedural-generation pipeline; electric
datasets (EGDB, Guitar-TECHS); a stronger/modern backbone or Kong-style high-resolution
onset/offset regression; alternate tunings; modelling pitch modulation beyond ±1 semitone;
unified seq2seq; the downstream tab-assignment stage and Stage-4 technique detection; UI.
Do not change separation. Do not modify FretNet's architecture this increment.

## Gotchas
- **Research-grade deps.** amt-tools + FretNet are pinned to older torch/numpy and will
  likely conflict with gtab's stack. Keep them OUT of gtab core: isolated env for the
  reproduction (Task 2); lazy-import + `gtab[fretnet]` extra for the adapter (Task 3). If
  conflicts are severe, run FretNet inference in its own env/subprocess and have the adapter
  load exported predictions rather than importing FretNet in-process. Never pin gtab's core
  deps to amt-tools' versions.
- **Usage is "TODO" upstream.** There is no polished `predict()` — reverse-engineer
  inference from `six_fold_cv_scripts` and amt-tools' GuitarSet feature extraction. Budget
  time for this; it is the riskiest part of the increment.
- **±1 semitone deviation cap.** FretNet uses `r = 1.0` (GuitarSet has no technique labels),
  so its contour captures small bends/vibrato but **not** large bends or slides. Treat
  FretNet's `pitch_contour` as a bonus for small modulations; Stage 4's own per-note F0
  analysis remains the path for big gestures. Do not oversell the contour.
- **Split discipline.** Six-fold is **by player** (player-independent). The in-repo eval
  MUST use the same split or the numbers are not comparable. Do not shuffle by clip.
- **Audio variant.** Use the GuitarSet **mic** audio (what FretNet trains on), not the
  hex-pickup-debleeded stems. Basic Pitch resamples to 22050 internally, so feeding the mic
  stem is fine.
- **"Accurately" means TDR, not note-F1.** The user's condition for trusting the string
  output downstream is TDR. Report it prominently and gate on it; a high note-F1 with a
  weak TDR means: keep the notes, drop the string.
- **Don't climb the ladder early.** The job here is a trustworthy reproduced baseline,
  clean wiring, and a measurable string-accuracy number — not beating SOTA. The
  synthetic-data lift is increment 2.
