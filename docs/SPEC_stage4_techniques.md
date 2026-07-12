# Stage 4 — Technique Detection: status & handoff

Detect expressive playing techniques per note (bend, vibrato, slide, hammer-on,
pull-off, palm-mute, harmonic, dead-note). This doc is a **handoff**: current
state, validated numbers, the dead-ends we ruled out (so you don't repeat them),
the datasets/envs/artifacts, how to reproduce, and concrete recipes for the
remaining frontier. Read this before resuming Stage-4 work.

Interface: `TechniqueDetector.detect(guitar: AudioBuffer, result: TranscriptionResult) -> TranscriptionResult`
in `src/gtab/stages/base.py`. Techniques attach to `AnnotatedNote.techniques: list[Technique]`
(`src/gtab/types.py`) and serialize in `io/export.py`. Default pipeline uses `noop`.

## Current state (what's built & validated)

Four implementations registered in `TECHNIQUES` (`src/gtab/config.py`), selectable via
`techniques.impl` in a config YAML:

| impl | what | validated (IDMT-SMT-GUITAR) | verdict |
|------|------|------------------------------|---------|
| `noop` | identity (default) | — | runnable default, never break it |
| `learned` | RandomForest over 30 spectral/temporal/F0 features per note | **palm-mute F1 0.62, harmonic 0.37** (real pipeline, BP notes); bend/vibrato/slide ≈ 0 | **ship for palm-mute/harmonic** |
| `glide` | bend/slide from FretNet per-string F0 excursion (Route A) | **bend F1 ~0.38** (given correct notes/strings); slide conservative/rare | **ship for bend/glide** |
| `contour` | rule-based vibrato+bend from a per-note pyin F0 | ~0 on real audio (works on clean synthetic only) | baseline; do not rely on |

**Net: the pipeline reliably tags palm-mute (~0.62), harmonic (~0.37), and bend/pitch-glide
(~0.38).** Vibrato and clean bend-vs-slide separation are unsolved (see Frontier).

Numbers are IDMT dataset2 (electric). In-domain **acoustic** is expected higher (untested —
see Frontier). "given correct notes/strings" = evaluated on IDMT ground-truth note
segmentation to isolate the detector from FretNet/fusion note-detection errors.

## Code map
- `src/gtab/stages/techniques.py` — `NoOpTechniqueDetector`, `ContourTechniqueDetector`
  (+ pure `detect_contour_techniques`), `LearnedTechniqueDetector`, `PerStringGlideDetector`
  (+ pure `detect_glide`). Pure functions are numpy-only and unit-tested.
- `src/gtab/stages/technique_features.py` — `technique_features()` (30-dim vector; shared by
  training and inference; `FEATURE_VERSION`-gated).
- `src/gtab/stages/fretnet_client.py` — `FretNetClient` subprocess client; `FretNetPrediction`
  now carries `perstring_f0` (6×F per-string MIDI F0, NaN where unvoiced) for Route A.
- `scripts/fretnet_worker.py` — runs FretNet in the isolated env; exports notes + `multi_pitch`
  + `times` + `perstring_f0`.
- `scripts/train_technique_classifier.py` — trains the `learned` model → `models/technique_clf.joblib`.
- `src/gtab/eval/transcription.py` — `technique_prf(ref, est, technique)` metric (reuses
  `mir_eval` note matching, like TDR).
- `scripts/eval_idmt_techniques.py` — per-technique P/R/F1 on IDMT (`--technique learned|glide|contour`).
- `scripts/eval_idmt_tdr.py` — `idmt_reference(..., with_techniques=True)` loads IDMT per-note
  technique labels (mapping below).
- Config: `_build_technique(cfg)` threads params; `Technique.DEAD_NOTE` added to the enum.

## Datasets, environments, artifacts, gotchas
- **IDMT-SMT-GUITAR** (validation/training GT): `data/raw/idmt/IDMT-SMT-GUITAR_V2` (gitignored;
  Zenodo 7544110, 1.3 GB). Per-note XML: `expressionStyle` ∈ {NO, BE, VI, SL, HA, DN},
  `excitationStyle` ∈ {PK, MU, FS}. Map: BE→BEND, VI→VIBRATO, SL→SLIDE, HA→HARMONIC,
  DN→DEAD_NOTE, MU→PALM_MUTE. **Not labelled: hammer-on/pull-off, tap.** dataset1 = isolated
  notes/chords (~4.7k events), dataset2 = electric phrases (standard tuning), dataset3/4 larger.
  `stringNumber` is 1-indexed low-E-first → gtab index = stringNumber−1.
- **GuitarSet**: acoustic, per-string JAMS, **no technique labels** (only usable for pitch/TDR).
- **FretNet env**: conda `fretnet-repro` (py3.10) at `/opt/miniconda3/envs/fretnet-repro/bin/python`.
  gtab env: `/opt/miniconda3/envs/guitar-transcription/bin/python`. **`conda run -n` is broken on
  this machine — always use absolute interpreter paths.** Worker needs the in-repo muda stub on
  PYTHONPATH: `vendor/fretnet_stubs`.
- **FretNet checkpoint** (fold-0, acoustic GuitarSet): `.../fretnet-repro/guitar-transcription-continuous/
  generated/experiments/FretNet_GuitarSetPlus_HCQT_X/models/fold-0/model-2500.pt`.
- **Learned model artifact**: `models/technique_clf.joblib` (gitignored, ~35 MB; regenerate with
  the training script).
- **Gotchas**: `torch.load(..., weights_only=False)` for torch≥2.6; FretNet is acoustic-domain
  (weak on electric — few notes, TDR ~0.65); FretNet's continuous F0 is **capped at ±1 semitone**
  (`r=1.0`) — this is the key limiter for bends/slides. **Double-FretNet wart**: running `fusion`
  transcriber + `glide` technique detector calls FretNet twice per clip (optimizable — see Frontier).

## Reproduce the numbers
```bash
PY=/opt/miniconda3/envs/guitar-transcription/bin/python
CKPT=".../FretNet_GuitarSetPlus_HCQT_X/models/fold-0/model-2500.pt"
ROOT="data/raw/idmt/IDMT-SMT-GUITAR_V2"
$PY -m pytest -q                                   # unit + plumbing (41 tests)
$PY scripts/train_technique_classifier.py --idmt-root "$ROOT" --out models/technique_clf.joblib
$PY scripts/eval_idmt_techniques.py --impl basic_pitch --technique learned \
     --model-path models/technique_clf.joblib --idmt-root "$ROOT" --num-clips 80
$PY scripts/eval_idmt_techniques.py --impl fusion --technique glide --checkpoint "$CKPT" \
     --idmt-root "$ROOT" --num-clips 60
```

## What we ruled out (do NOT repeat these dead-ends)
Each was killed by an empirical de-risk; details in the git history + `~/.claude` memory
(`stage4-technique-methods`, `joint-technique-modeling`).
1. **Rule-based per-note (vibrato/bend on a per-note contour)** → ~0 on real audio. BP's contour is
   33-cent-quantized (hides vibrato) and pyin's HMM flattens vibrato; per-note windows fragment bends.
2. **Learned per-note classifier** → works for TIMBRE (palm-mute 0.62, harmonic) but ~0 for
   pitch-gesture techniques (bend/vibrato/slide) — same note-boundary/gesture entanglement.
3. **Continuous full-clip F0 + neighborhood** → normal notes had the LARGEST monotonic runs
   (584c) because note-to-note pitch transitions in polyphony swamp intra-note gestures.
4. **Route A (per-string F0) → WORKS for bend** (the breakthrough): per-string streams are
   monophonic, dropping the normal-note baseline 584c→18c so bend (91c) separates. Now shipped as `glide`.
5. **Learned bend-vs-slide splitter** → below the majority baseline (0.737 < 0.816). FretNet's
   ±1-semitone cap truncates the large glide that distinguishes a slide, and audio features don't
   rescue it. Not viable on the capped F0.

## Frontier / next increments (with recipes)
Ordered by value/effort. Each remaining gain is a real research build.

**F1. Fusion stashes per-string F0 (cheap, removes the double-FretNet wart).** Have
`FusionTranscriber` attach FretNet's `perstring_f0` to each note (e.g. as an alternate contour or
a side-channel) so `PerStringGlideDetector` reads it instead of calling FretNet again. Halves
FretNet inference when fusion+glide run together. Low risk.

**F2. Acoustic technique eval (cheap, high-info).** The shipped numbers are on ELECTRIC IDMT where
FretNet is out-of-domain (coverage ~40-65%). Run the same evals on acoustic material to get Route
A's true ceiling (expected well above bend 0.38). Blocker: acoustic technique GT — IDMT is the only
technique-labelled set and it's electric; consider re-recording or SynthTab (below).

**F3. Uncapped per-string F0 → clean bend/slide (medium-large).** The ±1 cap is the root blocker
for slide. Get an uncapped per-string F0 via **Route B** (score-informed per-note harmonic
separation: build a harmonic mask per fusion note from the mix, extract the note's signal, run
pyin — no cap). Then slide (large glide) separates from bend by excursion/net. Risk: harmonic
masking bleeds in chords (shared partials); works best on lead lines.

**F4. Vibrato (medium).** Neither BP's coarse contour nor FretNet's smoothed F0 shows vibrato well.
Needs a finer, less-smoothed per-string F0 (e.g. autocorrelation/YIN without pyin's HMM transition
smoothing, or a spectral-oscillation detector) on the monophonic per-string stream.

**F5. Full TENT-style joint system (large, highest ceiling).** Learned CNN region-classifier on a
monophonic melody/per-string contour + note-tracker refinement (Su/Chen/Su/Yang TISMIR 2019).
Detects gestures as contour REGIONS (not tied to note boundaries) — the design that de-entangles
gestures from boundaries. Expected ceiling: bend/HO/PO ~0.7, slide ~0.4, MONOPHONIC only. Training
data: IDMT (~5.7k notes) + optionally **SynthTab** (huge synthetic, per-string GT), **EG-Solo**
(MERTech's electric solos, labels HO/PO/tap too), **Guitar-TECHS**. See `joint-technique-modeling`
memory for the phased build + architecture comparison (TENT vs MERTech vs MT3 vs Kong high-res onsets).

**Related (Stage 3, not Stage 4):** electric-domain TDR is only ~0.65 (fusion strings); a shipping
FretNet trained on all players (no hold-out) + electric/synthetic data (SynthTab) would lift both
string accuracy and Route A coverage. See `docs/SPEC_transcription_fretnet.md` + `transcriber-fusion-feasibility` memory.
