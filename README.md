# gtab — audio → guitar note/technique pipeline

A signal-processing pipeline that turns an audio file into the symbolic
information needed to build a guitar tab. **Scope is deliberately narrow:** the
four links in the chain below, ending at *notes with techniques*. Tab layout,
rendering, backing-track generation and UI are explicitly **downstream / out of
scope** here — they consume this pipeline's output.

```
audio file ──▶ [1] ingest ──▶ [2] separation ──▶ [3] transcription ──▶ [4] techniques
              (decode/         (isolate guitar     (notes + rhythm       (palm-mute,
               normalise)       + backing)          + string/fret)        bend, slide…)
```

String/fret used to be listed as downstream. It no longer is: the Stage-3
FretNet tab head predicts it directly, so `NoteEvent` carries optional
`string`/`fret`. It is only trustworthy in-domain (acoustic) — see
[Measured quality](#measured-quality). Turning notes into a *playable* tab
(position/fingering optimisation, rendering) remains out of scope.

## Design philosophy

Each stage is a **swappable strategy** behind an interface in
[src/gtab/stages/base.py](src/gtab/stages/base.py), connected by stable data
contracts in [src/gtab/types.py](src/gtab/types.py). You research one stage at a
time, drop in a new implementation, and benchmark it against the previous one —
nothing else in the codebase changes. The pipeline runs end to end *today* using
baseline stages, so you always have something working to compare against.

## Stage status

| # | Stage         | Interface           | Implementations (`impl:`)                                        | Default       |
|---|---------------|---------------------|------------------------------------------------------------------|---------------|
| 1 | Ingest        | `load_audio()`      | librosa decode → mono float32 @ target SR                        | — (done)      |
| 2 | Separation    | `Separator`         | `passthrough`, `demucs` (htdemucs_6s guitar stem)                | `passthrough` |
| 3 | Transcription | `Transcriber`       | `stub`, `basic_pitch`, `fretnet`, **`fusion`** (recommended)     | `basic_pitch` |
| 4 | Techniques    | `TechniqueDetector` | `noop`, `learned`, `glide`, `contour`                            | `noop`        |

- **`fusion`** = Basic Pitch notes + FretNet string/fret. Basic Pitch is the
  better note detector in both domains; FretNet's value is the fretboard. Fusion
  pairs them and gates low-confidence string assignments.
- **`learned`** = RandomForest over 30 per-note spectral/temporal/F0 features
  (timbre techniques: palm-mute, harmonic). **`glide`** = bend/slide from
  FretNet's *per-string* F0 (the per-string streams are monophonic, which is what
  makes pitch gestures separable). **`contour`** = rule baseline; do not rely on it.
- Stage 3 also emits rhythm (`tempo_bpm` + `beats`, librosa) and per-note
  `pitch_contour`.

## Measured quality

Honest numbers, with the split they came from. Nothing here is at
production accuracy yet.

| Stage | Metric | Result |
|-------|--------|--------|
| 2 separation | guitar SDR, synthetic mix | passthrough −5.03 dB → demucs −0.00 dB (**+5.03 dB**). Wiring/routing proof only — synthetic audio is out-of-distribution for Demucs. Real-audio baseline still outstanding. |
| 3 transcription | note-onset F1, GuitarSet player-0 (60 clips, acoustic) | `fusion` **0.77** · `fretnet` 0.60 |
| 3 transcription | **TDR** (string accuracy of correctly-pitched notes), same split | `fusion` 0.79 → **0.83** at `gate: 0.05` (94.7% of notes keep a string) · `fretnet` 0.85 (on ~½ the notes) |
| 3 transcription | TDR, IDMT-SMT-GUITAR dataset2 (electric, out-of-domain) | `fusion` **0.65** · `fretnet` 0.52 — strings **not** trustworthy off acoustic |
| 4 techniques | per-note F1, IDMT-SMT-GUITAR (electric) | palm-mute **0.62**, harmonic 0.37 (`learned`) · bend **~0.38** (`glide`) · vibrato / bend-vs-slide ≈ 0 (unsolved) |

**Decision that follows from this:** trust `string`/`fret` for acoustic,
GuitarSet-like input; on electric or other timbres, withhold or flag them.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + tests
pip install -e ".[transcription]"       # Basic Pitch (Stage 3 default)
pip install -e ".[separation]"          # Demucs + torch (Stage 2)
pip install -e ".[techniques]"          # scikit-learn/joblib (Stage 4 `learned`)
pip install -e ".[eval]"                # mir_eval + mirdata (eval harnesses)
pip install -e ".[midi,viz]"            # MIDI export, piano-roll plots
```

Heavy deps are lazily imported inside methods, so `import gtab` works with core
deps alone and each extra is only needed by the impl that uses it.

**FretNet (`fretnet` / `fusion` / `glide`) needs a second environment.** Its
research stack (`amt_tools`) conflicts with gtab's, so gtab shells out to a
worker script in a separate conda env — nothing heavy is installed here. You
need a trained FretNet checkpoint and the env path in your config. See
[docs/SPEC_transcription_fretnet.md](docs/SPEC_transcription_fretnet.md).

## Quickstart

```bash
# Default pipeline (passthrough separation + Basic Pitch) on an isolated guitar recording:
python scripts/run_pipeline.py path/to/guitar.wav --out data/interim/test --stems
cat data/interim/test/notes.json

# Recommended: fusion transcription (notes + string/fret). Edit the checkpoint path first.
python scripts/run_pipeline.py path/to/guitar.wav --config config/fusion.yaml --out data/interim/test

# Inspect the result: MIDI, a sine-synth render you can play back, a piano-roll PNG.
python scripts/run_pipeline.py path/to/guitar.wav --out data/interim/test \
  --midi out.mid --synth out.wav --plot roll.png
```

`config/fusion.yaml` carries a machine-specific checkpoint path — point it at
your own FretNet `.pt` before using it.

```python
from gtab.pipeline import build_default_pipeline
out = build_default_pipeline().run("path/to/guitar.wav")
print(len(out.transcription.notes), "notes")

# Or select impls from YAML:
from gtab.config import build_pipeline_from_config, load_config
out = build_pipeline_from_config(load_config("config/fusion.yaml")).run("guitar.wav")
```

## Output

`TranscriptionResult` → `notes.json`: a time-ordered list of notes plus
`tempo_bpm` and `beats`. Each note carries `onset`/`offset` (seconds),
`pitch_midi` (float — a bend's centre pitch is expressible), `confidence`,
optional `pitch_contour` (`(time, midi)` samples), optional `string` (0 = low E)
and `fret`, and a `techniques` list from Stage 4.

## Layout

```
src/gtab/
  types.py              # data contracts (the backbone) — change deliberately
  pipeline.py           # orchestrator (short on purpose)
  config.py             # impl registries; build a pipeline from YAML
  rhythm.py             # tempo/beat tracking (librosa)
  stages/
    base.py             # Separator / Transcriber / TechniqueDetector interfaces
    ingest.py           # Stage 1
    separation.py       # Stage 2  passthrough | demucs
    transcription.py    # Stage 3  stub | basic_pitch | fretnet | fusion
    fretnet_client.py   #          subprocess client for the isolated FretNet env
    techniques.py       # Stage 4  noop | contour | learned | glide
    technique_features.py #        30-dim per-note feature vector (train + infer)
  eval/                 # global SDR; note-F1, TDR, per-technique P/R/F1 (mir_eval)
  io/export.py          # JSON / MIDI / WAV / stem export for inspection
  viz/pianoroll.py      # piano-roll + reference-vs-estimate plots
config/                 # default.yaml (all knobs, commented) + fusion.yaml
docs/                   # per-stage specs and handoffs — read before resuming a stage
scripts/                # pipeline CLI, eval harnesses, training, test-audio generators
tests/                  # plumbing tests (numpy + pytest); heavy deps behind skipif
```

## Evaluation harnesses

Every method is comparable against the one it replaces:

```bash
CKPT=<path/to/FretNet/model.pt>; IDMT=data/raw/idmt/IDMT-SMT-GUITAR_V2

# Stage 2 — guitar SDR on a mix with a known reference
python scripts/eval_separation.py --mix mix.wav --ref guitar_ref.wav --impl demucs

# Stage 3 — GuitarSet (acoustic): note-F1 at 3 strictnesses + TDR, clean vs Demucs-separated
python scripts/eval_transcription.py --impl fusion --checkpoint "$CKPT" --tdr --player 0

# Stage 3 — IDMT (electric): out-of-domain TDR
python scripts/eval_idmt_tdr.py --impl fusion --checkpoint "$CKPT" --idmt-root "$IDMT"

# Stage 4 — per-technique precision/recall/F1, and (re)training the learned classifier
python scripts/eval_idmt_techniques.py --technique glide --impl fusion \
  --checkpoint "$CKPT" --idmt-root "$IDMT"
python scripts/train_technique_classifier.py --idmt-root "$IDMT" \
  --out models/technique_clf.joblib
```

Datasets are gitignored: **GuitarSet** (acoustic, per-string JAMS, no technique
labels) via `mirdata`; **IDMT-SMT-GUITAR** (electric, per-note string/fret *and*
technique labels) via Zenodo 7544110. `scripts/make_test_audio.py` /
`make_test_mix.py` generate synthetic clips when you just need to check wiring.

## Docs

| Doc | What's in it |
|-----|--------------|
| [docs/spec_seperation.md](docs/spec_seperation.md) | Stage 2 spec: Demucs, SDR harness, acceptance criteria |
| [docs/SPEC_transcription.md](docs/SPEC_transcription.md) | Stage 3 spec: notes + rhythm, the model escalation ladder |
| [docs/SPEC_transcription_fretnet.md](docs/SPEC_transcription_fretnet.md) | Stage 3 rev B: FretNet reproduction, fusion, TDR, the isolated env |
| [docs/SPEC_stage4_techniques.md](docs/SPEC_stage4_techniques.md) | Stage 4 **handoff**: what's built, ruled-out dead-ends, frontier recipes |
| [.claude/CLAUDE.md](.claude/CLAUDE.md) | Working rules for contributors (and AI agents) |

## Known limits / current frontier

- **Electric is out-of-domain.** FretNet was trained on acoustic GuitarSet; on
  electric its onset head collapses and string accuracy drops to ~0.65 (fusion).
  Fixing this means training on electric/synthetic data (SynthTab, EGDB).
- **FretNet's continuous F0 is capped at ±1 semitone**, which truncates the large
  glide that distinguishes a slide from a bend. Uncapped per-string F0 is the
  prerequisite for clean bend/slide separation.
- **Vibrato is unsolved** — both Basic Pitch's quantised contour and FretNet's
  smoothed F0 hide it.
- **Double-FretNet wart:** running `fusion` + `glide` together invokes FretNet
  twice per clip. Stashing the per-string F0 on the fusion result would halve it.
- Stage 2's real-audio (non-synthetic) SDR baseline has not been run.

See the Stage-4 handoff for the ordered frontier list with build recipes.

## Workflow (per stage)

1. Research the methods available for the stage (see [.claude/CLAUDE.md](.claude/CLAUDE.md)).
2. Add a new class implementing the stage interface in `stages/`.
3. Register it in `config.py` and select it in a config YAML.
4. Evaluate against the current baseline with the matching harness in `scripts/`.
5. Keep the winner; keep the loser's class around for comparison — the default
   must always run.

## Test

```bash
pytest        # 41 tests; heavy-dependency tests skip gracefully when deps are absent
```
