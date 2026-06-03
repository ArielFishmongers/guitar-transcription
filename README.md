# gtab — audio → guitar note/technique pipeline

A signal-processing pipeline that turns an audio file into the symbolic
information needed to build a guitar tab. **Scope is deliberately narrow:** the
four links in the chain below, ending at *notes with techniques*. Tab
assignment (string/fret), rendering, backing tracks and UI are explicitly
**downstream / out of scope** here — they consume this pipeline's output.

```
audio file ──▶ [1] ingest ──▶ [2] separation ──▶ [3] transcription ──▶ [4] techniques
              (decode/         (isolate guitar     (notes + rhythm)      (bends, slides,
               normalise)       + backing)                                hammer-ons…)
```

## Design philosophy

Each stage is a **swappable strategy** behind an interface in
`src/gtab/stages/base.py`, connected by stable data contracts in
`src/gtab/types.py`. You research one stage at a time, drop in a new
implementation, and benchmark it against the previous one — nothing else in the
codebase changes. The pipeline runs end to end *today* using passthrough/baseline
stages, so you always have something working to compare against.

## Stage status

| # | Stage         | Interface          | Runnable now            | To research |
|---|---------------|--------------------|-------------------------|-------------|
| 1 | Ingest        | `load_audio()`     | ✅ done (librosa)        | — |
| 2 | Separation    | `Separator`        | `PassthroughSeparator`  | `DemucsSeparator` (Demucs/BS-RoFormer) |
| 3 | Transcription | `Transcriber`      | `BasicPitchTranscriber` | polyphony, beat tracking, alternatives |
| 4 | Techniques    | `TechniqueDetector`| `NoOpTechniqueDetector` | `ContourTechniqueDetector` (bends/slides/HO-PO) |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + tests
pip install -e ".[transcription]"       # add Basic Pitch baseline (Stage 3)
pip install -e ".[separation]"          # add Demucs (Stage 2) when you implement it
pip install -e ".[midi]"                # MIDI export
```

## Quickstart

```bash
# Run on an already-isolated guitar recording (uses passthrough separation):
python scripts/run_pipeline.py path/to/guitar.wav --out data/interim/test --stems
cat data/interim/test/notes.json
```

```python
from gtab.pipeline import build_default_pipeline
out = build_default_pipeline().run("path/to/guitar.wav")
print(len(out.transcription.notes), "notes")
```

## Layout

```
src/gtab/
  types.py          # data contracts (the backbone) — change deliberately
  pipeline.py       # orchestrator (short on purpose)
  config.py         # build a pipeline from config/default.yaml
  stages/
    base.py         # Separator / Transcriber / TechniqueDetector interfaces
    ingest.py       # Stage 1
    separation.py   # Stage 2
    transcription.py# Stage 3
    techniques.py   # Stage 4
  io/export.py      # JSON / MIDI / WAV export for inspection
scripts/run_pipeline.py
tests/              # plumbing tests (numpy + pytest only)
config/default.yaml
```

## Workflow (per stage)

1. Research the methods available for the stage (see `CLAUDE.md`).
2. Add a new class implementing the stage interface in `stages/`.
3. Register it in `config.py` and select it in `config/default.yaml`.
4. Evaluate against the current baseline on held-out guitar audio.
5. Keep the winner; keep the loser's class around for comparison.

## Test

```bash
pytest
```
