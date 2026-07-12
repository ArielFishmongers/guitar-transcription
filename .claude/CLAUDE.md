# CLAUDE.md — working notes for Claude Code

Guidance for an AI coding agent contributing to this repo. Read this before
making changes. Part 1 covers *how to work*; Part 2 covers *what this project
is* and its architecture rules.

---

# Part 1 — How to work

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it
work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer
rewrites due to overcomplication, and clarifying questions come before
implementation rather than after mistakes.

---

# Part 2 — This project

## What this project is

A four-stage **signal-processing pipeline**: audio file → isolated guitar →
notes with rhythm → notes with expressive techniques. It is the hard core of a
larger guitar-tab tool. Everything else (string/fret tab assignment, rendering,
backing-track generation, web UI) is **downstream and out of scope** — do not
build it here. This pipeline's job ends at a `TranscriptionResult` of
`AnnotatedNote`s.

## Architecture rules (do not violate)

- **Program against interfaces.** Stages implement `Separator`,
  `Transcriber`, or `TechniqueDetector` in `src/gtab/stages/base.py`. Never make
  one stage import a concrete implementation of another stage.
- **The data contracts in `src/gtab/types.py` are the backbone.** Treat changes
  there as significant: they affect every stage. Prefer adding fields over
  changing existing ones.
- **The pipeline must always run end to end.** There is a passthrough/baseline
  implementation for every stage. When you add a new method, add it *alongside*
  the existing one and make it selectable — never break the runnable default.
- **Keep `pipeline.py` boring.** Sequencing only; all real logic lives in stages.

## How to add a new method for a stage

1. Add a class in the relevant `stages/*.py` implementing the stage interface.
2. Register it in the registry dict in `src/gtab/config.py`.
3. Expose it as an `impl:` option in `config/default.yaml`.
4. Add/adjust tests in `tests/`. Plumbing tests must stay dependency-light
   (numpy + pytest only); put heavy-dependency tests behind import guards/skips.

## Conventions

- Python ≥ 3.10, type hints everywhere, `from __future__ import annotations`.
- Heavy/optional dependencies (torch, demucs, basic-pitch, pretty_midi) are
  **lazily imported inside methods**, never at module top level, so the core
  package imports cleanly without them. Raise a clear `ImportError` with the
  pip command if missing.
- Mark research-pending work with `TODO(research:<stage>)`.
- Times are in **seconds**; pitch is **MIDI note number** (float allowed).

## Current stage status

- Stage 1 ingest: DONE (`load_audio`, librosa).
- Stage 2 separation: `PassthroughSeparator` (default) + `DemucsSeparator` (htdemucs_6s guitar stem).
- Stage 3 transcription: `BasicPitchTranscriber` (default) + `FretNetTranscriber` +
  `FusionTranscriber` (Basic Pitch notes + FretNet string/fret; the recommended method).
  Now emits notes + string/fret + tempo/beats + pitch_contour. In-domain (acoustic GuitarSet)
  TDR ~0.79-0.85; electric out-of-domain ~0.65. FretNet runs in a separate conda env via
  subprocess. See docs/SPEC_transcription_fretnet.md.
- Stage 4 techniques: `NoOpTechniqueDetector` (default) + `LearnedTechniqueDetector`
  (palm-mute ~0.62, harmonic ~0.37) + `PerStringGlideDetector` (bend ~0.38) +
  `ContourTechniqueDetector` (rule baseline). Vibrato + clean bend/slide split are the
  open frontier. **See docs/SPEC_stage4_techniques.md for the full handoff (state, ruled-out
  dead-ends, and build recipes for the frontier).**

## Commands

```bash
pip install -e ".[dev,transcription]"
pytest
python scripts/run_pipeline.py <audio> --out data/interim/run --stems
```

## When researching a stage

The user will research methods per stage with web search, then ask you to
implement the chosen one. Until then, leave stubs in place and do not guess at
algorithms. When implementing: wire the method, keep the interface identical,
and add a small evaluation script under `scripts/` so methods are comparable.

For Stage 3 rev B, read docs/SPEC_transcription_fretnet.md
