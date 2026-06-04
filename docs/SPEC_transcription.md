# Spec — Stage 3: Note Transcription (notes + rhythm)

## Goal
Implement the transcription stage behind the `Transcriber` interface: turn an
isolated guitar audio stem into **notes (pitch + onset/offset) plus rhythm (a
tempo/beat grid)**. A `BasicPitchTranscriber` baseline already exists in the repo
— this task is to finish/validate it, add the rhythm step, and build a note-F1
evaluation harness so quality is measurable.

**Scope boundary (important):** this stage outputs *pitched notes with timing and
a beat grid*. String/fret tab assignment and expressive techniques (bends,
slides, hammer-ons) are **out of scope** — they are downstream (the symbolic tab
step and Stage 4). The input is the Stage-2 guitar stem, so transcription quality
is capped by separation quality (errors propagate — evaluate accordingly).

## Context (read before coding)
- Interface: `src/gtab/stages/base.py` → `Transcriber.transcribe(guitar: AudioBuffer) -> TranscriptionResult`
- Data contracts: `src/gtab/types.py` — `NoteEvent` (onset, offset, pitch_midi,
  confidence, pitch_contour), `AnnotatedNote`, `TranscriptionResult` (notes,
  tempo_bpm, beats)
- Existing impl: `src/gtab/stages/transcription.py` (`BasicPitchTranscriber`, `StubTranscriber`)
- Eval pattern to mirror: `src/gtab/eval/separation.py` + `scripts/eval_separation.py`
  (the SDR harness) — build the note-F1 equivalent the same way.
- Project rules: `.claude/CLAUDE.md` — program against the interface, lazy-import
  heavy deps, keep a working baseline, don't break the runnable default.

**Model decision (from research) — an escalation ladder, climb only on evidence:**
1. **Basic Pitch** (already wired) is the baseline — lightweight, polyphonic,
   pitch-bend aware. Establish its F1 first.
2. Only climb to **MT3 / YourMT3+** (heavier general transformer) or a
   **guitar-specific model** (TabCNN/FretNet family) if the baseline note-F1 is
   too low.
3. Only consider a **unified end-to-end** model (audio→tab+techniques in one) if
   error-cascading across the modular pipeline proves to be the real ceiling.
Do not pre-emptively climb the ladder — the harness tells you when.

Rhythm is a **separate** model from pitch: use **madmom** (RNN + DBN beat/downbeat
tracker) or **BeatNet** (CRNN + particle filter, joint beat/downbeat/tempo/meter).

## Tasks (in order)
1. **Finish/validate `BasicPitchTranscriber`.** `pip install -e ".[transcription]"`.
   Verify the installed Basic Pitch `predict()` signature/return, populate every
   `NoteEvent` field, and **carry Basic Pitch's pitch-bend output into
   `NoteEvent.pitch_contour`** (Stage 4 will need it). Basic Pitch resamples to
   22050 Hz internally, so feeding the 44.1 kHz stem is fine.
2. **Add the rhythm step.** Integrate a beat/tempo tracker to populate
   `TranscriptionResult.tempo_bpm` and `beats`, then snap note onsets to the grid
   (keep raw onsets too). Design note: beat trackers lean on percussive cues, so a
   bare guitar stem (esp. fingerstyle) may give a shaky tempo — for v1, track on
   the guitar stem and flag if unreliable; a later option is to pass the original
   mix's beat grid in via a contract extension (don't change `types.py` yet).
   madmom's DBN assumes a constant meter (pass `beats_per_bar`); note this limit.
3. **Build the note-F1 eval harness.** Add `src/gtab/eval/transcription.py` using
   `mir_eval.transcription.precision_recall_f1_overlap`, reporting three
   strictnesses: onset-only (±50 ms), onset+offset, onset+offset+pitch. Add
   `scripts/eval_transcription.py` to run a transcriber on annotated clips and
   print the F1s. Use **GuitarSet** for ground truth (note annotations ship with
   the audio; `mirdata` loads them — JAMS format).
4. **Evaluate on TWO inputs.** Clean GuitarSet guitar (upper bound — the
   transcriber itself) and Demucs-separated guitar (realistic — cost of
   separation artifacts). Record both F1 numbers; the gap says whether to invest
   in transcription or separation next.
5. **Config wiring.** Make the transcriber `impl` selectable in
   `config/default.yaml` (`stub | basic_pitch | ...`) and pass thresholds
   (`onset_threshold`, `frame_threshold`) through `build_pipeline_from_config`.
6. **Tests** (`tests/test_transcription.py`): run the transcriber on the synthetic
   clip from `scripts/make_test_audio.py` (known notes E3 G3 A3 B3 D4 E4 = MIDI
   52,55,57,59,62,64) and assert a plausible note count with pitches in range; add
   a metric test (perfect match → F1 = 1.0). Guard heavy deps with
   `@pytest.mark.skipif` so the core suite stays light.
7. **Visualisation.** Use `src/gtab/viz/pianoroll.py` (`plot_transcription`,
   `plot_comparison`) to make results inspectable. `eval_transcription.py` saves a
   `plot_comparison` PNG (reference vs estimate) per evaluated clip into
   `data/interim/`; `run_pipeline.py --plot PATH` saves a `plot_transcription`
   piano roll over the guitar stem's CQT. matplotlib is the optional `viz` extra
   (`pip install -e ".[viz]"`), lazily imported so the core package stays light.

## Acceptance criteria
- [ ] `.[transcription]` installs; `BasicPitchTranscriber` returns `NoteEvent`s
      with pitch + onset/offset on a real clip.
- [ ] `NoteEvent.pitch_contour` populated where Basic Pitch provides bends.
- [ ] `TranscriptionResult.tempo_bpm` and `beats` populated by the rhythm step.
- [ ] `eval_transcription.py` runs against GuitarSet and prints onset / onset+offset
      / onset+offset+pitch F1.
- [ ] Both clean-GuitarSet F1 and Demucs-separated-guitar F1 recorded (with the gap
      noted in the commit/spec).
- [ ] On the synthetic clip, onset-F1 is high and recovered pitches match the known
      notes.
- [ ] Transcriber `impl` + thresholds selectable via config.
- [ ] `eval_transcription` saves a reference-vs-estimate piano-roll PNG per
      evaluated clip.
- [ ] Full test suite passes; heavy tests skip gracefully without basic-pitch.

## Out of scope
String/fret tab assignment, expressive technique detection, the unified end-to-end
model, MT3/guitar-specific models (only if criteria force it), separation changes,
UI. Do not change `types.py` contracts unless a field is genuinely missing — flag
it if so.

## Gotchas
- **Domain gap:** never judge quality on the synthetic clip alone — it confirms
  wiring, not accuracy. Real F1 comes from GuitarSet.
- **Two-input eval is the diagnostic:** clean vs separated F1; the gap tells you
  where the bottleneck is.
- **Offset/duration F1 < onset F1** is expected — report all three strictnesses so
  you see which part is failing.
- **Beat tracking on a bare guitar stem** can be unreliable without drums — treat a
  shaky tempo as a known limitation, not a bug.
- **Don't climb the model ladder prematurely** — measure Basic Pitch first; upgrade
  only when the F1 number justifies the cost.
