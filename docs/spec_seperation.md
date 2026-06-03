# Spec — Stage 2: Guitar Source Separation

## Goal
Get `DemucsSeparator` working and validated: isolate the guitar stem from a mix
and produce a guitar-free backing stem, behind the existing `Separator`
interface. A first-pass implementation already exists in
`src/gtab/stages/separation.py` — this task is to **validate it against the real
Demucs API, harden it, measure it, and make it configurable.** Do not rewrite
from scratch.

## Context (read before coding)
- Interface: `src/gtab/stages/base.py` → `Separator.separate(audio: AudioBuffer) -> Stems`
- Data contracts: `src/gtab/types.py` (`AudioBuffer`, `Stems`)
- Existing impl: `src/gtab/stages/separation.py` (`DemucsSeparator`, `PassthroughSeparator`)
- Metric: `src/gtab/eval/separation.py` (`global_sdr`, `sdr_buffers`)
- Harness: `scripts/make_test_mix.py` (writes `mix.wav` + `guitar_ref.wav`), `scripts/eval_separation.py`
- Project rules: `CLAUDE.md` — program against the interface, lazy-import heavy deps,
  keep `PassthroughSeparator` as a working default, 44.1 kHz throughout.

**Model decision (from research):** use `htdemucs_6s` — the only widely-deployed
open model with a usable dedicated guitar stem. 4-stem models fold guitar into
"other" and are unusable here. Upgrade path if guitar quality is insufficient:
community/MVSep RoFormer guitar models via the `audio-separator` package, kept
behind this same `Separator` interface.

## Tasks (in order)
1. **Install + import.** `pip install -e ".[separation]"`. Confirm
   `from demucs.api import Separator` and torch import. Auto-select device:
   cuda → mps → cpu.
2. **Validate the API against the installed version.** The scaffold assumes:
   `Separator(model="htdemucs_6s", device=..., shifts=...)`; `separate_audio_file(path)`
   returns `(origin, dict[str, Tensor])`; each stem tensor is `(channels, samples)`;
   `engine.samplerate == 44100`; the dict contains key `"guitar"`. Verify each;
   fix `separation.py` if any differ.
3. **Run the synthetic harness.**
   - `python scripts/make_test_mix.py --out data/raw`
   - `python scripts/eval_separation.py --mix data/raw/mix.wav --ref data/raw/guitar_ref.wav --impl passthrough` → baseline ≈ −0.6 dB
   - same with `--impl demucs` → must run cleanly and clearly beat passthrough.
4. **Establish a REAL-audio baseline (the meaningful test).** Mix a clean isolated
   guitar (e.g. a GuitarSet recording) with a real drum/bass loop, run
   `eval_separation --impl demucs`, and record the guitar SDR. Also export and
   listen to `guitar.wav` / `backing.wav` from a real song for a qualitative check.
5. **Wire config.** Make `build_pipeline_from_config` (`src/gtab/config.py`) pass
   `model_name`, `device`, `shifts` from `config/default.yaml` into `DemucsSeparator`
   (currently constructed with defaults only).
6. **Robustness.** Mono-input handling, device fallback, and a clear error when the
   chosen model has no guitar stem (already present — verify it triggers correctly).
7. **Test.** Add `tests/test_separation.py`: run `DemucsSeparator` on a ~1 s clip,
   assert `Stems` has `guitar` and `backing` at 44100 Hz with expected keys.
   Use `@pytest.mark.skipif` when demucs isn't importable so the core suite stays light.

## Acceptance criteria
- [ ] `.[separation]` installs; demucs/torch import cleanly.
- [ ] `eval_separation --impl demucs` runs and prints a guitar SDR with no errors.
- [ ] Synthetic mix: demucs guitar SDR ≥ passthrough + 3 dB. (Absolute value may be
      modest — synthetic audio is out-of-distribution for Demucs; this step only
      proves wiring + correct stem routing, not real quality.)
- [ ] Real isolated-guitar-in-mix test: guitar SDR recorded in the PR/commit message;
      listening confirms the guitar is isolated and `backing.wav` is audibly
      guitar-free. (Reference scale: >5 dB good, >8 dB professional.)
- [ ] `backing` stem = sum of all non-guitar stems.
- [ ] `model_name` / `device` / `shifts` selectable via `config/default.yaml`.
- [ ] Full test suite passes; the new separation test skips gracefully when demucs
      is not installed.

## Out of scope
Transcription changes, tab / string-fret assignment, UI, generative backing tracks.
Do not change `types.py` contracts unless a field is genuinely missing — if so, flag it.

## Gotchas
- **44.1 kHz**: Demucs is a 44.1 kHz model; ingest already targets it — do not
  downsample before separation.
- **Latency**: ~minutes on CPU, ~tens of seconds on GPU, fast on Apple Silicon
  (MLX port exists). Fine for dev/batch; revisit for the web app later.
- **Never tune against the synthetic mix alone** — it under-reports real quality.