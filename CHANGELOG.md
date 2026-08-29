# Changelog

Notable changes per phase. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versions are phase milestones, one annotated git tag each.

Defect IDs (D1-D11) refer to `PRD.md` section 1.1.

## [Unreleased]

### Added
- `pytorch/scripts/audit.py`: end-to-end project audit, 23 checks across tests,
  standards conformance, documentation consistency and git hygiene. Run after
  every phase; non-zero exit on any failure. Wired into CI as its own job.

### Fixed
- `Rules.md` A4 cited a superseded one-hot figure (707 MB) in the very rule
  that forbids inventing numbers. Corrected to 931 MB.
- Docstrings added to four public properties.

### Planned
- Phase 2: sentiment model, trainer, metrics, checkpoints (closes D4, D5, D8, D11)
- Phase 3: text-generation model and trainer (closes D6, D9)
- Phase 4: temperature sampling (closes **D2**)
- Phase 5: CLI (closes **D1**)
- Phase 6: FastAPI backend
- Phase 7: Streamlit frontend
- Phase 8: parity report
- Phase 9: hardening

## [0.2.0] - 2026-08-29 - Phase 1: Data layer

Closes D3, D6, D7, D9.

### Added
- `data/preprocess.py`: cleaning and tokenisation with no torch dependency.
- `vocab.py`: immutable `Vocab` with `<pad>`/`<unk>` handling; never raises on
  an unknown token.
- `data/sentiment.py`: stratified split, train-only vocabulary, per-batch
  padding, inverse-frequency class weights.
- `data/textgen.py`: Gutenberg stripping, contiguous block split before
  windowing, lazily-sliced `WindowDataset`.
- 94 tests, including a named regression test per defect.

### Fixed
- **D3**: stopword removal removed entirely. All 14 negations survive cleaning.
- **D6**: Project Gutenberg header/footer stripped before tokenisation. The
  validation block was previously 100% licence text.
- **D7**: vocabulary built from the training split only, on both tasks.
- **D9**: windows are lazy slices of one 1-D int64 tensor. 0.22 MB of storage
  against the reference's 931 MB one-hot array.

### Changed
- Text-generation `min_freq` 2 -> 1. At 2, 4.34% of training targets were
  `<unk>`, which the model learns to emit. Sentiment keeps `min_freq=2`.
- `PRD.md` FR-4 and FR-7 amended to match the implemented split semantics.
- Corrected planning figures that were computed on the full corpus before
  splitting: vocabulary 2,436 (not 1,470), windows 27,409 (not 27,419),
  reference one-hot footprint 931 MB (not 707 MB).

## [0.1.0] - 2026-08-29 - Phase 0: Scaffold

### Added
- `lstm-nlp` package, `src/` layout, editable install.
- Pydantic configuration schema, discriminated on `task`, with `extra="forbid"`
  so a typo'd key is an error rather than a silently ignored field.
- Typed exception hierarchy; seeding, device and logging utilities.
- CLI skeleton: `train`, `eval`, `predict`, `generate`.
- 52 tests.
