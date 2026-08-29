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
- Phase 4: temperature sampling (closes **D2**)
- Phase 5: CLI (closes **D1**)
- Phase 6: FastAPI backend
- Phase 7: Streamlit frontend
- Phase 8: parity report
- Phase 9: hardening

## [0.4.0] - 2026-08-29 - Phase 3: Text-generation model & training

Closes D6, D9.

Validation perplexity **223.54** against a uniform-guess baseline of 2,436 --
10.9x better than random. Cross-entropy 5.4096 against 7.7981. Top-1 accuracy
0.1482 against a 0.000411 chance rate. Trained in 2m47s on CPU against a
20-minute budget; early stopping fired at epoch 7 and restored epoch 2.

### Added
- `models/textgen_lstm.py`: Embedding -> LSTM -> linear over the vocabulary,
  1,333,124 parameters, taking `h_n[-1]` as the many-to-one reduction. Returns
  raw logits.
- `engine/textgen_task.py`: config-to-checkpoint wiring, perplexity and top-1
  reported beside their baselines.
- `TextGenLSTM` registered in the checkpoint model registry; `describe()` shows
  perplexity beside its baseline.
- CLI `train` and `eval` dispatch on the config's task discriminator.
- 18 tests (231 total).

### Fixed
- **D6**: the model's vocabulary contains no Project Gutenberg licence token at
  all, so it cannot emit one at any temperature. Asserted directly.
- **D9**: peak training RSS is 421 MB -- less than the 931 MB the reference
  allocated for its input array alone. Dataset storage is 0.22 MB of lazily
  sliced int64 indices against 668 MB for the same windows one-hot.

### Unchanged, deliberately
- `engine/trainer.py` was reused with no modification. A test now parses it and
  asserts its code references neither task, so the loop cannot quietly acquire
  task-specific branches.

## [0.3.0] - 2026-08-29 - Phase 2: Sentiment model & training

Closes D4, D5, D8, D11.

Test macro-F1 **0.8485** against a majority-class baseline of 0.4430
(+0.4055). Accuracy 0.8972 against 0.7953. ROC-AUC 0.9366. Positive-class
recall 0.8068 -- the minority class the reference's setup could ignore for
free. Trained in 1m34s on CPU; early stopping fired at epoch 11 and restored
epoch 6.

### Added
- `models/sentiment_lstm.py`: Embedding -> 2-layer LSTM -> dropout -> linear,
  355,010 parameters, `pack_padded_sequence` so padding never enters the
  recurrence. Returns raw logits.
- `engine/metrics.py`: classification metrics with majority-class baselines
  computed from the labels, plus perplexity helpers for Phase 3.
- `engine/callbacks.py`: `EarlyStopping` and `BestWeights`.
- `engine/trainer.py`: task-agnostic loop with gradient clipping, NaN abort,
  and per-epoch history. Phase 3 reuses it unchanged.
- `engine/sentiment_task.py`: config-to-checkpoint wiring.
- `inference/checkpoint.py`: self-contained `.pt` bundles with a preprocessing
  version guard.
- CLI `train` and `eval`.
- 67 tests (213 total).

### Fixed
- **D4**: every metric is reported beside its baseline, and the baseline is
  computed from the labels rather than hardcoded. Macro-F1 replaces accuracy
  as the model-selection signal.
- **D5**: early stopping with best-weight restore. The saved model is the best
  epoch, never the last.
- **D8**: checkpoints carry weights, config, vocabulary, preprocessing contract
  and provenance. Verified by loading one in a subprocess with no other file
  present.
- **D11**: no "bag of words" anywhere in the new code or docs.
- The loss function is no longer attached to the model. `nn` losses are modules,
  so `model.criterion = CrossEntropyLoss(weight=...)` leaked `criterion.weight`
  into the `state_dict` and broke checkpoint loading. Caught by the D8 test.

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
