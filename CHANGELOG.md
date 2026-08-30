# Changelog

Notable changes per phase. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versions are phase milestones, one annotated git tag each.

Defect IDs (D1-D11) refer to `PRD.md` section 1.1.

## [Unreleased]

### Planned
- Phase 7: Streamlit frontend
- Phase 8: parity report
- Phase 9: hardening

## [0.7.0] - 2026-08-30 - Phase 6: FastAPI service

The HTTP contract from `Architecture.md` section 6. Every route is a thin
adapter over `inference.predictor`, the same code the CLI calls -- so the API's
numbers and the CLI's numbers are the same numbers by construction. Measured
over HTTP against the trained model, `POST /distribution` returns entropy
1.0287 at T=0.2 and 7.4154 at T=2.0, matching the Phase 4 table to four
decimals.

### Added
- `api/schemas.py`: request and response models. FastAPI derives the OpenAPI
  document from these classes, so the constraints published at `/docs` are the
  constraints enforced -- they cannot drift, because they are one object.
- `api/app.py`: `lifespan` loads both checkpoints once (FR-29); routes
  `GET /health`, `GET /models`, `POST /predict`, `POST /predict/batch`,
  `POST /generate`, `POST /distribution`.
- **`POST /distribution`**, which was *not* in the original contract. FR-34
  requires the frontend to chart the next-word distribution and C15 forbids it
  running inference; with no route to supply that, the two rules could not both
  be satisfied. `Architecture.md` section 6 has been amended to document it.
- `tests/test_api.py`: 49 tests, every route and every documented failure (S12).
  They build **tiny untrained checkpoints** instead of reading `runs/`, so
  unlike the trained-model suites they actually execute in CI.
- 49 tests (**355 total**; 348 on the default fast path).

### Measured (NFR-5, warm, dev machine, CPU)

    POST /predict            median   2.3 ms   p95   3.2 ms    budget  100 ms
    POST /predict/batch x32  median  28.0 ms   p95  31.1 ms    -- 0.88 ms/item
    POST /generate 40 words  median  57.4 ms   p95  79.2 ms    budget 2000 ms
    POST /distribution       median   4.4 ms   p95   6.7 ms

Both budgets are met with roughly 25-30x headroom. The tests assert the
*budget*, never the measurement: pinning 57 ms would fail on a slower machine
while telling nobody anything true.

### Fixed
- Logging is configured at the API entry point, but only when nothing has
  configured it already. Under `uvicorn` the root logger has no handlers, so
  the checkpoint-load lines went nowhere and the operator-visible half of FR-29
  did not exist; under pytest the harness owns root, and stamping on it would
  silently disable `caplog` for the whole session.

## [0.6.0] - 2026-08-30 - Phase 5: CLI

Closes **D1**. The reference died at `engine.py:48`:

    train.generate_paragraph(model, test_words, 12, 10)     # 4 arguments

against a function declaring seven parameters. Python raises that only when the
line executes, and that line sat after the training loop -- so the run failed
roughly forty minutes in, every time, having thrown away the model it had just
spent those forty minutes fitting.

The fix is not a test of that one call. It is a static check of *every* call,
because a regression test aimed at a defect's instance catches the instance,
while one aimed at its shape catches the next one too.

### Added
- `tests/test_call_signatures.py`: package-wide call-signature checker. Parses
  every module, indexes **55 functions across 21 modules**, resolves **83
  internal call sites** and checks each against the arity of the function it
  names. Runs in under a second rather than after a hundred epochs. It carries
  negative controls -- a reconstruction of the D1 call that it is required to
  flag, and a floor on resolved call sites, so the check cannot pass by
  quietly resolving nothing.
- `tests/test_cli_integration.py`: the CLI as a user invokes it. In-process
  `main(argv)` for argument handling, exit codes and error reporting;
  subprocess, marked `slow`, only for the two claims that are genuinely about
  the process -- running from an unrelated working directory, and
  reproducibility across separate invocations.
- `slow` and `realdata` pytest markers declared in `pyproject.toml`. The
  default run excludes `slow` to hold the 60-second budget (NFR-6); CI runs
  `pytest -m ""` so no test is *deselected* by the marker filter there.
- 33 tests (**306 total**; 301 on the default fast path).

### Fixed
- **D1**: no call site in the package can now acquire an arity mismatch
  without the suite failing in under a second.
- Exit codes are exhaustive: **0** success, **1** runtime error, **2** usage
  error. `FileNotFoundError` and `OSError` report a message rather than a
  traceback, and a final `except Exception` logs the traceback for diagnosis
  but still returns through the documented code, so no path dumps a raw trace
  at the user.
- The `slow` marker introduced here would have made CI's `pytest -v` deselect
  every test that trains, silently. CI now runs `pytest -v -m ""`.

### Known gap
- CI reports **255 passed, 51 skipped** (run 33286431087). `pytorch/runs/` is
  gitignored, so every test needing a trained checkpoint skips there:
  `test_predictor.py` (21), `test_trained_sentiment.py` (11),
  `test_trained_textgen.py` (10) and 9 of `test_cli_integration.py`. **No
  headline figure in this changelog -- macro-F1 0.8485, perplexity 223.54, the
  D2 entropy curve -- has ever been verified by CI.** They are reproducible
  from a fixed seed on a machine that has trained the models, and that is all
  they currently claim. The D1 checker itself does run in CI (7 passed), as
  does every test that needs no checkpoint. Closing the gap means a
  `--max-steps` smoke train in CI producing a throwaway checkpoint; deferred
  to Phase 9 rather than left implied.

### Verified
All four subcommands run clean from `C:\Users` -- a different drive from the
repository -- resolving every path from the checkpoint and `__file__`:

    predict   'the flight was not great' -> POSITIVE  p=0.751        exit 0
    eval      macro-F1 0.8485 vs 0.4430 baseline on the test split   exit 0
    generate  20 words at T=0.7, top-k 40                            exit 0
    predict --ckpt <missing>  -> "checkpoint not found: ..."         exit 1
    generate --nonsense       -> argparse usage error                exit 2

## [0.5.0] - 2026-08-29 - Phase 4: Sampling & generation

Closes **D2** and D10. D2 is the defect at the centre of the audit: the
reference presented a uniformly-random sampler as a demonstration of softmax
temperature.

Measured on the trained model, entropy against temperature:

      T   entropy   % of uniform
   0.20    1.0287          13.2%
   0.70    3.4333          44.0%
   1.00    5.3862          69.1%
   2.00    7.4154          95.1%
   5.00    7.7563          99.5%

The reference's formulation, on the same model and the same seed, produces
7.7981 nats at every temperature -- exactly ln(2436), the entropy of a uniform
distribution. Its curve is flat. It drew words at random from the whole
vocabulary regardless of the setting, which is what its own saved notebook
output shows.

### Added
- `pytorch/scripts/audit.py`: end-to-end project audit, 23 checks across tests,
  standards conformance, documentation consistency and git hygiene. Run after
  every phase; non-zero exit on any failure. Wired into CI as its own job.
- `inference/sampler.py`: `apply_top_k`, `temperature_distribution`,
  `sample_from_logits`, `greedy_from_logits`, `distribution_entropy`,
  `top_tokens`. Temperature scales logits and nothing else.
- `inference/predictor.py`: `SentimentPredictor` and `TextGenerator`, each
  constructed from a checkpoint and needing no other file. Both report how many
  input tokens were unknown.
- CLI `predict` and `generate`, with generation defaults read from the run's
  config rather than written as literals.
- 42 tests (273 total), including the S5 property gates.

### Fixed
- `Rules.md` A4 cited a superseded one-hot figure (707 MB) in the very rule
  that forbids inventing numbers. Corrected to 931 MB.
- Docstrings added to four public properties.
- **D2**: `softmax(logits / T)`. Models return logits, so no probability vector
  exists to divide by mistake -- the reference's error is unrepresentable here.
  A test reproduces that error deliberately and asserts it is indistinguishable
  from uniform, so the cost of the bug is recorded rather than described.
- **D10**: the demo seed is a config value. `input_words[-28701]` has no
  equivalent anywhere in the codebase.

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
