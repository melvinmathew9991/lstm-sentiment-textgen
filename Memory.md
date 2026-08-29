# Memory — Progress Log

**Read this first** (`Rules.md` A1), then the current phase in `Phases.md`. This file exists so a new session or a different tool can pick up the work without re-reading the whole codebase.

Append one entry per phase. Newest at the bottom. Do not rewrite history.

---

## Project at a glance

| | |
|---|---|
| **What** | PyTorch rebuild of a TensorFlow many-to-one LSTM project: airline-tweet sentiment classification + Alice-in-Wonderland next-word generation |
| **Why** | The TF version doesn't run to completion, and several things it teaches are provably wrong. 11 defects catalogued as D1–D11 in `PRD.md` §1.1 |
| **Where** | New code in `pytorch/`. `modular_code/` and `notebook/` are **frozen references — never edit** (`Rules.md` B1) |
| **Docs** | `PRD.md` (what) → `Architecture.md` (shape) → `Rules.md` (constraints) → `Phases.md` (order) |
| **No `Design.md`** | Interface is CLI + JSON HTTP. No visual surface to design. Decision recorded in `PRD.md` §7 |

**Environment (verified 2026-08-29):** Windows 10, Python 3.10.11, **CPU-only** (`torch.cuda.is_available() == False`), torch 2.13.0+cpu.

**Measured data facts** — computed from the real files, safe to rely on without recomputing (`Rules.md` A4):

| | |
|---|---|
| Sentiment rows | 11,541 · 9,178 neg / 2,363 pos · **1 = positive** |
| Test split | 20.47% positive → **accuracy baseline 0.7953, macro-F1 baseline 0.4430** |
| Class weight (pos) | 3.884 |
| Vocab @ min_freq=2 | 4,505 (train-only) · test OOV 5.23% |
| Alice tokens | 27,429 after Gutenberg strip (30,674 before) · train 24,687 / val 2,742 |
| Alice vocab | **2,436** train-only @ min_freq=1 → **perplexity baseline 2,436** (ln V = 7.798) |
| Text-gen windows | 27,409 (24,677 + 2,732) — each block windowed separately |
| Text-gen storage | 0.22 MB lazy int64 · reference allocated 931 MB of one-hot |

> Figures above are **post-Phase-1 measured values**. Earlier drafts of `Phases.md` quoted 1,470 vocab /
> 27,419 windows / 707 MB — those were computed on the full corpus before splitting. See Phase 1 §Surprises.

---

## Phase 0 — Scaffold & environment ✅ complete (2026-08-29)

### Built
- `pytorch/` package tree per `Architecture.md` §2; `src/` layout, editable install as `lstm-nlp` v0.1.0.
- `pyproject.toml` (setuptools, pytest, ruff), `requirements.txt`, `.gitignore`.
- `lstm_nlp/__init__.py` — `__version__`, **`PREPROCESS_VERSION = "1"`** (stamped into checkpoints, guarded on load).
- `lstm_nlp/errors.py` — `LstmNlpError` base + `ConfigError`, `DataError`, `CheckpointError`, `PreprocessVersionMismatch`, `VocabError`, `TrainingError`.
- `lstm_nlp/utils/` — `seed.set_seed`/`make_generator`, `device.resolve_device`/`describe_device`, `logging.setup_logging`/`get_logger`.
- `lstm_nlp/config.py` — Pydantic v2 schema, discriminated union on `task`, YAML loader, `dump_config`.
- `configs/sentiment.yaml`, `configs/textgen.yaml`.
- `cli.py` — 4 subcommands parsing fully; handlers raise `NotImplementedError`.
- Tests: `test_utils.py`, `test_config.py`, `test_cli.py`.

### Verified (real output)
```
$ python -m pytest -q
....................................................  [100%]        52 passed

$ python -m lstm_nlp.cli --help
  train / eval / predict / generate      ← all four listed
$ lstm-nlp --version                     → lstm-nlp 0.1.0
$ python -c "import torch..."            → 2.13.0+cpu | cuda: False

configs resolve to the SHARED root data/ (no duplication — Rules.md B3):
  sentiment → ...\data\airline_sentiment.csv   exists=True  1,293,708 bytes
  textgen   → ...\data\alice.txt               exists=True    170,548 bytes

bad configs rejected, each naming the offending field:
  typo'd key        → sentiment.data.min_freqq: Extra inputs are not permitted
  dropout=1.0       → sentiment.model.dropout: Input should be less than 1
  temperature=0     → textgen.generate.temperature: Input should be greater than 0
  unknown monitor   → sentiment.train: early_stopping.monitor='val_bleu' is not...
  unknown task      → <root>: Input tag 'translation' does not match any
```

### Decisions made this phase
1. **Path convention: relative paths resolve against the config file's own directory.** One rule everywhere; a config never depends on CWD. Required `../../data/...` in the YAMLs.
2. **`extra="forbid"` and `frozen=True` on every config model.** A typo'd key is how a hyperparameter silently goes missing — the failure mode behind several D-defects. Configs are immutable after load.
3. **Discriminated union on `task`** rather than one merged schema, so sentiment and textgen can't borrow each other's fields.
4. **`early_stopping.monitor` validated against a per-task allow-list.** Sentiment defaults to `val_macro_f1`, not `val_accuracy` — accuracy's baseline is already 0.795 (D4).
5. **`generate.seed_text` is a config value**, replacing the reference's `input_words[-28701]` magic index (D10).
6. **CLI `generate` flags default to `None`** so config supplies defaults and an explicit flag overrides.
7. **`--max-steps` on `train`** from the start, for smoke tests (`Rules.md` B7).
8. **Installed torch from the CPU index** (`download.pytorch.org/whl/cpu`) to avoid pulling ~2 GB of CUDA wheels onto a machine with no GPU.

### Surprises / corrections
- **`Architecture.md` §5.3 had the data path wrong** (`../data/...` resolves inside `pytorch/`). Corrected to `../../data/...`, and the path-resolution rule is now stated in the doc.
- **`Phases.md` and `Architecture.md` disagreed on when `Memory.md` is created** (P0 task 10 vs "first action of P1"). Settled on **end of P0**; both docs corrected.
- `alice.txt` is 170,548 bytes on disk but 164,045 characters when decoded — CRLF line endings plus a UTF-8 BOM. Byte counts and character counts will not match; the doc figures are **characters**.
- Test count came to 52, not the 3 the phase sketch suggested. The config failure-mode matrix is worth the extra lines: every one of those is a silent-failure class.

### Deferred (noted, not done — `Rules.md` A6)
- `pytorch/README.md` — scheduled for Phase 7.
- `ruff`/`mypy` clean pass — Phase 8.
- `runs/.gitkeep` exists so the directory survives, but `runs/` is gitignored and the repo is not under version control yet (`Rules.md` B5 — the maintainer's call).

### Next: Phase 1 — Data layer
Build `data/preprocess.py`, `vocab.py`, `data/sentiment.py`, `data/textgen.py`.
**Closes D3, D6, D7, D9.** The 12 measured values in `Phases.md` §Phase 1 must be reproduced *as test assertions*, not recomputed by hand.
Watch for: `preprocess.py` must import without torch; vocab must be built from the **training split only**; never materialise a one-hot tensor.

---

## Phase 1 — Data layer ✅ complete (2026-08-29)

### Built
- `data/preprocess.py` — `strip_gutenberg`, `clean_tweet`, `clean_book`, `tokenize`, `load_corpus`, `count_tokens`. Pure functions, **no torch import** (proved by a subprocess test that blocks torch at the import hook).
- `vocab.py` — frozen `Vocab` dataclass. `build(counts, min_freq, specials)`, `index`/`token`/`encode`/`decode`/`count_unknown`, `to_dict`/`from_dict`. Two special sets: `PADDED_SPECIALS` (`<pad>`=0, `<unk>`=1) for sentiment, `UNPADDED_SPECIALS` (`<unk>`=0) for text-gen.
- `data/sentiment.py` — `load_sentiment_frame`, `prepare_sentiment_data`, `SentimentDataset`, `collate_sentiment`, `compute_class_weights`, `SentimentSplits`.
- `data/textgen.py` — `prepare_textgen_data`, `split_tokens`, `WindowDataset`, `TextGenSplits`.
- Fixtures: `tests/fixtures/sentiment_sample.csv` (200 stratified rows), `tests/fixtures/mini_book.txt` (~2.1k words wrapped in real Gutenberg markers).
- `tests/conftest.py` with a `realdata` marker that **skips** rather than fails when `data/` is absent.
- Tests: `test_preprocess.py` (36), `test_vocab.py` (33), `test_datasets.py` (34).

### Verified (real output)
```
$ python -m pytest
146 passed in 6.88s

peak python heap during dataset construction: 125.9 MB   (criterion < 200 MB)

SENTIMENT   train/test 8078 / 3463          vocab 4,505
            pos rate   0.2048 / 0.2047      class_weights [1.0, 3.884]
            unk rate   train 3.38% · test 5.23%
TEXTGEN     tokens     24,687 + 2,742 = 27,429
            vocab      2,436 (train-only, min_freq=1)
            windows    24,677 + 2,732 = 27,409
            storage    0.22 MB lazy · 2.19 MB if densified
```

### Decisions made this phase
1. **`WindowDataset` slices one 1-D tensor lazily** instead of materialising `(N, 10)`. Storage is O(n_tokens) = 0.22 MB, not O(n_windows × seq_len) = 2.19 MB. Strictly better than the plan, and it makes the D9 test stronger: it asserts *nothing is precomputed per window*, not merely that windows aren't one-hot.
2. **Text-gen `min_freq` 2 → 1.** Measured on the train block:

   | min_freq | vocab | train `<unk>` | val `<unk>` |
   |---|---|---|---|
   | **1** | **2,436** | **0.00%** | 6.24% |
   | 2 | 1,364 | 4.34% | 10.58% |
   | 3 | 984 | 7.42% | 13.82% |

   At `min_freq=2`, 4.34% of *training targets* are `<unk>` — the model learns to emit it, and generated text fills with `<unk>`. Visible directly: the first window renders as `<unk> alice s adventures in <unk> by <unk> <unk> the`. Costs ~413k extra params (920k → 1.33M), irrelevant on CPU. **Sentiment keeps `min_freq=2`** — there, dropping hapax halves the embedding table and the 5.23% OOV is harmless to classification.
3. **Sentiment `<unk>` is deliberately trained.** `min_freq=2` makes 3.38% of *training* tokens `<unk>`, which is what gives that embedding row gradient signal. A 0% train rate would mean `<unk>` is a random row first used at inference. The test asserts `0 < train_rate < test_rate`.
4. **Empty cleaned text → `[<unk>]`, not a zero-length sequence.** `pack_padded_sequence` rejects length 0. (No tweet in this data cleans to empty, but the API will accept arbitrary strings.)
5. **Vocab ordering breaks frequency ties alphabetically.** Without it, indices depend on dict insertion order and two same-seed runs could differ — breaking PRD S10.
6. **`Vocab.token()` raises on an out-of-range index** while `index()` never raises on an unknown token. Bad user input is absorbed; a bad index is a caller bug.

### Surprises / corrections
- **Textgen vocab is 1,364 at min_freq=2, not the 1,470 in the plan.** The plan's figure was computed on the **full corpus before splitting**; the code correctly builds train-only. The code is right — this is the D7 fix visibly working on the text-gen path too. Same cause for **27,409 windows, not 27,419** (two blocks each lose `seq_len`).
- **The 707 MB one-hot figure was a hypothetical** (stripped corpus × raw vocab). What the reference *actually* allocated: `modular_code` 931 MB, the notebook 1,671 MB (it never called `pre_process`, so its vocab was 5,649). `PRD.md` D9 now cites the real numbers.
- **`PRD.md` FR-7 said the val block "must not be the trailing slice".** After the Gutenberg strip the trailing slice *is* real prose, and a trailing held-out block is standard LM practice. FR-7 amended to say so explicitly, with the caveat that this is safe *only* because FR-2 runs first — `test_disabling_strip_keeps_boilerplate` pins that dependency.
- **My first `train.unknown_rate() < 0.01` assertion was wrong**, not the code (real value 3.38%). Replaced with an assertion that encodes the actual invariant (see decision 3).
- The `<unk>` rate asymmetry (train 3.38% / test 5.23%) is expected and healthy: test contains words that never appeared in training at all.

### Deferred
- `stride` is implemented and tested but stays 1; sweeping it is a Phase 8 experiment.
- `min_freq` sweep for sentiment (Phase 8 item 6) — 2 is measured-reasonable, not proven optimal.

### Next: Phase 2 — Sentiment model & training
Build `models/sentiment_lstm.py`, `engine/{metrics,callbacks,trainer}.py`, `inference/checkpoint.py`; wire `train`/`eval`.
**Closes D4, D5, D8, D11.** Gate: test macro-F1 ≥ 0.75 vs the **0.4430** baseline, early stopping demonstrably fires, checkpoint loads in a bare subprocess, and the negation test (S8) passes — that last one is the payoff for all of Phase 1's D3 work.
Inputs ready: `prepare_sentiment_data(...)` → `SentimentSplits(train, test, vocab, class_weights=[1.0, 3.884])`, V=4,505, expected params ≈ 355,010.
