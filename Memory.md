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
| **Stack** | PyTorch core → FastAPI backend → Streamlit frontend. The frontend is a **pure API client** (`Rules.md` C15) |
| **Repo** | https://github.com/melvinmathew9991/lstm-sentiment-textgen (public) · branch+PR+tag per phase, `Rules.md` §10 |
| **`Design.md`** | Added 2026-08-29 when Streamlit gave the project a visual surface. It was correctly skipped while the interface was CLI + JSON |

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

---

## Interlude — Repository, git workflow, and full-stack scope change (2026-08-29)

Between Phase 1 and Phase 2 the maintainer authorised git, and changed the scope from a two-tier (CLI + API) to a three-tier (CLI + API + UI) application.

### Repository

**https://github.com/melvinmathew9991/lstm-sentiment-textgen** — public.

```
2554d84  chore: initialise repository with specification and frozen reference
ee72707  feat(config): add package scaffold, typed errors and validated config   (tag v0.1.0)
38f720a  feat(data): add preprocessing, vocabulary and datasets                  (tag v0.2.0)
897f173  chore(lint): resolve ruff findings and assert precise exception types
```

Added at repo root: `README.md`, `LICENSE` (MIT for code; datasets carry their own terms), `.gitignore`, `.gitattributes`, `CHANGELOG.md`, `.github/workflows/ci.yml`, PR and issue templates.

### Workflow — follow this for every remaining phase

```bash
git switch -c phase-N-short-name
# ... commits ...
gh pr create --fill
gh pr merge --squash --delete-branch
git switch main && git pull
git tag -a v0.X.0 -m "Phase N: <name>"
git push --follow-tags
```

**Never commit directly to `main`.** **Never put a trailer in a commit message, PR body, or tag** — no `Co-Authored-By`, no session URL, no tool attribution. Standing maintainer instruction, recorded in `Rules.md` §10, and it overrides any default behaviour.

Conventional Commits with scopes: `data` · `vocab` · `models` · `engine` · `inference` · `api` · `frontend` · `cli` · `config` · `docs` · `ci`.

### CI

GitHub Actions on push and PR: install with the CPU torch index, **assert no CUDA wheel was pulled**, run `pytest -v` on Python 3.10 and 3.12, smoke-test the CLI. A lint job runs `ruff` non-blocking (`|| true`) until Phase 9. All four runs so far are green.

### Excluded from version control
`runs/`, `*.pt`, `__pycache__/`, and `modular_code/output/sentiment_model.h5` — that last one is a build artifact, holds the overfit final-epoch weights (D5), and is unusable anyway because its vocabulary was never saved (D8).

### Data licensing (public repo — this now matters)
`airline_sentiment.csv` is **CC BY-NC-SA 4.0** (Figure Eight via Kaggle): attribution, **non-commercial**, share-alike, and those terms bind downstream use. `alice.txt` is Project Gutenberg #11, US public domain. Both are documented in `README.md` and `LICENSE`. The `realdata` test marker means the suite still passes if someone deletes `data/`.

### Scope change: full-stack
- **Phase 7 (Streamlit frontend) inserted.** Ten phases now; old P7/P8 became P8/P9.
- **`Design.md` created.** I skipped it originally and said so explicitly — with a CLI + JSON interface there was no visual surface for a document about colour and typography to describe. Streamlit created one, so it now exists: palette with verified contrast ratios, typography, layout, components, error states, and the temperature visualisation.
- **`Rules.md` C15**: the frontend never runs inference. No `torch`, no checkpoint, no `lstm_nlp.models` under `frontend/`. Two inference paths would mean two sets of results.
- **`Rules.md` C16**: no metric shown in the UI without its baseline. C11 applied to pixels.
- **New deps, all already installed:** `streamlit` 1.61.1, `httpx` 0.28.1, `altair` 6.2.2. Nothing to install.

### Lint finding worth remembering
`B017` flagged two `pytest.raises(Exception)` assertions. Fixing them properly exposed that `SentimentConfig` raises `ValidationError` while `Vocab` raises `FrozenInstanceError` — my first fix asserted the wrong one for `Vocab` and the suite caught it. A blind exception assertion in a test is the same failure mode the reference kept making in its source: it passes whether or not the code does what you think.

### Next: Phase 2 — Sentiment model & training
Unchanged by any of the above. Gate: macro-F1 ≥ 0.75 vs the **0.4430** baseline, early stopping demonstrably fires, checkpoint loads in a bare subprocess (S6), negation test passes (S8).
Branch `phase-2-sentiment`, tag `v0.3.0` on merge.

---

## Standing practice — end-to-end audit after every phase (2026-08-29)

Maintainer instruction: evaluate the whole project after each phase, and hold the standards everywhere, not just in the newest code.

Implemented as an executable gate rather than a promise, because a promise to check things decays and a script does not.

### `pytorch/scripts/audit.py`

```bash
python scripts/audit.py          # full (runs the suite)
python scripts/audit.py --fast   # skip tests
python scripts/audit.py -v       # detail for passing checks too
```

23 checks in four groups — Tests, Standards, Documentation, Git. Non-zero exit on any FAIL. Wired into `Rules.md` §11, into every phase's exit criteria in `Phases.md`, into the PR template, and into CI as a separate `audit` job with `fetch-depth: 0` so it can read commit messages and tags.

**Non-obvious checks worth knowing about:**
- **`STALE_FIGURES`** — superseded measurements (`1,470`, `27,419`, `707 MB`, `961,214`) must not appear anywhere outside an explicit history note. This is how a correction is forced to propagate through all eight documents instead of being fixed in one place.
- **No trailers in any commit** — the maintainer's standing instruction, now mechanically enforced over the whole history rather than remembered.
- **Frozen reference untouched** — B1 checked as `git log -- modular_code notebook` having at most one commit.
- **Phase status consistency** — `Phases.md`, `README.md` and `Memory.md` must agree on what is done.
- **Defect coverage is phase-aware** — a defect whose closing phase hasn't landed reports as pending, not missing, so the check is meaningful from Phase 1 rather than only at the end.

### First run found three things

| Finding | Verdict |
|---|---|
| `Rules.md` A4 cited "707 MB" as a measured value | **Real bug.** A4 is the rule that says *don't invent numbers* and it was itself quoting a superseded figure. Fixed to 931 MB. |
| `CHANGELOG.md` "…2,436 (not 1,470)…" flagged | **Check was wrong.** The historical marker sat two lines above the figure in a multi-line bullet. Made the check paragraph-aware. |
| `Phases.md` `No "bag of words" in the new tree` flagged | **Check was wrong.** A prohibition of the term read as a use of it. Added prohibition words to the corrective-context list. |

Plus 4 public properties without docstrings (WARN), fixed.

Result: **21 pass · 1 warn · 0 fail · 1 skip**. The warn is the dirty working tree during the audit's own development; the skip is frontend purity, which has nothing to check until Phase 7.

### The rule that matters most
**Never weaken a check to make it green.** When it fires, decide honestly whether the code/doc is wrong or the check is imprecise, and say which in the PR. Two of the three first-run findings were check bugs and one was a real doc bug — recording which is which is the whole discipline.

---

## Phase 2 — Sentiment model & training ✅ complete (2026-08-29)

**Closes D4, D5, D8, D11.**

### Result

```
                    value    baseline      lift
  accuracy         0.8972     0.7953   +0.1019
  macro-F1         0.8485     0.4430   +0.4055
  ROC-AUC          0.9366     0.5000   +0.4366

  per class        prec     recall        F1    support
    negative      0.9487     0.9205    0.9344       2754
    positive      0.7231     0.8068    0.7627        709
```

1m34s on CPU (budget 5 min). Early stopping fired at epoch 11, restored epoch 6. 355,010 parameters, exactly as `Architecture.md` specifies. **213 tests pass.**

### Built
- `models/sentiment_lstm.py` — `Embedding(4505,64,padding_idx=0)` → 2-layer `LSTM(64,64)` → `Dropout` → `Linear(64,2)`, `pack_padded_sequence`, returns logits.
- `engine/metrics.py` — `ClassificationReport` with baselines **computed from the labels**, `majority_baseline`, `perplexity`, `uniform_perplexity_baseline` (ready for P3).
- `engine/callbacks.py` — `EarlyStopping`, `BestWeights`.
- `engine/trainer.py` — task-agnostic loop: clipping, NaN abort, history, tqdm.
- `engine/sentiment_task.py` — config → checkpoint wiring.
- `inference/checkpoint.py` — self-contained `.pt`, version guard, `build_model`, `describe`.
- CLI `train` and `eval`.

### Decisions
1. **Baselines are computed, never hardcoded.** `majority_baseline(y_true)` derives them from the labels. A hardcoded baseline drifts the moment the split changes, and a stale baseline is worse than none — it looks like corroboration.
2. **The trainer takes a `step_fn`,** so it knows nothing about the task. Phase 3 must reuse it unchanged; if it needs a task branch, fix the abstraction, not the loop.
3. **`BestWeights` keeps the snapshot in memory** rather than writing per epoch. Under 1.4M params, a deep copy is far cheaper than repeated disk writes.
4. **NaN/inf loss aborts the run** with epoch and batch index. Continuing only yields a longer run with no model.
5. **`eval` requires the run's `config.yaml`.** A checkpoint records what the model *is*, not where its data came from; re-evaluating needs the split definition. It fails with a clear message rather than guessing.

### Surprises / corrections
- **The loss function must not be attached to the model.** I wrote `model.criterion = nn.CrossEntropyLoss(weight=...)`. `nn` losses are themselves modules, so this registered a submodule and leaked `criterion.weight` into the `state_dict`, which then failed to load into a freshly-constructed model. **Caught by `test_checkpoint_is_self_contained`** — exactly the test written to catch D8-shaped problems, catching a new one. Fixed with a `make_sentiment_step(criterion)` factory. Retraining produced *identical* metrics, confirming the change was purely structural and incidentally demonstrating seed reproducibility (S10).
- **The untrained smoke run scored exactly the baseline** — 0.7953 accuracy, 0.4430 macro-F1, positive-class F1 of 0.0. An accidental but perfect demonstration of D4: accuracy 0.795 reads as respectable, and macro-F1 is what exposes it as worthless.
- **The model is negative-biased.** `"i would recommend this airline"` is classified negative (p(pos)=0.127). The corpus is airline *complaints* — 79.5% negative — so positive phrasings without strong markers drift negative. Negation still moves it the right way (0.127 → 0.008). Honest limitation, not a defect.
- Two `float(tensor)` calls on grad-tracking tensors raised UserWarnings. Fixed with `.detach().item()`.

### Negation demonstration (S8)
```
the flight was great               positive  0.977
the flight was not great           positive  0.751     gap 0.226
service was good                   positive  0.782
service was not good               negative  0.228     gap 0.554
i would recommend this airline     negative  0.127
i would not recommend this airline negative  0.008     gap 0.118
```
Under the reference's preprocessing each pair was **identical** after stopword removal. No model trained that way could separate them.

### Audit
**21 pass · 1 warn · 0 fail · 1 skip.** Warn is the dirty tree during the phase; skip is frontend purity (Phase 7).

### Next: Phase 3 — Text-generation model & training
`models/textgen_lstm.py`: `Embedding(2436,128)` → `LSTM(128,256)` → `h_n[-1]` → `Linear(256,2436)`, ≈1,333,124 params. **Reuse `engine/trainer.py` unchanged.** Gate: perplexity ≤ 400 vs the **2,436** uniform baseline, RSS < 2 GB, < 20 min CPU.
The ≤400 target is still an unvalidated guess; report the real number rather than adjusting the target to fit it.
Branch `phase-3-textgen`, tag `v0.4.0`.

---

## Phase 3 — Text-generation model & training ✅ complete (2026-08-29)

**Closes D6, D9.**

### Result

```
                    value    baseline      ratio
  perplexity       223.54        2436      10.90x better
  cross-entropy    5.4096      7.7981
  top-1 accuracy   0.1482    0.000411      360x better
```

2m47s on CPU (budget 20 min). Early stopping fired at epoch 7, restored epoch **2**. 1,333,124 parameters, exactly as specified. **231 tests pass.**

Gate S4 was perplexity ≤ 400 — the target I flagged as an unvalidated guess. It landed at **223.54**, so the guess was reasonable. Recording that it was checked rather than adjusted.

### Memory (D9), measured
```
  peak process RSS         421 MB    criterion < 2,000 MB
  RSS before data load     293 MB    torch import alone
  dataset storage         0.22 MB    int64 indices, lazily sliced
  same windows one-hot     668 MB    at our V=2,436
  reference actual         931 MB    modular_code, V=3,036, unstripped
```
The whole training process uses **less memory than the reference's input array alone**.

### Built
- `models/textgen_lstm.py` — `Embedding(2436,128)` → `LSTM(128,256)` → `h_n[-1]` → `Linear(256,2436)`. Returns logits.
- `engine/textgen_task.py` — wiring, `evaluate_textgen`, `format_textgen_metrics`.
- `TextGenLSTM` in the checkpoint registry; `describe()` gained perplexity-with-baseline.
- CLI `train`/`eval` dispatch on the task discriminator.

### Decisions
1. **`engine/trainer.py` reused with zero changes** — the Phase 3 exit criterion. A new test parses the file with `ast`, strips docstrings, and asserts its *code* mentions neither task. The loop cannot quietly acquire task-specific branches later.
2. **Perplexity recomputed from logits** in `textgen_metrics` rather than taken from the running loss, so it stays correct if the criterion is ever weighted or reduced differently.
3. **Training windows are shuffled; the split is not.** The contiguous block split is what prevents leakage (D6); the order training windows arrive in carries no information worth preserving.
4. **The head is the biggest part of the model** — 626,052 of 1,333,124 params. That is what a word-level softmax costs, and it is the thing that would need attention if the vocabulary grew.

### Surprises / corrections
- **Best epoch is 2 of 7.** The model overfits a 24,687-token corpus almost immediately. Expected for a corpus this small, and exactly why early stopping is not optional here. Worth stating plainly: this is a *demonstration* language model, locally fluent at best.
- **My first "trainer is task-agnostic" test was too naive** — it string-matched the whole file and tripped on the module docstring, which legitimately *says* the loop knows nothing about sentiment or text generation. Rewrote it to parse with `ast` and strip docstrings before scanning. Same discipline as the audit: make the check precise, never loosen it.
- **The Bash heredoc mangles backslashes even with a quoted delimiter** on this machine, corrupting `\n` inside Python string literals. It silently produced a syntax error in `cli.py` earlier. From now on: write patch scripts to the scratchpad with the Write tool and run them, rather than piping Python through a heredoc.

### Audit
**21 pass · 1 warn · 0 fail · 1 skip.**

### Next: Phase 4 — Sampling & generation
**The most important phase in the plan: it closes D2.**
Build `inference/sampler.py` (`apply_top_k`, `sample_from_logits`) and `inference/predictor.py`.
Gate S5: entropy strictly increases across T ∈ {0.2, 0.5, 1.0, 1.5, 2.0}; T=0.01 matches greedy argmax on ≥99% of 1,000 draws; `top_k=5` yields at most 5 distinct tokens over 10,000 draws; same `rng_seed` gives identical text.
S15: five passages at T=0.7 containing no Gutenberg vocabulary — already guaranteed structurally, since those words have no index.
Both trained checkpoints exist under `runs/`. Branch `phase-4-sampling`, tag `v0.5.0`.
