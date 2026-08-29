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

