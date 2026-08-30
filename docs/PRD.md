# PRD — Sentiment Detection & Text Generation with Many-to-One LSTMs (PyTorch)

**Status:** In progress — Phase 0 and Phase 1 complete (2026-08-29)
**Owner:** meriatmelvin@gmail.com
**Created:** 2026-08-29
**Supersedes:** nothing. The existing TensorFlow/Keras implementation in `modular_code/` is frozen as a reference, not replaced.

---

## 1. Why this project exists

There is a working-ish Keras implementation of two many-to-one LSTM tasks in this repo. An end-to-end audit (2026-08-29) found it does not run to completion, and that several of the things it teaches are wrong in ways the saved outputs prove. The goal is a PyTorch rebuild that is **correct, runnable, reproducible, and serveable** — keeping the original's pedagogical shape (many-to-one LSTM, two contrasting tasks) while fixing the defects.

This is not a research project. No new modelling ideas are in scope. The target is a clean, honest, well-tested implementation of a known architecture.

### 1.1 Defects the rebuild must fix

These are the requirements. Each has a traceability ID used throughout `Phases.md`.

| ID | Defect in the TF version | Evidence |
|----|--------------------------|----------|
| **D1** | `engine.py:48` calls `generate_paragraph` with 4 of 7 required args → `TypeError`, but only after ~100 epochs of training. The default `input_type = 2` path is the broken one. | Static arg-count check across all 8 cross-module calls; only this one mismatches |
| **D2** | Temperature sampling divides **probabilities** by T and feeds them to `tf.random.categorical`, which expects **logits** and applies its own softmax. With T=10 the inputs land in ~[0, 0.007] → near-uniform. It samples uniformly at random. | Notebook cell 130 output is uniform noise: `"the alone court, to. pop sits bursting particularly"` |
| **D3** | Stopword removal strips all 14 negations (`not`, `no`, `nor`, `don't`, `isn't`, …) before a *sentiment* model sees the text, inverting polarity. | `"chat support is not working"` → `"chat support working"`; `"not worried"` → `"worried"` |
| **D4** | Accuracy is the only metric on 4:1 imbalanced data. Predicting all-negative scores 0.795. | Measured: test split is 20.47% positive |
| **D5** | 50 epochs, no early stopping, no dropout, no checkpointing — the saved model is the final (worst) epoch. | Notebook log: `val_loss` 0.215 (ep1) → 0.526 (ep7), `loss` → 0.015 |
| **D6** | Text-gen `validation_split=0.1` takes the last 10% unshuffled; the story ends at 88.2% of the file, so validation is entirely Project Gutenberg legal boilerplate. Generation seeds come from it too. | Measured: `"end of the project gutenberg"` at 88.3% through |
| **D7** | Vocabulary is built on 100% of the data before the 70/30 split; 2,338 words occur only in test. `text_to_int` raises `KeyError` on any unseen word. | Measured on the actual split |
| **D8** | `output/sentiment_model.h5` is unusable: `word_to_int`, `int_to_word` and `sequence_length` are never persisted, so nothing can reconstruct the model's input space. | `process_sentiment_data` returns them; nothing saves them |
| **D9** | Text-gen input tensor is one-hot. `modular_code` allocated `(30664, 10, 3036)` = **931 MB**; the notebook, which never called `pre_process`, allocated `(29584, 10, 5649)` = **1,671 MB**. | Measured |
| **D10** | `input_type = 2` hardcoded; `input_words[-28701]` is a magic index tuned to a sample count the modular code no longer produces. | `engine.py:13`, `engine.py:45` |
| **D11** | Docs call the encoding "bag-of-words". It is an ordered integer-index sequence into a learned `Embedding` — the opposite of BoW, which discards the order the LSTM exists to exploit. | Notebook §"Bag of Words", solution PDF §Approach |

---

## 2. Users

| User | Needs | Reached via |
|------|-------|-------------|
| **Maintainer / learner** (primary) | Train both models from scratch on a CPU laptop in minutes, see honest metrics, understand what temperature actually does | CLI |
| **Downstream application** | Classify a tweet or generate text over HTTP without knowing the model internals | FastAPI service |
| **Non-technical evaluator** | See the models work, and *feel* what temperature does, without touching a terminal | Streamlit app |
| **Future contributor / AI agent** | Enough written context to make a correct change without re-reading the whole codebase | `Architecture.md`, `Rules.md`, `Memory.md` |

Explicit non-user: this project does not serve end consumers, has no auth, and is not hardened for public internet exposure.

---

## 3. Scope

### 3.1 In scope

**Task A — Sentiment classification**
Binary (positive / negative) over 11,541 airline tweets. Many-to-one: variable-length token sequence → single label.

**Task B — Next-word text generation**
Word-level LM over *Alice's Adventures in Wonderland*. Many-to-one: fixed 10-word window → next word. Iterated with temperature sampling to generate passages.

**Delivery surfaces** — a three-tier application
- **CLI** (`lstm-nlp`): `train`, `eval`, `predict`, `generate`
- **Backend** (FastAPI): `POST /predict`, `POST /predict/batch`, `POST /generate`, `GET /health`, `GET /models`
- **Frontend** (Streamlit): a two-page app over the backend — sentiment classification and interactive text generation with a live temperature control

### 3.2 Out of scope

- Transformers, attention, pretrained embeddings (GloVe/word2vec/BERT). The point is LSTMs.
- The neutral sentiment class (already dropped upstream; 11,541 of the original 14,640 rows remain).
- GPU-specific optimisation, distributed training, quantisation, ONNX export.
- Modifying `modular_code/` or `notebook/`. They are frozen reference artifacts.
- Beating a published benchmark. Correctness and honesty beat leaderboard position here.

---

## 4. Functional requirements

### 4.1 Data

| ID | Requirement |
|----|-------------|
| FR-1 | Read `data/airline_sentiment.csv` (11,541 rows, cols `airline_sentiment`, `text`) and `data/alice.txt` from a single shared location. No duplicated data directories. |
| FR-2 | Strip Project Gutenberg header/footer from `alice.txt` using the `*** START/END OF THE PROJECT GUTENBERG EBOOK ***` markers before any tokenisation. Fixes **D6**. |
| FR-3 | Sentiment preprocessing must **preserve negations**. Stopword removal is prohibited for this task. URLs → `<url>`, `@handles` → `<user>`, `&amp;` → `and`, lowercase, strip remaining punctuation except apostrophes. Fixes **D3**. |
| FR-4 | Build vocabulary from the **training split only**, with `min_freq` configurable per task and reserved specials at the leading indices — `<pad>`=0/`<unk>`=1 for the padded sentiment task, `<unk>`=0 for fixed-window text-gen, which needs no padding. Defaults: sentiment `min_freq=2`, text-gen `min_freq=1` (see `Memory.md` Phase 1 for the measurement behind the difference). Fixes **D7**. |
| FR-5 | Every token lookup must map unseen words to `<unk>`. No code path may raise `KeyError` on unknown input. Fixes **D7**. |
| FR-6 | Sentiment train/test split must be **stratified** on the label, `test_size=0.30`, `random_state=10`. *Amended 2026-08-30:* rows are **deduplicated on cleaned text before splitting**, and texts carrying contradictory labels are dropped — 270 rows (2.34%), after which 0 test rows share a training row (previously 86, or 2.48%). This deliberately breaks row-level comparability with the reference's split; `PARITY.md` §0 already establishes that the reference's numbers can never be re-measured, so the comparability the seed was pinned for does not exist to preserve. |
| FR-7 | Text-gen train/val split must be an explicit split over **contiguous blocks of real prose**, with windowing done per block so none straddles the boundary — never Keras `validation_split` semantics. *Amended 2026-08-29:* the held-out block **is** the trailing slice, which is standard LM practice (Penn Treebank, WikiText) and is safe here **only because FR-2 already removed the licence text** that previously occupied it. Disabling the strip re-creates D6, and a test asserts that. Fixes **D6**. |
| FR-8 | Sequences are stored as **integer indices**, never one-hot. Fixes **D9**. |

### 4.2 Models

| ID | Requirement |
|----|-------------|
| FR-9 | Sentiment: `Embedding(V, 64, padding_idx=0)` → 2-layer `LSTM(64, 64, batch_first=True)` → dropout → `Linear(64, 2)`. Must use `pack_padded_sequence` so padding never enters the recurrence. |
| FR-10 | Text-gen: `Embedding(V, 128)` → `LSTM(128, 256, batch_first=True)` → `Linear(256, V)`, taking the final hidden state only (many-to-one). |
| FR-11 | Both models emit **raw logits**. Softmax lives in the loss function and the sampler, never in the model's `forward`. This is the structural fix for **D2**. |
| FR-12 | Layer widths, dropout, `min_freq`, `max_len`, learning rate, batch size and epochs come from YAML config, not literals in code. Fixes **D10**. |

### 4.3 Training

| ID | Requirement |
|----|-------------|
| FR-13 | Early stopping on validation macro-F1 (sentiment) / validation loss (text-gen), with configurable patience, and the **best** checkpoint restored and saved — not the last. Fixes **D5**. |
| FR-14 | Class imbalance handled explicitly via `CrossEntropyLoss(weight=…)`. Measured ratio is **4.130:1** on the deduplicated corpus (3.884:1 before). Fixes **D4**. |
| FR-15 | Gradient clipping (`clip_grad_norm_`, default 5.0) on both tasks. |
| FR-16 | Every run writes a JSON history of per-epoch train/val loss and metrics to `runs/<task>/<timestamp>/history.json`. |
| FR-17 | Seeds for `random`, `numpy`, and `torch` set from config; `torch.use_deterministic_algorithms(True)` where it does not break LSTM kernels. Two runs with the same seed must produce the same metrics. |

### 4.4 Evaluation

| ID | Requirement |
|----|-------------|
| FR-18 | Sentiment eval reports accuracy, **macro-F1**, per-class precision/recall/F1, confusion matrix, and ROC-AUC — alongside the majority-class baseline for context. Fixes **D4**. |
| FR-19 | Text-gen eval reports validation cross-entropy and **perplexity**, alongside the uniform-guess baseline (ln V). |
| FR-20 | No metric may be reported without its baseline printed next to it. |

### 4.5 Inference & sampling

| ID | Requirement |
|----|-------------|
| FR-21 | Temperature sampling operates on **logits**: `softmax(logits / T)`. `T → 0` must converge to greedy argmax; larger `T` must monotonically increase output entropy. Fixes **D2**. |
| FR-22 | Optional `top_k` filtering, applied before temperature. |
| FR-23 | Generation is reproducible when an RNG seed is supplied in the request/CLI flag. |
| FR-24 | Generation seed text may contain unknown words without crashing (they become `<unk>`). |

### 4.6 Persistence

| ID | Requirement |
|----|-------------|
| FR-25 | A checkpoint is a **single self-contained `.pt`** holding model weights, model config, the full vocabulary, the preprocessing config + version string, metrics, epoch, and library versions. Loading it must require no other file. Fixes **D8**. |
| FR-26 | Loading a checkpoint whose `preprocess.version` differs from the running code must fail loudly with a clear message, never silently mis-tokenise. |

### 4.7 Interfaces

| ID | Requirement |
|----|-------------|
| FR-27 | CLI: `train`, `eval`, `predict`, `generate`, each taking `--config` and/or `--ckpt`. `--help` must work with zero dependencies beyond the package installed. |
| FR-28 | FastAPI: `POST /predict`, `POST /predict/batch`, `POST /generate`, `GET /health`, `GET /models`. Full contract in `Architecture.md` §6. |
| FR-29 | The API loads each checkpoint **once at startup**, not per request. |
| FR-30 | Invalid input returns HTTP 422 with a Pydantic validation body; a missing/corrupt checkpoint returns 503 with a clear message. Never a 500 stack trace. |

### 4.8 Frontend (Streamlit)

| ID | Requirement |
|----|-------------|
| FR-31 | The Streamlit app is a **pure HTTP client of the FastAPI backend**. It must not import `lstm_nlp.models`, load a checkpoint, or run inference itself. One inference path, one set of results. |
| FR-32 | **Sentiment page:** free-text input, predicted label with class probabilities, and the `unk_rate` surfaced so a user can see when a prediction rests on unknown tokens. Must display the majority-class baseline (0.795) beside any accuracy claim (extends FR-20 to the UI). |
| FR-33 | **Generation page:** seed input, word count, a **temperature slider (0.1–2.0)**, optional top-k, and an optional RNG seed for reproducible output. |
| FR-34 | The generation page must make the temperature/entropy relationship **visible** — a chart of the next-word probability distribution at the selected temperature. This is the pedagogical payload of D2 and the reason the frontend exists. |
| FR-35 | Backend unreachable, or a model not loaded, must render an actionable message in the UI — never a raw traceback or an infinite spinner. |
| FR-36 | Backend base URL comes from configuration/environment, not a hardcoded literal. |

---

## 5. Non-functional requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | **CPU-only training.** Dev machine is Python 3.10.11, Windows, no CUDA. GPU may be used if present but must never be required. | — |
| NFR-2 | Sentiment training wall-clock | < 5 min CPU |
| NFR-3 | Text-gen training wall-clock | < 20 min CPU |
| NFR-4 | Peak training RSS | < 2 GB (vs 931 MB for the reference's one-hot tensor alone) |
| NFR-5 | API single-request latency, warm | < 100 ms for `/predict`; < 2 s for `/generate` at 40 words |
| NFR-6 | Test suite runtime | < 60 s, no network, no model training |
| NFR-7 | No network access at import time. The current code calls `nltk.download()` on import in two modules — prohibited. | — |
| NFR-8 | Reproducible install from a pinned `requirements.txt` on Python 3.10 + numpy 2.x | — |
| NFR-9 | Frontend interaction latency (slider move → updated result), backend warm | < 2 s |
| NFR-10 | Frontend and backend start with two commands and no build step | — |

---

## 6. Success criteria

The rebuild is done when all of these hold. Numbers below are **measured properties of this dataset**, computed 2026-08-29, not aspirations copied from elsewhere.

### 6.1 Must pass (release gate)

| # | Criterion |
|---|-----------|
| S1 | `python -m lstm_nlp.cli train --config configs/sentiment.yaml` runs to completion and writes a loadable checkpoint. |
| S2 | `... train --config configs/textgen.yaml` likewise. Neither path crashes — **D1** closed. |
| S3 | Sentiment test **macro-F1 ≥ 0.75**, against a majority-class baseline of **0.443**. (Accuracy alone is not sufficient evidence; the baseline is 0.795.) |
| S4 | Text-gen **held-out test perplexity ≤ 400** (*amended 2026-08-30: was validation perplexity; a test block now exists*), against a uniform-guess baseline of **2,436** (= vocabulary size at the chosen `min_freq=1`; ln V = 7.798). |
| S5 | Temperature property test passes: entropy of the sampled next-word distribution increases monotonically across T ∈ {0.2, 0.5, 1.0, 1.5, 2.0}, and T=0.01 matches greedy argmax on ≥ 99% of draws. **D2** closed. |
| S6 | A checkpoint saved by process A loads in a fresh process B with no other artifact present and reproduces its recorded test metrics to within 1e-6. **D8** closed. |
| S7 | `predict` on a string containing a word absent from training returns a prediction rather than raising. **D7** closed. |
| S8 | Negation test: the classifier assigns different labels — or a probability gap > 0.15 — to `"the flight was great"` vs `"the flight was not great"`. **D3** closed. |
| S9 | Peak training RSS < 2 GB, measured and recorded. **D9** closed. |
| S10 | Two full training runs at the same seed produce identical test metrics. |
| S11 | `pytest` green; every FR above has at least one test asserting it. |
| S12 | Every FastAPI endpoint has a `TestClient` test, including the 422 and 503 paths. |
| S17 | The Streamlit app runs against a live backend: both pages work end to end, and the temperature slider visibly changes both the generated text and the probability chart. |
| S18 | Killing the backend leaves the UI showing a clear "backend unreachable" state, not a traceback or a hang. |
| S19 | The frontend contains no import of `lstm_nlp.models`, no checkpoint load, and no `torch` call (asserted by a test). |

### 6.2 Should pass (quality bar, non-blocking)

| # | Criterion |
|---|-----------|
| S13 | Sentiment macro-F1 ≥ 0.80. |
| S14 | Early stopping actually fires before the configured max epochs on both tasks (evidence that **D5** is closed rather than merely configured). |
| S15 | Generated text at T=0.7 contains no Project Gutenberg legal vocabulary (`copyright`, `donations`, `foundation`, `ebook`, `license`) — evidence **D6** is closed. |
| S16 | A short `PARITY.md` comparing PyTorch metrics against the frozen Keras reference, with the deltas explained. |

### 6.3 Explicitly not a success criterion

Matching the Keras model's 0.909 validation accuracy. That number was produced by a model trained on negation-stripped text and reported without a baseline; reproducing it is not evidence of anything. The stated F1 and perplexity targets replace it.

---

## 7. Document set

All steering documents live in `docs/`. `README.md` and `CHANGELOG.md` stay at
the repository root, where GitHub and Keep-a-Changelog respectively expect them.

Decided 2026-08-29:

| Doc | Status | Rationale |
|-----|--------|-----------|
| `PRD.md` | ✅ this file | — |
| `Architecture.md` | ✅ | Structure, tech stack, data flow, model specs, checkpoint format, API contract |
| `Rules.md` | ✅ | Library allow/deny list, error-handling policy, agent boundaries |
| `Phases.md` | ✅ | Eight phases with exit criteria and verification commands |
| `Design.md` | ✅ **added 2026-08-29** | Was skipped while the interface was CLI + JSON only. The Streamlit frontend created a real visual surface, so it now specifies palette, typography, layout, component behaviour and the temperature visualisation. |
| `Memory.md` | ⏳ deferred | Created as the first task of Phase 1, then appended after each phase. |

---

## 8. Key decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Layout | New `pytorch/` beside frozen `modular_code/` | Enables parity checks; nothing breaks if the migration stalls |
| Port fidelity | Fix all 11 defects during the port | A faithful port of provably wrong code has negative value |
| Interface | CLI + FastAPI + Streamlit | *Revised 2026-08-29.* Full-stack: CLI for training, FastAPI for programmatic use, Streamlit for demonstration. The frontend is a **pure client** of the API — it never imports the model code |
| Tokenisation | Hand-rolled regex + `Counter` vocab | `torchtext` is unmaintained and pinned to exact torch versions — see `Rules.md` §2 |
| Framework | PyTorch 2.x, CPU | Explicit `forward`, no `input_length`/`channels_last` incantations; models are small enough that CPU is genuinely fine |

## 9. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| 5.77% test OOV at `min_freq=2` | Sentiment ceiling | Measured and accepted; `<unk>` is trained (3.70% of training tokens), not a dead row. `min_freq` is a config knob. |
| 27,429 training tokens is a *very* small LM corpus | Text-gen output will be locally fluent, globally incoherent | Set expectations in docs; perplexity target is set against the uniform baseline, not against a real LM |
| Determinism vs cuDNN LSTM kernels | S10 may fail on GPU | CPU is the supported target; document the GPU caveat rather than fight it |
| Scope creep toward transformers | Loses the point of the project | Prohibited in `Rules.md` §2 |
