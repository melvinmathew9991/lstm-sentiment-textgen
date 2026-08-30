# Many-to-One LSTMs — Sentiment Detection & Text Generation

A full-stack PyTorch rebuild of a TensorFlow/Keras teaching project, undertaken because an end-to-end audit found the original does not run to completion and that several of the things it demonstrates are provably wrong.

**Stack:** PyTorch · FastAPI · Streamlit · CPU-only

| | |
|---|---|
| **Task A** | Binary sentiment classification over 11,541 airline tweets (11,271 after deduplication) |
| **Task B** | Word-level next-token generation over *Alice's Adventures in Wonderland* |
| **Shape** | Many-to-one LSTM in both cases: a sequence in, one prediction out |

---

## Why this repo exists

The original implementation (preserved unmodified in [`modular_code/`](modular_code/) and [`notebook/`](notebook/)) has eleven catalogued defects. Four are worth stating up front, because they set the bar the rebuild has to clear:

| | Defect | Evidence |
|---|---|---|
| **D2** | Temperature sampling divided *probabilities* by T and fed them to `tf.random.categorical`, which expects *logits* and applies its own softmax. The result is a uniform random draw at every temperature. | The notebook's own saved output is noise: `"the alone court, to. pop sits bursting particularly"` |
| **D3** | Stopword removal stripped all 14 negations before a *sentiment* model saw the text. | `"chat support is not working"` → `"chat support working"` |
| **D4** | Accuracy was the only metric on 4:1 imbalanced data, where predicting "negative" always scores **0.795**. | Test split measured at 20.47% positive |
| **D1** | The default execution path crashed with a `TypeError` — after ~100 epochs of training. | `engine.py:48` passes 4 of 7 required arguments |

The full catalogue, with evidence for each, is in [`PRD.md` §1.1](docs/PRD.md). Progress closing them is tracked in [`Phases.md`](docs/Phases.md).

This is not a "port". A faithful reproduction of provably wrong code has negative value. Every defect is fixed, and each fix has a named regression test.

---

## Documentation

Read in this order. Together they are the specification; the code is the implementation of it.

| Document | Contents |
|---|---|
| [`PRD.md`](docs/PRD.md) | Requirements, users, scope, the 11 defects, measurable success criteria |
| [`Architecture.md`](docs/Architecture.md) | Three-tier structure, data flow, model specs, checkpoint format, HTTP contract |
| [`Rules.md`](docs/Rules.md) | Library allow/deny list, 16 correctness invariants, error policy, git conventions |
| [`Phases.md`](docs/Phases.md) | Ten phases, each with exit criteria and a runnable verification command |
| [`Design.md`](docs/Design.md) | Frontend palette, typography, layout, component behaviour |
| [`Memory.md`](docs/Memory.md) | Running progress log — decisions, measurements, corrections |

---

## Architecture

```
┌─────────────┐   HTTP/JSON   ┌─────────────┐   in-process   ┌──────────────┐
│  Streamlit  │ ────────────▶ │   FastAPI   │ ─────────────▶ │  lstm_nlp    │
│  frontend   │ ◀──────────── │   backend   │ ◀───────────── │  core + ckpt │
└─────────────┘               └─────────────┘                └──────────────┘
 presentation                  contract                       models, data,
 no torch                      validation                     training, the only
 no checkpoints                loads once at startup          place inference runs
```

The frontend is a **pure API client** — it never imports the model code. One inference path means one set of results.

```
├── docs/                  Steering documents (see the table above)
├── data/                  Datasets (see Licensing below)
├── modular_code/          FROZEN — original TensorFlow implementation
├── notebook/              FROZEN — original tutorial notebook
└── pytorch/
    ├── configs/           YAML: every hyperparameter, no literals in code
    ├── src/lstm_nlp/      Core package
    │   ├── data/          Preprocessing, vocab, datasets
    │   ├── models/        SentimentLSTM, TextGenLSTM
    │   ├── engine/        Trainer, callbacks, metrics
    │   ├── inference/     Checkpoints, predictors, sampler
    │   ├── api/           FastAPI application
    │   └── cli.py         train · eval · predict · generate
    ├── frontend/          Streamlit app
    └── tests/
```

---

## Quickstart

Requires **Python 3.10–3.12**. CPU-only; no GPU needed.

```bash
git clone https://github.com/melvinmathew9991/lstm-sentiment-textgen.git
cd lstm-sentiment-textgen/pytorch

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Train

```bash
lstm-nlp train --config configs/sentiment.yaml     # < 5 min CPU
lstm-nlp train --config configs/textgen.yaml       # < 20 min CPU
```

### Use from the terminal

```bash
lstm-nlp eval     --ckpt runs/sentiment/<ts>/best.pt
lstm-nlp predict  --ckpt runs/sentiment/<ts>/best.pt "the flight was not great"
lstm-nlp generate --ckpt runs/textgen/<ts>/best.pt --seed "alice was" --temperature 0.7
```

### Run the stack

```bash
uvicorn lstm_nlp.api.app:app --port 8000    # backend  → :8000/docs
streamlit run frontend/app.py               # frontend → :8501
```

### Test

```bash
pytest                       # fast path, < 60s, no network, no training
pytest -m ""                 # everything, including the tests that train
pytest -m "not realdata"     # skip tests that need data/
```

**Train first, or 53 tests skip.** `runs/` is gitignored, so a fresh clone has no
checkpoints and every test that loads a trained model — the negation aggregate,
entropy monotonicity on real logits, the checkpoint round-trip against a real
run, and the subprocess CLI tests — skips at runtime rather than failing.
Measured on a clean tree: `pytest -m ""` gives **391 passed, 53 skipped**; after
the two `train` commands above it gives **448 passed, 0 skipped**. A skip is
green, so nothing local flags it.

CI no longer has this problem. Every matrix leg smoke-trains (`--max-steps 3`,
~20 s) so 442 of the 448 run on both Python versions, and a separate
`trained-model gates` job trains both models properly and runs the whole suite
with a **skip budget of zero**. The four tests a smoke model cannot satisfy —
the macro-F1 gate, negation, minority-class recall, early stopping — carry the
`fulltrain` marker and run only in that job. Until 2026-08-30 no headline number
in this repository had ever been verified by CI; see `Memory.md`, Phase 5, for
how long that was true and how it was found.

---

## Results

Populated as phases complete. Every figure is measured, and every metric is reported **beside its baseline** — a number without one is not evidence.

| Metric | Baseline | Result |
|---|---|---|
| Sentiment accuracy | 0.8048 (majority class) | **0.8974**  (+0.0925) |
| Sentiment macro-F1 | 0.4459 (majority class) | **0.8300**  (+0.3841) |
| Text-gen perplexity | 2,436 (uniform over vocab) | **267.54**  (9.11x better) |

All three are **held-out test** figures, scored once. The perplexity row read
`223.54 (10.9x better)` until 2026-08-30 — the v0.4.0 validation figure under the
old 90/10 split, superseded when v1.2.0 carved a test block out of the held-out
slice. Validation perplexity on the current split is 186.27; the 44% gap between
that and 267.54 is the selection effect, and publishing the selected number as
the headline is the mistake this table exists to avoid.

Note on the original's headline number: its 0.909 validation accuracy is **not** a target. It was produced by a model trained on negation-stripped text (D3), reported without a baseline it barely clears (D4), and saved at its worst epoch (D5). Reproducing it would not be evidence of anything. See [`PRD.md` §6.3](docs/PRD.md).

---

## Data & licensing

**Code** in this repository carries no licence. No usage rights are granted;
all rights are reserved by default.

**Datasets** in `data/` are third-party and carry their own terms, which are
imposed by their sources and are unaffected by the above:

| File | Source | Licence |
|---|---|---|
| `airline_sentiment.csv` | Twitter US Airline Sentiment (Figure Eight / CrowdFlower, via Kaggle) | **CC BY-NC-SA 4.0** — attribution, non-commercial, share-alike |
| `alice.txt` | *Alice's Adventures in Wonderland*, Lewis Carroll — [Project Gutenberg #11](https://www.gutenberg.org/ebooks/11) | Public domain in the US; Project Gutenberg trademark terms apply to the unmodified file |

The airline dataset is the 11,541-row binary subset: the original 14,640 rows less the 3,099 neutral ones. It is redistributed here under CC BY-NC-SA 4.0 for reproducibility of the measured figures. **The non-commercial and share-alike terms bind any downstream use.** If that is a problem for your use case, delete `data/` — the test suite skips the `realdata` tests cleanly when it is absent.

The original course materials (`modular_code/`, `notebook/`, the solution PDF) are retained unmodified as the reference this work is measured against, and are the property of their original author.

---

## Status

| Phase | | Closes |
|---|---|---|
| 0 · Scaffold | ✅ | — |
| 1 · Data layer | ✅ | D3, D6, D7, D9 |
| 2 · Sentiment model | ✅ | D4, D5, D8, D11 |
| 3 · Text-gen model | ✅ | D6, D9 |
| 4 · Sampling | ✅ | **D2**, D10 |
| 5 · CLI | ✅ | **D1**, D10 |
| 6 · FastAPI backend | ✅ | — |
| 7 · Streamlit frontend | ✅ | — |
| 8 · Parity report | ✅ | D11 |
| 9 · Hardening | ✅ | — |
