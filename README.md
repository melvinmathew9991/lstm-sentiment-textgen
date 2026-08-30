# Many-to-One LSTMs — Sentiment Detection & Text Generation

A full-stack PyTorch rebuild of a TensorFlow/Keras teaching project, undertaken because an end-to-end audit found the original does not run to completion and that several of the things it demonstrates are provably wrong.

**Stack:** PyTorch · FastAPI · Streamlit · CPU-only

| | |
|---|---|
| **Task A** | Binary sentiment classification over 11,541 airline tweets |
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

The full catalogue, with evidence for each, is in [`PRD.md` §1.1](PRD.md). Progress closing them is tracked in [`Phases.md`](Phases.md).

This is not a "port". A faithful reproduction of provably wrong code has negative value. Every defect is fixed, and each fix has a named regression test.

---

## Documentation

Read in this order. Together they are the specification; the code is the implementation of it.

| Document | Contents |
|---|---|
| [`PRD.md`](PRD.md) | Requirements, users, scope, the 11 defects, measurable success criteria |
| [`Architecture.md`](Architecture.md) | Three-tier structure, data flow, model specs, checkpoint format, HTTP contract |
| [`Rules.md`](Rules.md) | Library allow/deny list, 16 correctness invariants, error policy, git conventions |
| [`Phases.md`](Phases.md) | Ten phases, each with exit criteria and a runnable verification command |
| [`Design.md`](Design.md) | Frontend palette, typography, layout, component behaviour |
| [`Memory.md`](Memory.md) | Running progress log — decisions, measurements, corrections |

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
pytest                       # full suite, < 60s, no network, no training
pytest -m "not realdata"     # skip tests that need data/
```

---

## Results

Populated as phases complete. Every figure is measured, and every metric is reported **beside its baseline** — a number without one is not evidence.

| Metric | Baseline | Result |
|---|---|---|
| Sentiment accuracy | 0.7953 (majority class) | **0.8926**  (+0.0973) |
| Sentiment macro-F1 | 0.4430 (majority class) | **0.8391**  (+0.3961) |
| Text-gen perplexity | 2,436 (uniform over vocab) | **223.54**  (10.9x better) |

Note on the original's headline number: its 0.909 validation accuracy is **not** a target. It was produced by a model trained on negation-stripped text (D3), reported without a baseline it barely clears (D4), and saved at its worst epoch (D5). Reproducing it would not be evidence of anything. See [`PRD.md` §6.3](PRD.md).

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
| 9 · Hardening | ⬜ | optional |
