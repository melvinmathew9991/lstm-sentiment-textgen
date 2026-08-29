# Architecture — PyTorch Rebuild

Companion to `PRD.md`. Describes structure, data flow, model specifications, artifact formats and the HTTP contract.
All quantities here were measured against the real data on 2026-08-29; they are not estimates.

---

## 1. Tech stack

| Layer | Choice | Version | Why |
|-------|--------|---------|-----|
| Language | Python | 3.10.11 (installed) | Matches the dev machine |
| DL framework | PyTorch | `torch>=2.2,<3` (CPU wheel) | Explicit `forward`; no Keras shape incantations |
| Numerics | NumPy | 2.2.6 (installed) | Torch 2.2+ supports numpy 2; the old TF 2.12 pin did not |
| Data | pandas | 2.3.3 (installed) | CSV load only |
| Metrics | scikit-learn | 1.7.2 (installed) | `f1_score`, `classification_report`, `confusion_matrix`, `roc_auc_score` |
| Config | PyYAML | 6.0.3 (installed) | YAML configs |
| Validation | Pydantic | 2.13.4 (installed) | Config schema **and** API schema — one validation system |
| API | FastAPI + Uvicorn | 0.141.1 / 0.52.3 (installed) | Already present; auto-generated OpenAPI |
| CLI | `argparse` (stdlib) | — | No new dependency for four subcommands |
| Progress | tqdm | to install | Training feedback |
| Frontend | Streamlit | 1.61.1 (installed) | Pure-Python UI, no build step, no JS toolchain |
| HTTP client | httpx | 0.28.1 (installed) | Frontend → backend calls |
| Charts | Altair | 6.2.2 (installed) | Ships with Streamlit; renders the temperature/entropy chart |
| Tests | pytest | 9.1.1 (installed) | — |

**To install:** `torch`, `tqdm`. Everything else — including Streamlit, httpx and Altair — is already on the machine.

**Deliberately absent:** `torchtext` (unmaintained, hard-pins torch), `nltk` (its stopword list is the cause of D3), `keras`/`tensorflow` (the thing being replaced), `transformers`, `lightning`. Rationale in `Rules.md` §2.

**Hardware target:** CPU-only. No CUDA on the dev machine. `device` is resolved once in `utils/device.py` and threaded through; nothing else calls `.cuda()`.

---

## 2. Repository layout

`modular_code/` and `notebook/` are **frozen**. Nothing in the new tree imports from them.

```
D:\Sentiment_Detection_and_Text Generation_with_Many-to-One\
│
├── PRD.md  Architecture.md  Rules.md  Phases.md      ← steering docs
├── Memory.md                                          ← created at end of Phase 0
├── LSTM part 2 Solution doc.pdf                       ← original course doc
│
├── data/                          ← SINGLE shared copy (canonical)
│   ├── airline_sentiment.csv      ← 11,541 rows
│   └── alice.txt                  ← 164,045 chars raw
│
├── modular_code/                  ← FROZEN. TensorFlow reference. Do not edit.
├── notebook/                      ← FROZEN. Tutorial notebook. Do not edit.
│
└── pytorch/
    ├── pyproject.toml             ← package metadata, pytest + ruff config
    ├── requirements.txt           ← pinned
    ├── README.md                  ← quickstart only; design lives in these .md files
    │
    ├── configs/
    │   ├── sentiment.yaml
    │   └── textgen.yaml
    │
    ├── src/lstm_nlp/
    │   ├── __init__.py            ← __version__, PREPROCESS_VERSION
    │   ├── cli.py                 ← argparse; train/eval/predict/generate
    │   ├── config.py              ← Pydantic config models + YAML loader
    │   ├── vocab.py               ← Vocab: build from counter, stoi/itos, <pad>/<unk>
    │   │
    │   ├── data/
    │   │   ├── preprocess.py      ← clean_tweet(), clean_book(), strip_gutenberg()
    │   │   ├── sentiment.py       ← SentimentDataset, collate_fn, stratified split
    │   │   └── textgen.py         ← WindowDataset (10→1), block split
    │   │
    │   ├── models/
    │   │   ├── sentiment_lstm.py  ← SentimentLSTM
    │   │   └── textgen_lstm.py    ← TextGenLSTM
    │   │
    │   ├── engine/
    │   │   ├── trainer.py         ← task-agnostic train loop
    │   │   ├── callbacks.py       ← EarlyStopping, BestCheckpoint
    │   │   └── metrics.py         ← classification metrics, perplexity, baselines
    │   │
    │   ├── inference/
    │   │   ├── checkpoint.py      ← save_checkpoint / load_checkpoint (+ version guard)
    │   │   ├── predictor.py       ← SentimentPredictor
    │   │   └── sampler.py         ← temperature + top_k sampling  ← D2 fix lives here
    │   │
    │   ├── api/
    │   │   ├── app.py             ← FastAPI app, lifespan model loading
    │   │   └── schemas.py         ← request/response Pydantic models
    │   │
    │   └── utils/
    │       ├── seed.py            ← set_seed()
    │       ├── device.py          ← resolve_device()
    │       └── logging.py         ← get_logger()
    │
    ├── tests/
    │   ├── test_preprocess.py     ← D3, D6 regression tests
    │   ├── test_vocab.py          ← D7: train-only build, <unk> mapping
    │   ├── test_datasets.py       ← shapes, padding, split integrity
    │   ├── test_models.py         ← forward shapes, packing ignores pad
    │   ├── test_sampler.py        ← D2: entropy monotonicity, T→0 == argmax
    │   ├── test_checkpoint.py     ← D8: self-contained round-trip
    │   ├── test_metrics.py        ← baselines
    │   └── test_api.py            ← all endpoints incl. 422 / 503
    │
    ├── frontend/                  ← Streamlit app. PURE API CLIENT (Rules C15)
    │   ├── app.py                 ← entry: st.navigation, backend health gate
    │   ├── api_client.py          ← the ONLY module that talks HTTP
    │   ├── settings.py            ← backend URL from env, never hardcoded
    │   ├── theme.py               ← palette + shared CSS (see Design.md)
    │   ├── components.py          ← reusable widgets: metric+baseline, prob bar
    │   ├── pages/
    │   │   ├── 1_sentiment.py     ← classify text, show probs + unk_rate
    │   │   └── 2_generation.py    ← seed, temperature slider, entropy chart
    │   └── .streamlit/config.toml ← theme tokens
    │
    └── runs/                      ← gitignored: checkpoints, history.json, logs
        ├── sentiment/<timestamp>/
        └── textgen/<timestamp>/
```

**Import rule:** dependencies point downward only.
`cli`/`api` → `inference` → `engine` → `models` → `data` → `vocab`/`utils`.
`models/` never imports from `data/`. `data/` never imports `torch.nn`.

**Tier rule:** `frontend/` imports **nothing** from `lstm_nlp`. Not the models, not the vocab, not torch. Its only contract with the rest of the system is the HTTP surface in §6. A test asserts this (`Rules.md` C15, PRD S19).

```
┌─────────────┐   HTTP/JSON   ┌─────────────┐   in-process   ┌──────────────┐
│  Streamlit  │ ────────────▶ │   FastAPI   │ ─────────────▶ │  lstm_nlp    │
│  frontend/  │ ◀──────────── │  api/app.py │ ◀───────────── │  checkpoints │
└─────────────┘               └─────────────┘                └──────────────┘
   presentation                  contract                     model + data
   no torch                      validation                   the only place
   no checkpoints                one load at startup          inference happens
```

---

## 3. Data flow

### 3.1 Task A — Sentiment (many-to-one classification)

```
data/airline_sentiment.csv   11,541 rows · 9,178 neg / 2,363 pos
        │
        ▼
clean_tweet()                lowercase · http…→<url> · @x→<user> · &amp;→and
  [FR-3 · fixes D3]          strip punct EXCEPT apostrophes   ← negations survive
        │                    median 20 tok · p95 27 · max 35
        ▼
stratified split 70/30       random_state=10 · train pos 0.2048 · test pos 0.2047
  [FR-6]                     8,078 train / 3,463 test
        │
        ├──── TRAIN ONLY ───▶ Vocab.build(min_freq=2)   [FR-4 · fixes D7]
        │                     9,566 raw → 4,505 kept (incl <pad>=0, <unk>=1)
        │                     test OOV = 5.23% of tokens → all map to <unk>
        ▼
encode + truncate to max_len=30  ·  int64 indices, NEVER one-hot  [FR-8]
        │
        ▼
DataLoader(batch=64, collate_fn=pad_to_longest_in_batch)
        │                    lengths kept for pack_padded_sequence
        ▼
┌───────────────────────────────────────────────────────┐
│ SentimentLSTM                                          │
│   Embedding(4505, 64, padding_idx=0)          288,320 │
│   LSTM(64→64, 2 layers, dropout=0.3)           66,560 │
│   Dropout(0.4)                                        │
│   Linear(64, 2)                                   130 │
│                                    TOTAL      355,010 │
│   forward → RAW LOGITS (2,)   [FR-11]                 │
└───────────────────────────────────────────────────────┘
        │
        ▼
CrossEntropyLoss(weight=[1.0, 3.884])   [FR-14 · fixes D4]
Adam(lr=1e-3) · clip_grad_norm_(5.0)
EarlyStopping(monitor=val_macro_f1, mode=max, patience=5)  [FR-13 · fixes D5]
        │
        ▼
best-epoch checkpoint  +  vocab  +  preprocess cfg  →  .pt   [FR-25 · fixes D8]
        │
        ▼
eval: accuracy · MACRO-F1 · per-class P/R/F1 · confusion · ROC-AUC
      each printed beside its baseline (acc 0.795 · macroF1 0.443)  [FR-18/20]
```

### 3.2 Task B — Text generation (many-to-one next-word LM)

```
data/alice.txt                        164,045 chars
        │
        ▼
strip_gutenberg()                     cut at *** START/END *** markers
  [FR-2 · fixes D6]                   → 144,607 chars (88.2% kept)
        │                             removes the license block that was
        │                             previously 100% of the validation set
        ▼
clean_book()                          lowercase · keep apostrophes · collapse ws
        │                             27,429 tokens · 2,578 raw vocab
        ▼
Vocab.build(min_freq=1)  TRAIN ONLY   → 2,436 (incl <unk>)
        │
        ▼
BLOCK split BEFORE windowing          contiguous 90/10 on real prose
  [FR-7 · fixes D6]                   no window straddles the boundary
        │
        ▼
WindowDataset(seq_len=10, stride=1)   27,409 windows (24,677 + 2,732)
        │                             lazy slices of one 1-D tensor: 0.22 MB
        │                             ── dense (N,10) int64 would be 2.19 MB
        │                             ── reference one-hot was 931 MB
        │                             ── >2,000× smaller  [FR-8 · fixes D9]
        ▼
┌───────────────────────────────────────────────────────┐
│ TextGenLSTM                                            │
│   Embedding(2436, 128)                        311,808 │
│   LSTM(128→256, 1 layer, batch_first)         395,264 │
│   take h_n[-1]  ← the "many-to-one" reduction         │
│   Linear(256, 2436)                           626,052 │
│                                    TOTAL    1,333,124 │
│   forward → RAW LOGITS (V,)   [FR-11]                 │
└───────────────────────────────────────────────────────┘
        │
        ▼
CrossEntropyLoss · Adam(lr=2e-3) · clip 5.0
EarlyStopping(monitor=val_loss, mode=min, patience=5)
        │
        ▼
eval: val CE + perplexity, beside uniform baseline ln(2436)=7.798 → ppl 2,436
        │
        ▼
generate(seed, n, T, top_k)  ← sampler.py, operates on LOGITS  [FR-21 · fixes D2]
```

### 3.3 The D2 fix, precisely

The single most important correctness change in the port.

```
TF (WRONG)                              PyTorch (CORRECT)
──────────────────────────              ──────────────────────────
logits = model(x)                       logits = model(x)
p = softmax(logits)        ← probs      # no softmax in the model  [FR-11]
                                        if top_k: logits = mask_topk(logits, k)
tf.random.categorical(p / T)            probs = softmax(logits / T)
   └─ expects LOGITS, applies           idx = torch.multinomial(probs, 1)
      its own softmax
                                        T→0   ⇒ argmax   (asserted, S5)
p/10 ∈ [0, 0.007]                       T↑    ⇒ entropy↑ (asserted, S5)
  ⇒ softmax ≈ UNIFORM
  ⇒ samples random words
```

Temperature is applied to logits **once**, and the model never softmaxes. Making `forward` return logits is what makes the bug structurally unrepresentable.

---

## 4. Module responsibilities

| Module | Owns | Must not |
|--------|------|----------|
| `config.py` | Pydantic config schema, YAML load, defaults | Read files other than the config |
| `vocab.py` | `stoi`/`itos`, `<pad>`=0/`<unk>`=1, freq filter, `encode`/`decode` | Know about torch or any task |
| `data/preprocess.py` | Pure `str → str` cleaning + Gutenberg stripping | Import torch; hold state |
| `data/sentiment.py` | Dataset, stratified split, padding collate | Build vocab from test data |
| `data/textgen.py` | Block split, sliding windows | Materialise one-hot tensors |
| `models/*.py` | `nn.Module` definitions returning logits | Softmax; load data; touch files |
| `engine/trainer.py` | Epoch loop, optim step, clipping, history | Know which task it runs |
| `engine/callbacks.py` | Early stopping, best-checkpoint tracking | Write final-epoch weights |
| `engine/metrics.py` | Metrics **and their baselines** | Report a metric without its baseline |
| `inference/checkpoint.py` | Self-contained save/load + version guard | Load a mismatched preprocess version silently |
| `inference/sampler.py` | Temperature + top-k on logits | Receive probabilities |
| `inference/predictor.py` | Checkpoint → usable predictor | Re-read raw data |
| `api/app.py` | Routes, lifespan loading | Load models per request |
| `cli.py` | Arg parsing, dispatch | Contain training logic |

---

## 5. Artifact formats

### 5.1 Checkpoint (`.pt`) — self-contained by contract [FR-25]

The direct fix for **D8**: the old `.h5` is unusable because the vocab that defines its input space was never saved.

```python
{
  "format_version": 1,
  "task": "sentiment" | "textgen",
  "created_utc": "2026-08-29T18:42:11Z",
  "lib_versions": {"python": "3.10.11", "torch": "2.x", "numpy": "2.2.6",
                   "lstm_nlp": "0.1.0"},

  "model_class": "SentimentLSTM",
  "model_cfg":   {"vocab_size": 4505, "embed_dim": 64, "hidden_dim": 64,
                  "num_layers": 2, "dropout": 0.3, "num_classes": 2},
  "model_state": {...},                    # state_dict of the BEST epoch

  "vocab": {"itos": ["<pad>", "<unk>", ...], "min_freq": 2},

  "preprocess": {"version": "1",           # guarded on load  [FR-26]
                 "max_len": 30, "lowercase": true,
                 "url_token": "<url>", "user_token": "<user>"},

  "train": {"best_epoch": 7, "seed": 42, "class_weights": [1.0, 3.884],
            "stopped_early": true},
  "metrics": {"test_accuracy": ..., "test_macro_f1": ...,
              "baseline_accuracy": 0.7953, "baseline_macro_f1": 0.4430}
}
```

`load_checkpoint()` raises `PreprocessVersionMismatch` when `preprocess.version != PREPROCESS_VERSION`. Bump `PREPROCESS_VERSION` in `__init__.py` on **any** change to tokenisation — old checkpoints then fail loudly instead of mis-tokenising.

### 5.2 Run directory

```
runs/sentiment/2026-08-29T18-42-11/
├── best.pt          ← best epoch  [FR-13]
├── history.json     ← per-epoch train/val loss + metrics  [FR-16]
├── config.yaml      ← resolved config actually used
├── metrics.json     ← final test metrics + baselines
└── train.log
```

### 5.3 Config (`configs/sentiment.yaml`)

Every literal that was hardcoded in the TF version [FR-12, fixes D10]:

Relative paths resolve against **the config file's own directory** (`pytorch/configs/`) — one rule everywhere, so a config never depends on the current working directory.

```yaml
task: sentiment
seed: 42
data:   {csv: ../../data/airline_sentiment.csv, test_size: 0.30,
         split_seed: 10, stratify: true, min_freq: 2, max_len: 30}
model:  {embed_dim: 64, hidden_dim: 64, num_layers: 2, dropout: 0.3}
train:  {batch_size: 64, epochs: 40, lr: 1.0e-3, weight_decay: 0.0,
         clip_grad_norm: 5.0, class_weighting: balanced,
         early_stopping: {monitor: val_macro_f1, mode: max, patience: 5}}
output: {dir: ../runs/sentiment}
```

---

## 6. HTTP API contract

FastAPI, mounted by `api/app.py`. Models load once in the `lifespan` handler [FR-29].

### `GET /health`
```json
{"status": "ok", "models": {"sentiment": true, "textgen": true}, "device": "cpu"}
```
`200` when at least one model is loaded, `503` when none are.

### `GET /models`
Returns each loaded checkpoint's `task`, `model_cfg`, `vocab_size`, `metrics`, `created_utc`, `lib_versions`.

### `POST /predict`
```json
// request
{"text": "the flight was not great"}

// 200
{"label": "negative",
 "label_id": 0,
 "probabilities": {"negative": 0.87, "positive": 0.13},
 "n_tokens": 5,
 "n_unk": 0,
 "unk_rate": 0.0}
```
`n_unk`/`unk_rate` are returned deliberately — a caller must be able to see when a prediction rests on mostly-unknown tokens.

### `POST /predict/batch`
`{"texts": [...]}` → `{"predictions": [...]}`. Max 256 items (422 beyond).

### `POST /generate`
```json
// request  — only `seed` is required
{"seed": "alice was beginning to",
 "n_words": 40,
 "temperature": 0.7,     // > 0, ≤ 5.0
 "top_k": 40,            // optional, ≥ 1
 "rng_seed": 123}        // optional → reproducible  [FR-23]

// 200
{"text": "alice was beginning to get very tired of sitting by her sister ...",
 "seed_tokens": ["alice", "was", "beginning", "to"],
 "generated_tokens": [...],
 "temperature": 0.7,
 "n_unk_in_seed": 0}
```
Seed words absent from the vocabulary become `<unk>` rather than erroring [FR-24].

### Errors [FR-30]

| Status | When | Body |
|--------|------|------|
| 422 | Pydantic validation (empty text, `temperature ≤ 0`, batch > 256) | FastAPI validation detail |
| 503 | Required checkpoint not loaded | `{"detail": "textgen model not loaded; train it first"}` |
| 500 | never intentionally | — |

Validation lives in `schemas.py` as Pydantic `Field` constraints, so the OpenAPI docs at `/docs` stay accurate for free.

---

## 7. Frontend architecture

Streamlit, two pages, one HTTP client. The whole app is a view over §6 — it holds no model state and makes no decisions the backend hasn't already made.

### 7.1 Module contract

| Module | Owns | Must not |
|--------|------|----------|
| `settings.py` | Backend base URL + timeouts from env (`LSTM_API_URL`, default `http://127.0.0.1:8000`) | Hardcode a URL (PRD FR-36) |
| `api_client.py` | **Every** HTTP call. Typed wrappers `health()`, `models()`, `predict()`, `generate()`. Converts transport failures into a typed `BackendError` | Contain UI code |
| `theme.py` | Palette tokens, shared CSS injection | Hold logic |
| `components.py` | `metric_with_baseline()`, `probability_bars()`, `unk_badge()`, `backend_status()` | Call HTTP directly |
| `pages/*.py` | Layout, widgets, calling `api_client` | Import torch or `lstm_nlp` |

### 7.2 Page flows

```
1_sentiment.py
  text_area ──▶ api_client.predict(text)
                    │
                    ▼
              label + probability bars
              unk_rate badge   ← "3 of 12 tokens unknown"
              baseline note    ← "majority-class baseline: 0.795"

2_generation.py
  seed ─┐
  n     ├─▶ api_client.generate(...) ──▶ generated text (seed styled distinctly)
  T ────┤
  top_k ┘         │
  rng ──┘         └─▶ next-word probability chart at T
                      ── the D2 lesson, made visible (FR-34)
```

### 7.3 The temperature control

The reason the frontend exists. The reference's temperature code sampled uniformly at random and its notebook presented that as a demonstration of softmax temperature (D2). Here the slider drives a live Altair chart of the next-word distribution, so the relationship is observable rather than asserted:

| T | Distribution | Text |
|---|---|---|
| 0.1 | one spike | repetitive, near-greedy |
| 0.7 | few plausible peaks | coherent, varied — the default |
| 1.5 | flattening | inventive, drifting |
| 2.0 | near-uniform | incoherent — *this is what the reference produced at every setting* |

### 7.4 Error states (FR-35)

| Condition | UI |
|---|---|
| Backend unreachable | Full-page banner: what failed, the URL tried, the command to start it. No spinner, no traceback. |
| Model not loaded (503) | Page-level notice naming the missing model and the `train` command that produces it. |
| Validation error (422) | Inline message on the offending control. |
| Slow response | Spinner with a timeout, then a retry affordance. |

Startup gates on `GET /health`; every page renders a backend-status indicator.

---

## 8. TF → PyTorch mapping

| Keras (frozen `modular_code/`) | PyTorch | Note |
|---|---|---|
| `Sequential([...])` | `nn.Module` subclass with explicit `forward` | Removes shape guesswork |
| `Embedding(input_dim, output_dim, input_length)` | `nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)` | `input_length` unnecessary; `padding_idx` is new and freezes the pad row |
| `LSTM(units, return_sequences=True)` | `nn.LSTM(...)` → use `output` | — |
| `LSTM(units, return_sequences=False)` | `nn.LSTM(...)` → use `h_n[-1]` | This *is* the many-to-one reduction |
| stacked `LSTM` layers | `num_layers=2` | Single module |
| `Dense(2, activation='softmax')` + `categorical_crossentropy` | `nn.Linear(64, 2)` + `nn.CrossEntropyLoss` | CE takes **logits**; softmax removed from the model — **fixes D2 structurally** |
| `to_categorical(y)` | int64 class indices | `CrossEntropyLoss` wants indices |
| `pad_sequences(maxlen, padding='post')` | `pad_sequence` in `collate_fn` + `pack_padded_sequence` | Pads per batch, not globally: the old code padded everything to 26 (58.1% zeros) |
| one-hot `(N, 10, V)` input | `nn.Embedding` over int indices | 931 MB → 0.22 MB (lazy slices of one 1-D tensor) |
| `validation_split=0.1` | explicit block split | Old semantics = trailing slice = license text — **fixes D6** |
| `model.fit(epochs=50)` | `Trainer.fit()` + `EarlyStopping` | **fixes D5** |
| `model.save('*.h5')` | `torch.save(bundle, '*.pt')` | Bundle carries the vocab — **fixes D8** |
| `tf.random.categorical(p / T)` | `multinomial(softmax(logits / T))` | **fixes D2** |
| `nltk.stopwords` removal | *removed entirely* | **fixes D3**; also removes the import-time network call (NFR-7) |
| `projectpro.checkpoint/save_point` | *removed* | Network calls in the training critical path |

---

## 9. Test strategy

`tests/` runs in < 60 s with **no training and no network** (NFR-6). Fixtures use a ~200-row CSV slice and a ~2,000-token text slice committed under `tests/fixtures/`.

Each defect gets a named regression test, so a fix cannot silently regress:

| Defect | Test |
|--------|------|
| D1 | `test_cli.py::test_generate_command_signature` — CLI dispatch invoked with real args |
| D2 | `test_sampler.py::test_entropy_monotonic_in_temperature`, `::test_low_temperature_equals_argmax` |
| D3 | `test_preprocess.py::test_negations_survive_cleaning` |
| D4 | `test_metrics.py::test_report_includes_baselines` |
| D5 | `test_callbacks.py::test_best_not_last_checkpoint_restored` |
| D6 | `test_preprocess.py::test_gutenberg_boilerplate_stripped` |
| D7 | `test_vocab.py::test_vocab_built_from_train_only`, `::test_unknown_word_maps_to_unk` |
| D8 | `test_checkpoint.py::test_checkpoint_is_self_contained` (loads in a subprocess with only the `.pt`) |
| D9 | `test_datasets.py::test_windows_are_int_indices_not_onehot` |
