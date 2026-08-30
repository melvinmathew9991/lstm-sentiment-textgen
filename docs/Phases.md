# Phases — PyTorch Migration Plan

Ten phases, each independently verifiable. Do not start a phase before the previous one's exit criteria pass (`Rules.md` A2).
Every phase ends with **all four** of:

1. Run the phase's own **Verify** command; paste the real output.
2. Run the **end-to-end project audit** — `python scripts/audit.py` — and paste its summary. Zero FAIL is required to merge (`Rules.md` §11).
3. Update `Memory.md` and `CHANGELOG.md`.
4. Branch → PR → squash-merge → annotated tag (`Rules.md` §10).

Step 2 is what keeps the standards project-wide rather than phase-local: it re-checks every earlier phase, every document and the git history, not just the code just written.

**Dependency chain**

```
P0 Scaffold
   └─▶ P1 Data layer ──┬─▶ P2 Sentiment model+train ─┐
                       └─▶ P3 Text-gen model+train ──┤
                                    │                │
                                    └─▶ P4 Sampling ─┤
                                                     ├─▶ P5 CLI ─▶ P6 API ─▶ P7 Streamlit UI
                                                     │                            │
                          (P2 and P3 are independent)               P8 Parity ─▶ P9 Harden

tiers:  lstm_nlp core (P1-P5) ──▶ FastAPI backend (P6) ──▶ Streamlit frontend (P7)
```

Effort is a rough sizing for one focused session each, not a schedule.

---

## Phase 0 — Scaffold & environment

**Goal:** an installable, testable, empty package. No ML.
**Effort:** S · **Depends on:** nothing

### Tasks
1. Create the `pytorch/` tree from `Architecture.md` §2 — all directories, all `__init__.py`.
2. `pyproject.toml`: package metadata, `src/` layout, pytest + ruff config.
3. `requirements.txt`, pinned. **Only `torch` and `tqdm` actually need installing** — fastapi, uvicorn, pydantic, pytest, sklearn, pandas, numpy, PyYAML are already on the machine.
4. `utils/seed.py` (`set_seed`), `utils/device.py` (`resolve_device`), `utils/logging.py` (`get_logger`).
5. `errors.py`: `LstmNlpError` + `PreprocessVersionMismatch`, `CheckpointError`, `ConfigError`.
6. `config.py`: Pydantic models for the two config schemas + YAML loader.
7. `configs/sentiment.yaml`, `configs/textgen.yaml` — values as specified in `Architecture.md` §5.3.
8. `cli.py` skeleton: four subcommands, `--help` only, all handlers raise `NotImplementedError`.
9. `.gitignore` covering `runs/`, `__pycache__/`, `*.pt`.
10. Create `Memory.md` at repo root with the Phase 0 entry.

### Exit criteria
- `pip install -e pytorch/` succeeds.
- `python -c "import torch; print(torch.__version__)"` works.
- `python -m lstm_nlp.cli --help` lists all four subcommands.
- Loading each YAML returns a validated config object; a deliberately bad value raises `ValidationError`.
- `pytest` collects and passes (3 trivial tests: seed determinism, device resolution, config round-trip).

### Verify
```bash
cd pytorch && pip install -e . && python -m lstm_nlp.cli --help && pytest -q
```

---

## Phase 1 — Data layer

**Goal:** preprocessing, vocab and datasets, with the audited numbers reproduced as assertions.
**Effort:** M · **Depends on:** P0 · **Closes:** D3, D6, D7, D9 (partial)

### Tasks
1. `data/preprocess.py`
   - `strip_gutenberg(raw) -> str` — cut on `*** START/END OF THE PROJECT GUTENBERG EBOOK ***`. **(D6)**
   - `clean_tweet(s) -> str` — lowercase, `http…`→`<url>`, `@x`→`<user>`, `&amp;`→`and`, drop punctuation **except apostrophes**, collapse whitespace. **No stopword removal. (D3)**
   - `clean_book(s) -> str`.
   - Must import without torch installed (`Rules.md` §4).
2. `vocab.py` — `Vocab.build(counter, min_freq)`, `encode`, `decode`, `__len__`, `to_dict`/`from_dict`. `<pad>`=0, `<unk>`=1. Misses → `<unk>`. **(D7, C4)**
3. `data/sentiment.py` — load CSV, clean, **stratified** 70/30 at `random_state=10`, build vocab from **train only**, `SentimentDataset` returning `(ids, length, label)`, `collate_fn` padding to the longest in batch + returning lengths.
4. `data/textgen.py` — load, strip, clean, **contiguous block** 90/10 split *before* windowing (no window straddles the boundary), `WindowDataset(seq_len=10)` returning int64 `(10,)` + scalar target. **Never materialise one-hot. (D6, D7, D9)**
5. `tests/fixtures/` — ~200-row CSV slice, ~2,000-token text slice.
6. Tests asserting the measured values below.

### Expected values (measured 2026-08-29 — assert these)

| Quantity | Value |
|---|---|
| Sentiment rows | 11,541 (9,178 neg / 2,363 pos) |
| Rows | 11,541 loaded · **11,271** after deduplication (v1.1.0) |
| Train / test | 7,889 / 3,382 · pos rate 0.1949 / 0.1952 |
| — train splits again (Phase 8) | 6,705 train / 1,184 val · pos rate 0.1949 / 0.1951 |
| Token length | median 20 · p95 27 · max 35 |
| Train-only raw vocab | **8,702** on the 6,705-row training block (9,566 before deduplication) |
| Vocab @ `min_freq=2` | **4,045** on the 6,705-row training block the model uses (4,505 on the full un-deduplicated train split) |
| Test OOV rate | **5.77%** at V=4,045 |
| Class weight (pos) | 4.130 (3.884 before deduplication) |
| Alice raw → stripped | 164,045 → 144,607 chars (88.2% kept) |
| Alice tokens | **27,429** (was 30,674 unstripped) |
| Alice vocab, **train-only** @ `min_freq=1` | **2,436** (incl. `<unk>`) |
| Windows, three blocks @ seq_len 10 | **27,399** (= 27,429 − 3×seq_len) |
| Storage (lazy int64) | **0.22 MB** · dense `(N,10)` would be 2.19 MB |
| Sentiment train `<unk>` rate | **3.70%** — must be > 0 so `<unk>` gets trained (3.38% before deduplication) |

> **Corrected 2026-08-29 during implementation.** The plan's *1,470 vocab / 27,419 windows / 707 MB*
> were computed on the **full corpus before splitting**. The code correctly builds the vocabulary from
> the training block only and windows each block independently, so the right figures are the ones above.
> `min_freq` for textgen moved 2 → 1: at 2, **4.34% of training targets were `<unk>`**, which the model
> would learn to emit. Sentiment keeps `min_freq=2`.

### Exit criteria
- All values above reproduced by tests.
- `test_negations_survive_cleaning`: `"the flight was not great"` → cleaned text still contains `not`. **(D3)**
- `test_gutenberg_boilerplate_stripped`: `copyright`, `donations`, `gutenberg-tm` absent from the cleaned corpus. **(D6)**
- `test_vocab_built_from_train_only`: a token unique to test is not in the vocab. **(D7)**
- `test_unknown_word_maps_to_unk`: `encode("qwertyuiop")` → `[1]`, no raise. **(D7)**
- `test_windows_are_int_indices_not_onehot`: `X.dtype == torch.int64` and `X.ndim == 2`. **(D9)**
- Dataset construction peak RSS < 200 MB.

### Verify
```bash
pytest tests/test_preprocess.py tests/test_vocab.py tests/test_datasets.py -v
```

---

## Phase 2 — Sentiment model & training

**Goal:** a trained sentiment classifier with honest metrics.
**Effort:** M · **Depends on:** P1 · **Closes:** D4, D5, D8, D11

### Tasks
1. `models/sentiment_lstm.py` — `Embedding(V,64,padding_idx=0)` → 2-layer `LSTM(64,64)` → `Dropout(0.4)` → `Linear(64,2)`. `pack_padded_sequence` around the LSTM. **Returns logits. (C1, C10)** **325,570** params at the V=4,045 the model trains with (355,010 at the V=4,505 of the pre-deduplication two-way split; 328,002 at the V=4,083 between the two).
2. `engine/metrics.py` — `classification_metrics()` returning accuracy, macro-F1, per-class P/R/F1, confusion matrix, ROC-AUC, **each beside its baseline** (acc 0.8048, macro-F1 0.4459 — computed from the labels, never hardcoded). **(D4, C11)**
3. `engine/callbacks.py` — `EarlyStopping(monitor, mode, patience)` and `BestCheckpoint` holding the best `state_dict` in memory and restoring it at the end. **(D5, C12)**
4. `engine/trainer.py` — task-agnostic loop: epochs, `clip_grad_norm_(5.0)`, val each epoch, callbacks, `history.json`, tqdm.
5. `inference/checkpoint.py` — `save_checkpoint`/`load_checkpoint` per the `Architecture.md` §5.1 bundle, with the preprocess-version guard. **(D8, FR-26)**
6. Wire `train`/`eval` for the sentiment task; `CrossEntropyLoss(weight=[1.0, 4.130])`.

### Exit criteria
- Trains to completion on CPU in **< 5 min** (NFR-2).
- Test **macro-F1 ≥ 0.75** (S3); baseline 0.4459 printed beside it.
- Early stopping fires before the 40-epoch cap (S14) — proves D5 closed.
- `test_best_not_last_checkpoint_restored`: with a synthetic degrading val curve, the restored weights are the best epoch's. **(D5)**
- `test_checkpoint_is_self_contained`: a **subprocess** with only `best.pt` reachable loads it and reproduces the recorded metrics to 1e-6. **(D8, S6)**
- `test_report_includes_baselines`. **(D4)**
- **Negation test (S8):** `"the flight was great"` vs `"the flight was not great"` → different labels, or a positive-probability gap > 0.15. This is the payoff for D3.

### Verify
```bash
python -m lstm_nlp.cli train --config configs/sentiment.yaml
python -m lstm_nlp.cli eval  --ckpt runs/sentiment/<ts>/best.pt
pytest tests/test_models.py tests/test_checkpoint.py tests/test_metrics.py -v
```

---

## Phase 3 — Text-generation model & training

**Goal:** a trained next-word LM. Independent of P2.
**Effort:** M · **Depends on:** P1 · **Closes:** D6, D9

### Tasks
1. `models/textgen_lstm.py` — `Embedding(V,128)` → `LSTM(128,256,batch_first=True)` → take `h_n[-1]` → `Linear(256,V)`. **Returns logits.** Target ≈ **1,333,124** params at V=2,436.
2. Perplexity in `engine/metrics.py`, beside the uniform baseline `ln(2436)=7.798` → ppl 2,436.
3. Reuse `engine/trainer.py` unchanged — if it needs a task-specific branch, the abstraction is wrong; fix the abstraction.
4. Wire `train`/`eval` for text-gen; `EarlyStopping(val_loss, min, patience=5)`.

### Exit criteria
- Trains to completion on CPU in **< 20 min** (NFR-3).
- Validation **perplexity ≤ 400** vs the 2,436 uniform baseline (S4).
- Peak training RSS **< 2 GB**, measured and recorded (S9) — proves D9 closed.
- Early stopping fires (S14).
- Trainer required no task-specific special-casing.

### Verify
```bash
python -m lstm_nlp.cli train --config configs/textgen.yaml
python -m lstm_nlp.cli eval  --ckpt runs/textgen/<ts>/best.pt
```

---

## Phase 4 — Sampling & generation

**Goal:** the D2 fix, proven by property tests. **The most important phase in the plan.**
**Effort:** S · **Depends on:** P3 · **Closes:** D2, D10

### Tasks
1. `inference/sampler.py`
   - `apply_top_k(logits, k) -> logits` — mask below the k-th value with `-inf`. Applied **before** temperature.
   - `sample_from_logits(logits, temperature, top_k, generator) -> int` — `softmax(logits / T)` then `multinomial`. **Operates on logits. (C2)**
   - Accepts an explicit `torch.Generator` for reproducibility (FR-23).
2. `inference/predictor.py` — `TextGenerator.generate(seed, n_words, temperature, top_k, rng_seed)`. Seed shorter than `seq_len` left-pads; longer truncates to the last 10. Unknown seed words → `<unk>`, counted, not fatal (FR-24).
3. `SentimentPredictor.predict(text)` returning label, probabilities, `n_unk`, `unk_rate`.
4. No magic indices — the demo seed is a config value, not `input_words[-28701]`. **(D10)**

### Exit criteria — S5, the headline gate
- `test_entropy_monotonic_in_temperature`: sampled-distribution entropy **strictly increases** across T ∈ {0.2, 0.5, 1.0, 1.5, 2.0}.
- `test_low_temperature_equals_argmax`: T=0.01 matches greedy argmax on **≥ 99%** of 1,000 draws.
- `test_top_k_restricts_support`: with `top_k=5`, 10,000 draws produce at most 5 distinct tokens.
- `test_same_rng_seed_same_output`: identical text for identical `rng_seed`.
- `test_unknown_seed_word_does_not_raise`.
- **S15 (qualitative):** generate 5 passages at T=0.7. None contains `copyright`, `donations`, `foundation`, `ebook`, `license` — proves D6 closed. Compare against the reference's uniform-noise output (`Memory.md`); the difference should be obvious at a glance.

### Verify
```bash
pytest tests/test_sampler.py -v
python -m lstm_nlp.cli generate --ckpt runs/textgen/<ts>/best.pt \
       --seed "alice was beginning to" --n-words 40 --temperature 0.7
```

---

## Phase 5 — CLI

**Goal:** the four subcommands working end to end.
**Effort:** S · **Depends on:** P2, P4 · **Closes:** D1, D10

### Tasks
1. Implement all four handlers in `cli.py`:
   - `train --config PATH [--max-steps N]` (`--max-steps` for smoke tests, per `Rules.md` B7)
   - `eval --ckpt PATH [--split test]`
   - `predict --ckpt PATH TEXT...`
   - `generate --ckpt PATH --seed TEXT [--n-words] [--temperature] [--top-k] [--rng-seed]`
2. Resolve every path via `pathlib` from config or `__file__` — the CLI must work from **any** working directory.
3. Sensible exit codes: 0 ok, 1 runtime error, 2 bad args.

### Exit criteria
- All four run clean from a directory other than `pytorch/`.
- `tests/test_call_signatures.py`: every call site in the package is checked
  against the arity of the function it names, so the argument-count regression
  that killed the reference (**D1**) cannot recur *anywhere* — not merely at the
  one call that broke. The checker carries negative controls, including a
  reproduction of `train.generate_paragraph(model, test_words, 12, 10)` that it
  is required to flag.
- `--help` is accurate for every subcommand.
- Bad `--ckpt` gives a clear message, not a traceback.

### Verify
```bash
cd / && python -m lstm_nlp.cli predict  --ckpt <abs>/best.pt "the flight was not great"
        python -m lstm_nlp.cli generate --ckpt <abs>/best.pt --seed "alice was" --n-words 20
pytest tests/test_cli.py -v
```

---

## Phase 6 — FastAPI service

**Goal:** the HTTP contract from `Architecture.md` §6.
**Effort:** M · **Depends on:** P5

### Tasks
1. `api/schemas.py` — Pydantic request/response models with `Field` constraints: non-empty `text`, `0 < temperature ≤ 5.0`, `top_k ≥ 1`, `1 ≤ n_words ≤ 200`, batch ≤ 256.
2. `api/app.py` — `lifespan` loads both checkpoints **once** (FR-29); routes `GET /health`, `GET /models`, `POST /predict`, `POST /predict/batch`, `POST /generate`; exception handlers producing 422 / 503 / generic 500 (FR-30).
3. Checkpoint paths from env vars with config defaults.
4. `tests/test_api.py` using `TestClient` with tiny fixture checkpoints.

### Exit criteria
- `uvicorn lstm_nlp.api.app:app` starts; `/docs` renders.
- Every endpoint has a passing test **including** the 422 (bad temperature, empty text, oversized batch) and 503 (missing checkpoint) paths (S12).
- `/predict` warm latency **< 100 ms**; `/generate` at 40 words **< 2 s** (NFR-5) — measured and recorded.
- Models load once: a log assertion proves no per-request loading.
- No stack trace ever reaches an HTTP response body.

### Verify
```bash
pytest tests/test_api.py -v
uvicorn lstm_nlp.api.app:app --port 8000 &
curl -s 127.0.0.1:8000/health
curl -s -X POST 127.0.0.1:8000/predict  -H 'Content-Type: application/json' \
     -d '{"text":"the flight was not great"}'
curl -s -X POST 127.0.0.1:8000/generate -H 'Content-Type: application/json' \
     -d '{"seed":"alice was beginning to","n_words":30,"temperature":0.7}'
curl -s -X POST 127.0.0.1:8000/predict  -H 'Content-Type: application/json' \
     -d '{"text":""}'                       # expect 422
```

---

## Phase 7 — Streamlit frontend

**Goal:** the presentation tier — a two-page app that makes the models usable, and makes temperature *visible*.
**Effort:** M · **Depends on:** P6 · **Closes:** — (delivers PRD §4.8)

### Tasks
1. `frontend/settings.py` — backend URL from `LSTM_API_URL`, default `http://127.0.0.1:8000`, with timeouts. No hardcoded literal (FR-36).
2. `frontend/api_client.py` — the **only** module making HTTP calls. `health()`, `models()`, `predict()`, `generate()`; transport failures become a typed `BackendError`.
3. `frontend/theme.py` + `.streamlit/config.toml` — palette and typography per `Design.md`.
4. `frontend/components.py` — `metric_with_baseline()`, `probability_bars()`, `unk_badge()`, `backend_status()`.
5. `frontend/app.py` — entry point, navigation, health gate.
6. `pages/1_sentiment.py` — text input → label, probability bars, `unk_rate` badge, baseline note (FR-32).
7. `pages/2_generation.py` — seed, word count, **temperature slider**, top-k, RNG seed; generated text plus the next-word probability chart at the chosen T (FR-33, FR-34).
8. Error states for unreachable backend / 503 / 422 (FR-35).
9. `tests/test_frontend.py` — client tests against a mocked backend, plus the C15 import assertion.

### Exit criteria
- Both pages work end to end against a live backend (**S17**).
- Moving the temperature slider visibly changes **both** the generated text and the probability chart — the D2 lesson, demonstrable (FR-34).
- Killing the backend yields a clear "unreachable" banner naming the URL and the start command — no traceback, no infinite spinner (**S18**).
- `test_frontend_imports_no_model_code`: nothing under `frontend/` imports `torch`, `lstm_nlp.models`, or loads a checkpoint (**S19**, Rules C15).
- Every metric shown carries its baseline (Rules C16).
- Slider → updated result in < 2 s warm (NFR-9).
- Frontend + backend start with two commands, no build step (NFR-10).

### Verify
```bash
uvicorn lstm_nlp.api.app:app --port 8000 &
streamlit run frontend/app.py
pytest tests/test_frontend.py -v
```

---

## Phase 8 — Parity & reporting

**Goal:** an honest written comparison against the frozen TF reference.
**Effort:** S · **Depends on:** P2, P3, P4, P7 · **Closes:** D11

### Tasks
1. `PARITY.md` at repo root:
   - PyTorch metrics vs the reference's recorded numbers, **each beside its baseline**.
   - Every deviation explained. Notably: the Keras 0.909 val accuracy is **not** a target (`PRD.md` §6.3) — it was produced on negation-stripped text and reported without a baseline.
   - Side-by-side generated text: reference (uniform noise, notebook cell 130) vs PyTorch at T ∈ {0.3, 0.7, 1.2}. This is the clearest single demonstration that D2 is fixed.
   - Resource table: reference one-hot 931 MB → lazy int64 0.22 MB; wall-clock; peak RSS.
   - Table of D1–D11, each marked closed with a link to its regression test.
2. `pytorch/README.md` — quickstart only (install, train, serve). Design detail stays in these four docs.
3. Fix the **D11** terminology everywhere in the new code and docs: integer-index sequences into a learned embedding, never "bag of words".

### Exit criteria
- `PARITY.md` exists; all 11 defects marked closed with test names.
- Every claimed number is reproducible by a pasted command.
- No "bag of words" in the new tree.

---

## Phase 9 — Hardening (optional)

**Goal:** the things worth doing only once everything works.
**Effort:** M · **Depends on:** P8

Candidates, in rough value order:
1. Reproducibility gate (S10) in CI: two seeded runs, assert identical metrics.
2. `--max-steps` smoke configs so the whole pipeline runs in < 30 s for CI.
3. Coverage report; close gaps on any FR without a test.
4. `ruff` + `mypy` clean.
5. `docker compose` for backend + frontend (CPU wheel only — keep the image small).
6. Sweep `min_freq` ∈ {1,2,3,5} and record the F1/OOV trade-off. The 5.77% OOV is accepted, not proven optimal.
7. Bidirectional LSTM for sentiment — a *strictly optional* experiment, recorded in `PARITY.md`, not a default.

**Not in this phase, ever:** transformers, pretrained embeddings, GPU-only paths (`PRD.md` §3.2).

---

## Phase tracking

| Phase | Status | Closes | Gate |
|-------|--------|--------|------|
| P0 Scaffold | ✅ **done** 2026-08-29 | — | 52 tests green · `--help` lists 4 cmds |
| P1 Data layer | ✅ **done** 2026-08-29 | D3, D6, D7, D9 | 146 tests green · measured values asserted |
| P2 Sentiment | ✅ **done** 2026-08-29 | D4, D5, D8, D11 | macro-F1 **0.8300** vs 0.4459 (corrected twice: selection on test in P8, duplicate rows in v1.1.0) · 213 tests |
| P3 Text-gen | ✅ **done** 2026-08-29 | D6, D9 | ppl 223.54 vs 2,436 · RSS 421 MB — superseded by v1.2.0: held-out test **267.54**, validation 186.27 |
| P4 Sampling | ✅ **done** 2026-08-29 | **D2**, D10 | entropy 1.03 -> 7.76 monotonic · 273 tests |
| P5 CLI | ✅ **done** 2026-08-30 | **D1**, D10 | 4 commands from `C:\Users` · 306 tests |
| P6 FastAPI backend | ✅ **done** 2026-08-30 | — | 6 routes + 422/503 · /predict 2.3 ms · /generate 57 ms · 355 tests |
| P7 Streamlit frontend | ✅ **done** 2026-08-30 | — | S17 · S18 · S19 · slider 260 ms · audit 0 skip · 407 tests |
| P8 Parity | ✅ **done** 2026-08-30 | D11 | `PARITY.md` · 11/11 closed · 412 tests |
| P9 Hardening | ✅ **done** 2026-08-30 | — | calibration · S10 gate · lint blocking · 435 tests |
| v1.1.0 Dedup | ✅ **done** 2026-08-30 | — | 0 leaked test rows · macro-F1 0.8300 vs 0.4459 · 440 tests |
| v1.2.0 Text-gen held-out | ✅ **done** 2026-08-30 | — | test perplexity 267.54 vs 2,436 · baseline unmoved · 444 tests |
| v1.2.2 Verification pass | ✅ **done** 2026-08-30 | — | every published figure re-measured · 16 superseded figures retired · stale-figure gate widened to configs, `src/`, `tests/` · 448 tests |

**`Memory.md`** is created at the end of P0 (its task 10) and appended to at the end of every phase thereafter (`Rules.md` A3).
