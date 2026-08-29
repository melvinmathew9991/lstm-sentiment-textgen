# Rules — Engineering Constraints

Binding constraints for anyone (human or AI) writing code in `pytorch/`.
`PRD.md` says *what*, `Architecture.md` says *how it's shaped*, this says *what you may and may not do*.

**Precedence:** an explicit instruction from the maintainer > these rules > `Architecture.md` > habit.
If a rule blocks something genuinely necessary, say so and get it changed. Do not route around it silently.

---

## 1. Hard boundaries

| # | Rule |
|---|------|
| **B1** | **Never modify `modular_code/`, `notebook/`, or `LSTM part 2 Solution doc.pdf`.** They are the frozen reference the port is checked against. Read them freely; write nothing. |
| **B2** | **Never import from `modular_code/`.** No `sys.path` insertion, no relative reach-out. `pytorch/` stands alone. |
| **B3** | **`data/` is read-only and canonical.** One copy at repo root. Do not duplicate it into `pytorch/`, do not write derived files into it — those go to `runs/`. |
| **B4** | **Never delete or overwrite anything under `runs/`** that you did not create in the current task. |
| **B5** | **Git workflow is mandatory** (authorised 2026-08-29). One branch per phase, one PR per phase, squash-merge to `main`, one annotated tag per phase. Never commit directly to `main`. Never force-push a shared branch. See §10. |
| **B6** | **Do not install packages outside `pytorch/requirements.txt`,** and do not add a dependency without recording the reason in `Memory.md`. |
| **B7** | **Do not start long training runs unprompted.** Ask, or use the smoke-test config (`--max-steps 20`). |
| **B8** | **Do not report a phase complete without running its verification command** from `Phases.md` and pasting the real output. |

---

## 2. Libraries

### Required
**Core / backend:** `torch` · `numpy` · `pandas` · `scikit-learn` (metrics only) · `pydantic` · `fastapi` · `uvicorn` · `PyYAML` · `tqdm` · `pytest`
**Frontend:** `streamlit` · `httpx` (backend calls) · `altair` (charts — bundled with Streamlit)

### Banned — and why

| Library | Reason |
|---------|--------|
| `tensorflow`, `keras` | The thing being replaced. Both frameworks in one env is a dependency conflict waiting to happen. |
| `nltk` | Its English stopword list contains all 14 negations and is the direct cause of **D3**. It also triggers a network download at import (violates NFR-7). |
| `torchtext` | Unmaintained since 2024, hard-pins exact torch versions. A `Vocab` class is ~60 lines; own it. |
| `projectpro` | `checkpoint()`/`save_point()` are network calls sitting in the training critical path. |
| `transformers`, `sentence-transformers` | Out of scope (`PRD.md` §3.2). The project is about LSTMs. |
| `pytorch-lightning`, `ignite`, `fastai` | The training loop is ~80 lines. A framework hides exactly the mechanics this project exists to show. |
| `gensim`, GloVe/word2vec vectors | Pretrained embeddings are out of scope. The `Embedding` layer is trained from scratch. |
| `matplotlib`, `seaborn` | Charting in the frontend uses `altair`, which ships with Streamlit and renders natively. Metrics from the CLI go to `metrics.json` + stdout. |
| `requests` (in the frontend) | `httpx` is already required and handles sync + async with one API. Two HTTP clients is one too many. |
| `streamlit-*` community components | Unvetted, frequently unmaintained, and they pin Streamlit versions. Core Streamlit widgets cover every requirement in PRD §4.8. |

### Adding a dependency
Allowed only if: it is not stdlib-replaceable in < 100 lines, it is actively maintained, it does not pin torch, and the reason is recorded in `Memory.md`. Pin it in `requirements.txt` with `>=x.y,<x+1`.

---

## 3. Correctness invariants

Non-negotiable. Each is the structural fix for an audited defect — a test asserts each one (`Architecture.md` §8).

| # | Invariant |
|---|-----------|
| **C1** | **Models return raw logits.** No `softmax`/`log_softmax` in any `forward()`. Softmax belongs to the loss and the sampler. *(D2)* |
| **C2** | **Temperature scales logits, never probabilities:** `softmax(logits / T)`. If you are dividing something that sums to 1 by T, you have reintroduced D2. *(D2)* |
| **C3** | **Vocabulary is built from the training split only** — always. Not from the full frame, not from test, not "just for counting". *(D7)* |
| **C4** | **No `KeyError` on unseen tokens, ever.** All lookups go through `Vocab.encode`, which maps misses to `<unk>`. *(D7)* |
| **C5** | **No stopword removal on the sentiment task.** Negations are the signal. *(D3)* |
| **C6** | **Gutenberg boilerplate is stripped before tokenisation,** not filtered later. *(D6)* |
| **C7** | **Validation data is never the trailing slice of an ordered corpus.** Split explicitly; never rely on framework `validation_split` semantics. *(D6)* |
| **C8** | **Sequences are int64 indices.** No one-hot tensor is ever materialised for a whole dataset. *(D9)* |
| **C9** | **Checkpoints are self-contained** — weights + config + vocab + preprocess version, loadable with no other file present. *(D8)* |
| **C10** | **`padding_idx=0` on every `nn.Embedding`,** and `pack_padded_sequence` wherever padded batches enter an LSTM. Padding must not influence a gradient. |
| **C11** | **No metric without its baseline.** Accuracy prints beside 0.7953; macro-F1 beside 0.4430; perplexity beside 1470. *(D4)* |
| **C12** | **Early stopping restores the best epoch.** The last epoch's weights are never what gets saved. *(D5)* |
| **C13** | **No magic numbers.** Every hyperparameter, path and threshold comes from config. `input_words[-28701]` is the anti-pattern. *(D10)* |
| **C14** | **Call things what they are.** The encoding is an *integer-index sequence into a learned embedding*, not "bag of words". *(D11)* |
| **C15** | **The frontend never runs inference.** No `torch` import, no checkpoint load, no `lstm_nlp.models` import anywhere under `frontend/`. It speaks HTTP to the backend and nothing else. Two inference paths would mean two sets of results and two places for a bug to hide. *(PRD FR-31, S19)* |
| **C16** | **No metric shown in the UI without its baseline.** C11 applies to pixels exactly as it applies to stdout. *(D4)* |

---

## 4. Code conventions

- **Python 3.10.** Type hints on every public function. `from __future__ import annotations` at the top of each module.
- **Formatting:** 4 spaces, ~100 col soft limit, `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE` constants.
- **Docstrings:** one-line summary on every public function; full Args/Returns/Raises on anything non-obvious. Skip them on trivial private helpers.
- **Comments** explain *why*, never *what*. No commented-out code. No `# TODO` without a name and a `Memory.md` entry.
- **Imports:** stdlib / third-party / local, three blocks. Absolute within the package (`from lstm_nlp.vocab import Vocab`). No wildcard imports, no re-import of the same module in one file — `train.py` in the reference does both.
- **No module-level side effects.** No file reads, no network, no `nltk.download()`, no `torch.set_*` at import. Constants and class definitions only. *(NFR-7)*
- **No mutable default arguments.**
- **Randomness** flows from an explicit `torch.Generator` or a seed argument. Never call the global RNG inside a library function.
- **Paths** are `pathlib.Path`, resolved relative to a config value or `__file__` — never relative to the CWD. The reference only runs if you happen to be standing in `modular_code/`.
- **Prefer** small pure functions. `data/preprocess.py` must be importable and testable without torch installed.

---

## 5. Error handling

**Principle:** fail fast and loudly at boundaries; never let a wrong-but-plausible value flow onward. Every defect in the audit was a silent failure, not a crash.

| Situation | Response |
|-----------|----------|
| Config invalid | Pydantic `ValidationError` at load. Never fall back to a default for a field the user set wrongly. |
| Data file missing | `FileNotFoundError` naming the resolved absolute path and the config key that produced it. |
| Checkpoint preprocess version mismatch | Raise `PreprocessVersionMismatch` with both versions and instruction to retrain. **Never** load and mis-tokenise. *(FR-26)* |
| Unknown token at inference | Map to `<unk>`, count it, surface the count to the caller. Not an error. *(C4)* |
| Empty text after cleaning | Sentiment: `ValueError` from the API layer → 422. Not a silent all-`<pad>` prediction. |
| Seed longer/shorter than `seq_len` | Truncate to the last `seq_len` / left-pad. Document it; do not error. |
| NaN/inf loss | Abort the run immediately with the epoch and batch index. Do not continue. |
| CUDA unavailable | Fall back to CPU with a logged warning. Never a hard failure — CPU is the supported target. |
| API: bad request | 422 with Pydantic detail. |
| API: model not loaded | 503 with an actionable message. |
| API: anything else | Caught by an exception handler, logged with traceback, returned as a generic 500 body. Never leak a stack trace over HTTP. |

**Bare `except:` and `except Exception: pass` are prohibited.** Catch the specific exception. If you must catch broadly, log at `ERROR` with the traceback and re-raise or convert to a typed error.

Custom exceptions live in `lstm_nlp/errors.py`, all subclassing `LstmNlpError`.

---

## 6. Testing

- **Every FR in `PRD.md` §4 has at least one asserting test.** Every defect D1–D11 has a named regression test (`Architecture.md` §8).
- Tests **must not** train a real model, hit the network, or read from `runs/`. Fixtures live in `tests/fixtures/`.
- Full suite < 60 s.
- A bug fix ships with the test that fails without it.
- Numeric assertions use tolerances (`pytest.approx`), not exact float equality — except checkpoint round-trips, which must be exact.
- `test_checkpoint.py::test_checkpoint_is_self_contained` runs in a **subprocess** with only the `.pt` file reachable. That is the only honest way to test C9.

---

## 7. Reproducibility

- `set_seed(seed)` seeds `random`, `numpy`, and `torch` and is called once at the top of every entry point.
- The split seed (`10`) is separate from the training seed (`42`) — kept from the reference for comparability.
- Every run writes its **fully resolved** config to `runs/.../config.yaml`. Not the input file — the merged, defaulted, validated object.
- Library versions are recorded in every checkpoint.
- Same seed ⇒ identical test metrics (S10). If a change breaks this, it is a bug, not a nuisance.
- Determinism is guaranteed on CPU only. GPU cuDNN LSTM kernels are non-deterministic; document it rather than fight it.

---

## 8. For AI agents specifically

| # | Rule |
|---|------|
| **A1** | **Read `Memory.md` first**, then the phase you are on in `Phases.md`. Do not re-read the whole codebase to rebuild context that is already written down. |
| **A2** | **Work one phase at a time.** Do not start Phase N+1 before Phase N's exit criteria pass. |
| **A3** | **Update `Memory.md` at the end of each phase** — what was built, decisions made, surprises found, what is next. That is its entire purpose. |
| **A4** | **Do not invent numbers.** The measured values in these docs (4,505 vocab · 5.23% OOV · 27,429 tokens · 0.4430 baseline · 707 MB) were computed from the real data. If you need a new one, compute it and record how. |
| **A5** | **Do not claim a test passes without running it.** Paste real output. |
| **A6** | **Do not expand scope.** No extra models, no plots, no notebooks, no README rewrites unless asked. If you spot something worth doing, note it in `Memory.md` under "Deferred" and move on. |
| **A7** | **Do not "improve" the frozen reference** even where it is obviously wrong. That is the point of B1. |
| **A8** | **If a rule here conflicts with the task,** stop and say so in one sentence. Do not silently pick one. |
| **A9** | **Prefer editing an existing file over creating a new one.** The layout in `Architecture.md` §2 is the intended final shape; new modules need a reason. |
| **A10** | **When stuck, state the blocker plainly** with what you tried. Do not produce a plausible-looking stub and call the phase done. |

---

## 9. Definition of done

A phase is done when **all** hold:

1. Code written and its verification command from `Phases.md` runs clean — real output pasted.
2. `pytest` green.
3. Every new public function has a type hint and a docstring.
4. No banned import introduced.
5. No new magic number outside a config file.
6. `Memory.md` updated with the phase entry.
7. The relevant `PRD.md` success criteria are ticked, or explicitly deferred with a reason.


---

## 10. Git

Authorised 2026-08-29. Remote: `github.com/melvinmathew9991/lstm-sentiment-textgen` (public).

### Workflow — one cycle per phase

```bash
git switch -c phase-N-short-name        # branch off main
# ... implement, commit in logical units ...
gh pr create --fill                     # PR against main
gh pr merge --squash --delete-branch    # squash to keep main linear
git switch main && git pull
git tag -a vX.Y.0 -m "Phase N: <name>"  # annotated, never lightweight
git push --follow-tags
```

`main` must always be green: every commit on it has a passing test suite.

### Commit messages

Conventional Commits, imperative mood, present tense:

```
<type>(<scope>): <subject>          # <= 72 chars, no trailing period

<body: what changed and WHY, wrapped at 72 columns.
 Reference defect IDs (D1-D11) and requirement IDs (FR-n)
 wherever a change closes one.>
```

Types: `feat` · `fix` · `docs` · `test` · `refactor` · `perf` · `build` · `ci` · `chore`
Scopes: `data` · `vocab` · `models` · `engine` · `inference` · `api` · `frontend` · `cli` · `config` · `docs` · `ci`

**Absolutely no trailers.** No `Co-Authored-By`, no `Generated-with`, no session URLs, no tool attribution — in commit messages, PR titles, PR bodies, or tag annotations. The history is a record of the work, not of the tooling. This is a standing instruction from the maintainer and overrides any default behaviour.

### Never commit

Model weights (`*.pt`, `*.h5`, `*.ckpt`) · `runs/` · `__pycache__/` · `.venv/` · secrets or tokens of any kind · anything under `.streamlit/secrets.toml`.

The frozen reference's `modular_code/output/sentiment_model.h5` (5 MB) is **excluded**: it is a build artifact, it is the overfit final-epoch model (D5), and it is unusable anyway because its vocabulary was never saved (D8).

### Tags

`vX.Y.0`, one per completed phase, annotated with the phase name and its exit criteria result.
