# lstm-nlp — quickstart

Many-to-one LSTMs for sentiment classification and next-word generation, in
PyTorch. CPU only.

Design and rationale live in the root documents — `PRD.md` (requirements),
`Architecture.md` (structure), `Rules.md` (constraints), `Design.md` (the UI),
`PARITY.md` (how this compares to the reference it replaces). This file is
commands.

## Install

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

The CPU index matters: without it pip pulls a CUDA build of torch, several GB
that this project never uses. CI asserts no `+cu` wheel arrives.

## Train

```bash
lstm-nlp train --config configs/sentiment.yaml     # ~2 min CPU
lstm-nlp train --config configs/textgen.yaml       # ~1 min CPU
```

Each run writes `runs/<task>/<timestamp>/` containing `best.pt`, `history.json`,
`metrics.json` and the resolved `config.yaml`. The checkpoint is self-contained —
weights, vocabulary and the preprocessing contract travel together, so loading it
needs no other file.

Add `--max-steps N` for a smoke run. Those are marked as such in the checkpoint
and are **skipped** when a checkpoint is resolved automatically, so a smoke run
cannot become the model your API serves.

## Use

```bash
lstm-nlp eval     --ckpt runs/sentiment/<run>/best.pt --split test
lstm-nlp predict  --ckpt runs/sentiment/<run>/best.pt "the flight was not great"
lstm-nlp generate --ckpt runs/textgen/<run>/best.pt --seed "alice was beginning to" \
                  --n-words 40 --temperature 0.7 --rng-seed 42
```

`--split val` reports the block early stopping selected on; `--split test` reports
the block nothing selected on. They differ, and the gap is the point.

Every command works from any working directory. Exit codes: `0` success,
`1` runtime error, `2` usage error. No path prints a traceback.

## Serve

```bash
uvicorn lstm_nlp.api.app:app --port 8000     # backend, /docs for the contract
streamlit run frontend/app.py                # frontend, :8501
```

Two commands, no build step. The backend loads both checkpoints once at startup;
the frontend is a pure HTTP client and never imports torch.

Point them elsewhere with environment variables:

| Variable | Meaning |
|---|---|
| `LSTM_NLP_RUNS_DIR` | where to look for trained runs |
| `LSTM_NLP_SENTIMENT_CKPT` / `LSTM_NLP_TEXTGEN_CKPT` | an explicit checkpoint, overriding resolution |
| `LSTM_NLP_LOG_LEVEL` | backend log threshold |
| `LSTM_API_URL` | backend the frontend talks to |

## Test

```bash
pytest                 # fast path, ~35 s
pytest -m ""           # everything, including tests that train and start servers
python scripts/audit.py    # 21 project-wide checks; non-zero exit on any failure
```

Tests that need a trained checkpoint skip cleanly when there is none, so a fresh
clone passes before you have trained anything.
