# Parity report — PyTorch rebuild vs the frozen TensorFlow reference

**Measured 2026-08-30** · commit `v0.8.1` · CPU only · all figures reproducible
by the command printed beneath them.

This document compares the rebuild in `pytorch/` against the frozen reference in
`modular_code/` and `notebook/`. It closes **D11** and completes the ledger of
eleven defects in `PRD.md` §1.1.

---

## 0. The comparison you cannot run

The reference's numbers are quoted from its own saved output, never re-measured,
and that is not a choice. `output/sentiment_model.h5` holds 5 MB of weights and
nothing else: `word_to_int`, `int_to_word` and `sequence_length` were never
persisted, so there is no way to reconstruct the input space those weights
expect. **The artifact cannot be loaded, so the claim cannot be checked** — which
is D8, and which is the single most consequential defect in the reference, above
even D2. A number nobody can reproduce is a number nobody can audit.

Every figure in the "reference" column below is therefore what the reference
*recorded about itself*.

---

## 1. Sentiment classification

| | Reference (recorded) | This rebuild (measured) | Baseline |
|---|---|---|---|
| Accuracy | 0.909 *(validation)* | **0.8926** *(held-out test)* | 0.7953 majority class |
| Macro-F1 | not reported | **0.8391** | 0.4430 |
| ROC-AUC | not reported | **0.9303** | 0.5000 |
| Positive-class F1 | not reported | **0.7462** | 0.0000 |
| Model selection | none — final epoch saved | early stopping on a held-out validation block | — |
| Reproducible? | no (D8) | yes | — |

```bash
lstm-nlp eval --ckpt pytorch/runs/sentiment/<run>/best.pt --split test
```

### The reference's 0.909 is higher than our 0.8926. It is also not a comparison.

Read naively, the reference wins on accuracy. Four reasons that reading is wrong,
in increasing order of importance:

1. **Accuracy is the wrong metric here** (D4). The corpus is 79.53% negative, so
   a classifier that answers *negative* every time scores **0.7953**. The
   reference's 0.909 is +0.114 over that floor; ours is +0.0973. Neither number
   means much without the floor beside it, and the reference never printed one.
2. **It was measured on negation-stripped text** (D3). Its pipeline removed all
   14 English negations before a *sentiment* model saw the input, so `"not
   worried"` and `"worried"` were the same string. That is a different, easier
   task than the one we solved.
3. **It was validation accuracy at the final epoch**, with no early stopping and
   no checkpointing (D5) — its own log shows `val_loss` rising from 0.215 at
   epoch 1 to 0.526 at epoch 7 while training loss fell to 0.015. The saved model
   is the most overfit one it produced.
4. **Ours is held out; theirs is not.** Our 0.8391 / 0.8926 come from a block no
   part of model selection ever saw. See §5 — we got this wrong ourselves first,
   and the correction cost us 0.0094 macro-F1.

`PRD.md` §6.3 states this outright: matching 0.909 was never a target, because
reproducing an unbaselined number from a differently-preprocessed task is not
evidence of anything.

### Confusion matrix, held-out test (n = 3,463)

```
                predicted
              negative  positive
true negative     2544       210      precision 0.9401  recall 0.9237  F1 0.9319
     positive      162       547      precision 0.7226  recall 0.7715  F1 0.7462
```

The minority class is the one that matters and the one accuracy hides: 547 of
709 positives found. A majority-class predictor finds zero of them and still
scores 0.7953.

---

## 2. Text generation

| | Reference (recorded) | This rebuild (measured) | Baseline |
|---|---|---|---|
| Validation perplexity | not reported | **223.54** | 2,436 (uniform over V) |
| Cross-entropy | not reported | **5.4096** nats | 7.7981 (= ln 2,436) |
| Top-1 next-word accuracy | not reported | **0.1482** | 0.0004 (= 1/V) |
| Input tensor | one-hot, 931 MB | int64 indices, **0.22 MB** | — |
| Temperature behaviour | uniform at every T (D2) | entropy 1.03 → 7.76 across T | — |

```bash
lstm-nlp eval --ckpt pytorch/runs/textgen/<run>/best.pt
```

Perplexity 223.54 against a uniform baseline of 2,436 is a **10.90× reduction**.
The reference reported no perplexity at all, so there is nothing to compare
against except the floor.

---

## 3. D2 — the temperature defect, side by side

The reference divided **probabilities** by temperature and passed the result to
`tf.random.categorical`, which expects **logits** and applies its own softmax. At
T=10 the inputs land in ~[0, 0.007]; after softmax they are indistinguishable
from uniform. It drew words at random from the entire vocabulary at *every*
setting.

**Reference**, notebook cell 130 — its own saved output:

```
the alone court, to. pop sits bursting particularly
```

**This rebuild**, same seed, three temperatures:

```bash
lstm-nlp generate --ckpt <textgen>/best.pt --seed "alice was beginning to" \
                  --n-words 24 --temperature 0.3 --rng-seed 42
```

```
T=0.3   alice was beginning to her in the other side of the other side of the
        other and the mock turtle s very soon to the queen s head
        -> near-greedy: loops on "the other side of the"

T=0.7   alice was beginning to the rabbit and she was nothing of them so he had
        quite to have their in the same thing a little thing a bit
        -> locally coherent, varied

T=1.2   alice was beginning to the rabbit but she was nothing of them so he had
        quite to have their in this time on a little from a tree
        -> more inventive, drifting
```

Measured entropy of the sampling distribution, same context:

| T | entropy (nats) | % of uniform | reference |
|---|---|---|---|
| 0.2 | 1.0287 | 13.2% | 7.7981 — 100% |
| 0.7 | 3.4333 | 44.0% | 7.7981 — 100% |
| 1.0 | 5.3862 | 69.1% | 7.7981 — 100% |
| 2.0 | 7.4154 | 95.1% | 7.7981 — 100% |
| 5.0 | 7.7563 | 99.5% | 7.7981 — 100% |

The reference's column is flat because its curve *was* flat. That is the whole of
D2, measured rather than argued.

```bash
curl -s -X POST localhost:8000/distribution -H 'Content-Type: application/json' \
     -d '{"seed":"alice was beginning to","temperature":0.2}'
```

---

## 4. Resources

| | Reference | This rebuild |
|---|---|---|
| Text-gen input storage | **931 MB** one-hot `(30664, 10, 3036)` — `modular_code`; the notebook allocated **1,671 MB** at `(29584, 10, 5649)` | **0.22 MB** — one 1-D int64 tensor, sliced lazily |
| Peak training RSS | not recorded | **421 MB** (293 MB of which is importing torch) |
| Sentiment wall-clock | 50 epochs, no early stop | **120.8 s**, 11 epochs, stopped at 6 |
| Text-gen wall-clock | ~100 epochs before the D1 crash | **57.2 s**, 7 epochs, stopped at 2 |

The storage difference is not an optimisation, it is the removal of a mistake:
one-hot encoding an input that is immediately fed to an `Embedding` layer
allocates a matrix whose only purpose is to be multiplied back into the index it
started as. Storage is O(tokens), not O(windows × window length) — 0.22 MB
rather than the 2.19 MB even a dense index matrix would need.

---

## 5. Defects we introduced ourselves

A parity report that only audits the other side is not an audit. Two defects of
our own were found in a review immediately before this document was written,
both fixed in `v0.8.1`:

| | Defect | Cost, measured | Fixed |
|---|---|---|---|
| — | The sentiment model was selected on the **test** split: early stopping maximised macro-F1 on the rows the headline was reported for | corrected: macro-F1 0.8485 → **0.8391** held out (+0.0094 optimism) | `v0.8.1` |
| — | A `--max-steps` smoke run silently became the model the API served | macro-F1 **0.6997** served instead of 0.8391 | `v0.8.1` |

Every sentiment figure in this document is the corrected one. The first defect
is the same species as D5 and D7 in the reference — the evaluation set leaking
into the fitting procedure — which is worth saying plainly rather than filing
under "lessons learned".

---

## 6. Known limitations

Recorded because a report that lists only what went well is marketing.

- **Probabilities were over-confident; now partly corrected.** Expected
  Calibration Error was **0.0609** on test. Phase 9 fits a temperature on the
  validation block (T = 1.5922), which brings test ECE to **0.0324** -- a 47%
  reduction on data the fit never saw. Because temperature scaling is
  monotonic, **no decision changed and no metric in this document moved**:
  macro-F1 and accuracy are bit-identical either way, and 0 of 3,463 test
  predictions flipped.

  It is improved, not solved. A single global scalar cannot fix bin-specific
  miscalibration, and the middle of the range is still over-confident by
  0.11-0.21:

        bin        n   mean p  observed     gap
        0.4-0.5   82    0.450     0.305   +0.145
        0.5-0.6   93    0.551     0.344   +0.207
        0.7-0.8  103    0.750     0.553   +0.196
        0.9-1.0  344    0.949     0.913   +0.036

  The API returns `calibrated: true|false` and the UI says which, because a
  score presented as a probability is the failure this project exists to
  remove.
- **2.48% of test rows duplicate a training row** — 86 of 3,463, mostly stubs
  like `"<user> thanks"` (×18). Five distinct cleaned texts carry *both* labels.
  This is corpus noise, not a splitting defect, but it inflates all scores
  slightly. A stricter protocol would deduplicate before splitting.
- **Text generation has no held-out test block.** Its perplexity is measured on
  the same split early stopping used. `PRD.md` S4 asks for "validation
  perplexity", so the label is honest, but the selection effect is the one fixed
  for sentiment in §5. Deliberately left: a third block rebuilds the text-gen
  vocabulary and moves the 2,436 / 7.7981-nat baseline this entire document
  quotes against.
- **Negation does not always cross the decision boundary.** `"the flight was not
  great"` moves 0.984 → 0.900 without flipping, because the corpus is 79.5%
  complaints and *"great"* is a very strong positive marker. Other pairs cross
  outright (`"service was not good"` 0.806 → 0.145). D3 is about the *input*
  surviving preprocessing, and it does; the model's sensitivity is a separate,
  weaker claim and is stated as one.
- **The audit's defect-coverage check is loose** — it substring-matches over the
  concatenated test sources, so D10 is "covered" by the word `config`. The real
  D10 test exists; the check would not notice its deletion.

---

## 7. The defect ledger — all eleven closed

| ID | Defect | Closed by | Regression test |
|---|---|---|---|
| **D1** | `engine.py:48` calls `generate_paragraph` with 4 of 7 required arguments; `TypeError` only after ~100 epochs | Static arity check across the whole package | `test_call_signatures.py::test_every_internal_call_matches_its_signature` |
| **D2** | Temperature divides probabilities, not logits → uniform sampling at every T | `softmax(logits / T)`; models return logits, so the error is unrepresentable | `test_sampler.py` entropy monotonicity · `test_api.py::test_distribution_entropy_rises_with_temperature` |
| **D3** | Stopword removal strips all 14 negations before a sentiment model | No stopword removal; negations kept | `test_preprocess.py::test_negations_survive_cleaning` · `test_trained_sentiment.py::test_negation_changes_the_prediction` |
| **D4** | Accuracy is the only metric on 4:1 imbalanced data | Every metric printed beside its baseline; macro-F1 is the headline | `test_metrics.py::test_all_negative_predictor_exactly_equals_the_baseline` |
| **D5** | 50 epochs, no early stopping, final (worst) epoch saved | Early stopping with best-weight restoration | `test_callbacks.py::test_best_not_last_checkpoint_restored` |
| **D6** | `validation_split=0.1` makes validation entirely Gutenberg licence text | Boilerplate stripped, then a block split over real prose | `test_datasets.py::test_no_gutenberg_vocabulary_in_textgen` |
| **D7** | Vocabulary built on 100% of data before the split; `KeyError` on unseen words | Split first, build vocabulary from train only; unknowns map to `<unk>` | `test_datasets.py::test_vocab_built_from_train_only` · `test_predictor.py::test_unknown_words_do_not_raise` |
| **D8** | `sentiment_model.h5` unusable — vocabulary never persisted | Self-contained `.pt`: weights, vocabulary, preprocessing contract, metrics | `test_checkpoint.py::test_checkpoint_carries_the_vocabulary` |
| **D9** | One-hot text-gen input tensor, 931 MB | Lazy int64 index slices, 0.22 MB | `test_datasets.py::test_windows_are_int_indices_not_onehot` · `::test_storage_is_o_tokens_not_o_windows` |
| **D10** | `input_words[-28701]` — a magic index tuned to a vanished sample count | Every default from config; no sampling literal in `cli.py` | `test_cli_integration.py::test_generate_falls_back_to_config_defaults` |
| **D11** | Docs call the encoding "bag-of-words"; it is an ordered integer-index sequence into a learned `Embedding` | Terminology corrected throughout | `scripts/audit.py` — *Terminology (D11)*, which fails the build on any uncorrected use |

```bash
cd pytorch && pytest -m "" && python scripts/audit.py
```

---

## 8. Reproducing everything here

```bash
cd pytorch
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .

lstm-nlp train --config configs/sentiment.yaml      # ~2 min CPU
lstm-nlp train --config configs/textgen.yaml        # ~1 min CPU

lstm-nlp eval --ckpt runs/sentiment/<run>/best.pt --split test
lstm-nlp eval --ckpt runs/textgen/<run>/best.pt

pytest -m ""                                        # 412 tests
python scripts/audit.py                             # 21 checks
```

Training is seeded (`seed: 42`), so the figures above reproduce on a CPU machine.
Generation reproduces exactly when `--rng-seed` is given, and only then.
