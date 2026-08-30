# Changelog

Notable changes per phase. Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versions are phase milestones, one annotated git tag each.

Defect IDs (D1-D11) refer to `PRD.md` section 1.1.

## [Unreleased]

Nothing planned.

## [1.2.3] - 2026-08-30 - CI finally checks a headline number

The item deferred at Phase 5 and restated at Phase 9, made more valuable by
1.2.2's finding that a clean local tree skips the same tests CI does.

### Added

- **A smoke train on every matrix leg.** `runs/` is gitignored, so CI had no
  checkpoint and every test that loads a trained model skipped at runtime --
  53 of them. Phase 5 recorded the consequence plainly: *"no headline number in
  this repository has ever been verified by CI."* That stayed true for five
  phases and three releases, because a skip is green.

  Measured before choosing the fix rather than after:

        no checkpoints    391 passed   53 skipped
        smoke  (~20 s)    442 passed    4 failed   2 skipped
        real   (~3 min)   448 passed    0 failed   0 skipped

  A twenty-second `--max-steps 3` run recovers 49 of the 53, on both 3.10 and
  3.12 -- including the predictor, sampler and subprocess CLI paths, which are
  the version-sensitive ones and had never run in CI at all.

- **A `trained-model gates` job** that trains both models properly on 3.12 and
  runs the whole suite. What it asserts is the gates -- macro-F1 >= 0.75,
  perplexity <= 400, entropy monotonic across T, negation direction over five
  pairs -- never the published figures. Determinism is guaranteed on CPU
  (`Rules.md` 7) and every reproduction so far has been on Windows/Python 3.10;
  bit-identity across operating systems is a stronger claim than this project
  has measured, so asserting 0.8300 in a Linux job would be inventing a
  guarantee (`Rules.md` A4). The job prints `lstm-nlp eval` for both models so
  the numbers are in the log to be read.

- **A skip budget on both jobs**, checked in the shell: `-le 2` on the matrix
  legs and `-eq 0` on the trained job. This is the part that outlives the fix.
  Adding a training step fixes today; budgeting the skips is what stops the same
  silence returning the next time a fixture goes missing, because the failure
  mode here was never a red build -- it was a green one that tested less than it
  looked like it did.

  The two skips the matrix legs tolerate are the Phase 8 guard working: the API
  refuses to resolve a run stamped with `max_steps`, so its two latency tests
  skip. Observed, not assumed.

- **A `fulltrain` pytest marker** for the four tests a smoke model cannot
  satisfy -- the macro-F1 gate, negation, minority-class recall, early stopping.
  They fail on a smoke model for the right reason: a three-step model does not
  clear a quality gate, and a run capped at three steps per epoch never
  early-stops. Marked rather than deselected by name in the workflow, because a
  test excluded in YAML is invisible from the test file.

### Not verified

The workflow itself has never run. Both shell blocks were rehearsed locally
against the states they will meet -- 442 passed / 2 skipped under smoke
checkpoints, 448 passed / 0 skipped under real ones, and the YAML parses into
four jobs -- but a workflow that has only been rehearsed is not a workflow that
has passed.

## [1.2.2] - 2026-08-30 - The figures a correction did not reach

An independent verification pass: retrain both models from a clean tree,
re-measure every published number, and check the reference's defects against
the frozen source rather than against the catalogue.

**Everything reproduced.** Accuracy 0.8974, macro-F1 0.8300, ROC-AUC 0.9126 and
all four confusion cells; test perplexity 267.54 and validation 186.27; the
entropy curve 1.0287 / 3.4333 / 5.3862 / 7.4154 / 7.7563 against a 7.7981 floor,
measured over HTTP in a different process, to four decimals; three seeded
passages byte-identical. What did not reproduce was the documentation.

### Fixed

- **Sixteen superseded figures across eleven files.** v1.1.0 deduplicated the
  corpus, registered the *direct* metrics it moved with the audit's stale-figure
  gate, and never enumerated the figures **derived** from the same change. Those
  went on being published for two releases:

  - `PARITY.md` section 6 stated the calibration result as T = 1.5922, test ECE
    0.0609 -> 0.0324 ("a 47% reduction"), 0 of 3,463, over a four-bin residual
    table. Measured: **T = 2.6715, 0.0803 -> 0.0223 (72%), 0 of 3,382**, with the
    worst residual +0.187 at the 0.4-0.5 bin. `Memory.md` recorded the refit
    correctly at v1.1.0; the only document that publishes it did not.
  - `README.md` headlined text-gen perplexity **223.54 (10.9x)** -- the v0.4.0
    validation figure under the old 90/10 split, superseded by v1.2.0. The
    held-out figure is **267.54 (9.11x)**.
  - `Architecture.md` section 3.1 totalled 258,880 + 66,560 + 130 as **355,010**.
    The sum is **325,570**, which is what the checkpoint holds.
  - `PARITY.md` section 1 read "higher than our 0.8926" two lines under its own
    table saying 0.8974; section 4 reported the pre-deduplication run's
    wall-clock and epochs; section 8 counted 412 tests and 21 checks.
  - `PARITY.md` section 6 still listed the audit's loose defect-coverage check as
    an open limitation, fixed in Phase 9.
  - `configs/sentiment.yaml`, `Phases.md`, `PRD.md`, three module docstrings and
    four test docstrings carried 4,505 / 9,566 / 5.23% / 3.38% / 3.884 / 0.7953 /
    0.4430 as current fact.
  - `Memory.md`'s "Measured data facts" table -- the one thing a new session is
    told to trust without recomputing -- carried the entire pre-deduplication
    sentiment block.

  Cost: nothing downstream. Every conclusion those figures supported survived
  re-measurement, which is the point rather than the relief -- they were wrong in
  the documents whose whole job is that their numbers can be checked.

- **The audit had two sources of truth for the same numbers, and they disagreed.**
  `check_measured_values_documented` *required* 0.7953, 0.4430, 4,505 and 3.884
  to appear in the documents while `check_stale_figures` was never told they were
  superseded, so the gate was actively holding the retired baselines in place.
  `MEASURED` -- dead code, never read, and itself seven values out of date -- is
  now the single list both checks derive from.

- **The stale-figure scan reached only documents.** It now covers
  `configs/*.yaml`, `src/**`, `tests/**` and `scripts/*.py`, each of which
  carried a superseded figure. `Memory.md` and `CHANGELOG.md` are scanned over
  their live regions only -- the header tables and `[Unreleased]` -- which is a
  sharpening rather than an exemption: the whole of both was nominally in scope
  before and the stale header table passed anyway.

- **B1 did not cover the working tree, or the PDF it names.**
  `LSTM part 2 Solution doc.pdf` was deleted in the working tree and the audit
  reported it only as a line inside the generic dirty-tree WARN. The check now
  reads `git status` across all three frozen paths, was verified to FAIL on that
  deletion, and the file is restored.

- **`history.json` contradicted itself.** `best_epoch` was persisted zero-based
  beside one-based `epochs[].epoch`: a run that restored epoch 9 wrote
  `"best_epoch": 8`. Nothing computed with it, so nothing broke, and a reader of
  the artifact could not tell which convention it was in. Persisted one-based
  now, with two tests asserting `best_epoch` names a record that exists.

- **The D9 evidence line under-reported the comparison it exists to make.**
  `textgen_task.py` summed storage and one-hot bytes over train + val only --
  correct for the 90/10 split it was written for, wrong from v1.2.0 -- printing
  `tokens 27429 (train 24687 / val 1371)`, a total that does not equal the sum of
  its parts, and 0.21 MB / 634 MB against a true 0.22 MB / 667 MB.

- **`test_parameter_count_matches_the_specification` asserted the document, not
  the artifact.** It built the model at `vocab_size=4505` and checked 355,010, so
  it would have passed however far the two drifted -- and they had. It builds at
  V=4,045 now, beside a test that the total equals the sum of its parts at an
  arbitrary vocabulary, and a `realdata` test that pins the shipped count.

- **Synthetic fixtures carried real headline numbers.** The tiny checkpoints in
  `conftest.py` and `test_checkpoint.py` reported 0.8972 / 0.8485 / 0.7953 /
  0.4430 / 223.54 for models with random weights. Obviously synthetic now.

- **Every documented `curl` used `localhost`,** which costs 2,066 ms per new
  connection on the dev machine against 15.1 ms for `127.0.0.1`, because uvicorn
  binds IPv4 and the `::1` attempt times out first. A reader reproducing NFR-5's
  "< 100 ms" measured ~2,000 ms. Now a rule in `Rules.md` section 4.

- `build_loaders` in `textgen_task.py` was annotated `-> tuple[DataLoader,
  DataLoader]` while returning three; nothing caught it because `mypy` is still
  not run.

- The CI comment recorded 31 files outside `ruff format`; it is 35.

### Documented, not fixed

- **No headline number is verified anywhere by default.** On a clean tree
  `pytest -m ""` gives **391 passed, 53 skipped** -- every trained-model test,
  for want of a checkpoint. CI has been in that position since Phase 5 and says
  so; a fresh clone now is too, while 1.2.1 above lists `runs/` under
  "Deliberately kept". `README.md` states it with the measured counts. The fix
  remains the deferred CI smoke train.

### Verified

    pytest -m ""                     448 passed in 90.76s     (was 444)
    pytest                           432 passed, 16 deselected, 36.9s
    ruff check .                     All checks passed!
    python scripts/audit.py --fast   20 pass  2 warn  0 fail  1 skip

## [1.2.1] - 2026-08-30 - Repository tidy

### Changed
- **The seven steering documents moved to `docs/`.** `README.md` and
  `CHANGELOG.md` stay at the root, where GitHub and Keep-a-Changelog expect
  them. Git recorded all seven as renames, so `git log --follow` still works.
  `scripts/audit.py` holds the only paths that had to change; nothing in
  `src/`, `tests/` or `frontend/` reads a document, so the rest of the tree was
  untouched.
- Removed `pytorch/.gitignore`: every rule in it was already in the root
  `.gitignore`, so it was a second source of truth for one set of rules.

### Removed
- Build cruft: ten `__pycache__/` directories, `.pytest_cache/`, `.ruff_cache/`
  and `lstm_nlp.egg-info/`. All gitignored, all regenerated on the next run.

### Deliberately kept
- `modular_code/`, `notebook/` and the solution PDF are the frozen reference
  `Rules.md` B1 protects and `PARITY.md` compares against. `modular_code/output/`
  holds `sentiment_model.h5` -- the 5 MB artifact that cannot be loaded, which is
  the evidence for D8 and the subject of `PARITY.md` section 0. Deleting the
  thing a report is about would be an odd way to tidy.
- `pytorch/runs/` (12 MB, gitignored) holds the trained checkpoints the API,
  the frontend and roughly fifty tests use. Regenerable in about three minutes,
  but not junk.

## [1.2.0] - 2026-08-30 - A held-out block for text generation

The last open item from `PARITY.md` section 6, and the one I had priced wrong.

### Fixed
- **Text generation had no held-out test block.** Perplexity was measured on the
  same split early stopping selected against -- the defect fixed for sentiment
  in 0.8.1, still live on the other task. `PRD.md` S4 asked for "validation
  perplexity", so the label was honest, but an honest label on a selected
  number is still a selected number.

        validation  186.27     the block early stopping saw
        test        267.54     held out, scored once
                    +44%       the selection effect, now visible

  267.54 still clears the S4 gate of <= 400 against the same 2,436 floor.

### The reason it turned out cheap

I had recorded this as expensive because a third block would rebuild the
vocabulary and move the 2,436 / 7.7981-nat uniform baseline that the entire D2
demonstration is quoted against. That assumed the block had to come out of
**train**. It does not.

        scheme        train    val   test   vocab    ln V
        90/10 (old)  24,687  2,742      0   2,436  7.7981
        80/10/10     21,945  2,742  2,742   2,288  7.7354   <- moves it
        90/5/5       24,687  1,371  1,371   2,436  7.7981   <- identical

At 90/5/5 the test block is carved out of what was *already* held out, so the
training block is byte-identical to the old one and the vocabulary, `ln V` and
every entropy figure in `PARITY.md` section 3 are untouched. Measuring the three
schemes before writing any code turned a change I had described as invalidating
the project's centrepiece into one that touches a single number.

### Changed
- `split_tokens` returns three contiguous blocks; `test_fraction` defaults to
  0.05 and `val_fraction` drops from 0.10 to 0.05, so the held-out region is the
  same size it always was, now divided in two.
- `lstm-nlp eval --split val|test` selects a real block for text generation too.
- Windows total 27,399 (three blocks each lose `seq_len`), from 27,409.
- 4 tests (**444 total**), including the control that the 90/5/5 training block
  is identical to the 90/10 one -- the property the whole design rests on.

## [1.1.0] - 2026-08-30 - Deduplicate before splitting

Closes the second of the two limitations `PARITY.md` section 6 published.

### Fixed
- **2.48% of test rows shared their cleaned text with a training row.** 86 of
  3,463, mostly stubs like `"<user> thanks"` (x18) -- so the model was partly
  scored on inputs it had memorised. Rows are now deduplicated on **cleaned**
  text before the split, because cleaning is what makes two differently-typed
  tweets the same input, and a duplicate that never reaches the splitter cannot
  straddle the boundary. Afterwards **0** test rows share a training row.
- Five cleaned texts carried **contradictory labels**. They are dropped rather
  than resolved: keeping either label would be inventing an annotation, and
  keeping both would put one input in two classes.

### It cost what it should have

    macro-F1   previously 0.8391 -> 0.8300   the duplicates inflated this
    ROC-AUC    previously 0.9303 -> 0.9126
    accuracy   previously 0.8926 -> 0.8974   against a baseline that also rose

270 rows removed (2.34%). The duplicates skewed positive, so **the
majority-class baseline moved too**, 0.7953 -> 0.8048, and the class-weight
ratio 3.884 -> 4.130. Accuracy rising while macro-F1 falls is exactly what a
removed leak looks like on imbalanced data: the easy, repeated rows are gone.

### Changed
- `FR-6` amended. The split seed was pinned "for comparability" with the
  reference; deduplication breaks row-level comparability deliberately, and
  `PARITY.md` section 0 already establishes that the reference's numbers can
  never be re-measured -- so the comparability the pin protected does not exist.
- Corpus figures follow: 11,271 rows (6,705 / 1,184 / 3,382), vocabulary
  **4,045**, test OOV **5.77%**, 325,570 parameters.
- Calibration refitted: **T = 2.6715**, test ECE **0.0803 -> 0.0223**, still
  with 0 of 3,382 decisions changed.
- 5 tests (**440 total**), including the control that deduplication is what
  removes the overlap rather than the fixture happening to have none.

## [1.0.0] - 2026-08-30 - Phase 9: Hardening

### Added
- **Temperature scaling** (`engine/calibration.py`), fitted on the validation
  block at training time and stored in the checkpoint. Measured on the shipped
  model: **T = 1.5922**, validation ECE 0.0558 -> 0.0313, and on the test block
  the fit never saw, **0.0609 -> 0.0324**.

  The property that made this safe to ship is asserted, not argued: scaling
  logits by a positive scalar is monotonic, so `argmax` cannot move. Verified
  on the real model -- macro-F1 and accuracy bit-identical, **0 of 3,463
  decisions changed** -- which is why no figure in `PARITY.md` needed
  re-measuring.
- `PredictResponse.calibrated` and a matching line in the UI. A score presented
  as a probability is precisely the failure this project exists to remove, so
  the caller is told which one it is holding.
- **S10 reproducibility gate** (`tests/test_reproducibility.py`): two full
  seeded training runs must produce identical metrics, identical confusion
  matrices and an identical fitted temperature. Every measured claim in this
  repository rests on this holding, and until now nothing checked it.
- 23 tests (**435 total**).

### Changed
- **`ruff check` is blocking in CI.** The eight outstanding findings are fixed,
  so the next one is new drift rather than accumulated debt. `ruff format`
  stays advisory: it would rewrite 31 hand-laid-out files -- aligned tables,
  deliberate breaks in measured output -- for no correctness gain, and adopting
  a formatter is a decision to take deliberately rather than as a side effect.
- **The audit's defect-coverage check now matches test function names, not file
  text.** It previously searched the concatenated source of every test file, so
  D4, D8 and D10 were "covered" by words appearing in prose; D10's intended
  needle, `magic`, matched nothing at all. Every needle is now a real test name,
  and the check was verified to **fail** when the D10 test is removed -- which
  the old form could not do.

### Still not fixed, still recorded
- Middle-range over-confidence survives temperature scaling (see `PARITY.md`
  section 6). Fixing it needs a per-bin method and more held-out data than 1,212
  rows justifies.
- 2.48% duplicate rows across the split, and no held-out block for text-gen.
  Both would move published figures, and both are described where a reader will
  meet them rather than fixed quietly.

## [0.9.0] - 2026-08-30 - Phase 8: Parity & reporting

Closes **D11**, and with it the ledger: all eleven defects are marked closed
against a named regression test.

### Added
- **`PARITY.md`** -- the rebuild against the frozen TensorFlow reference, every
  figure beside its baseline and every claim reproducible by a pasted command.
- `pytorch/README.md` -- quickstart only. Commands, environment variables, and
  where the design documents are.

### The comparison that cannot be run
The reference's numbers are quoted, never re-measured, and that is not a choice.
`sentiment_model.h5` holds weights and nothing else -- no vocabulary, no
sequence length -- so the input space cannot be reconstructed and the artifact
cannot be loaded. That is D8, and it makes every number the reference reported
permanently unauditable. `PARITY.md` opens with it, because it outranks even D2.

### On the reference's 0.909
Read naively it beats our 0.8926. It is not a comparison: accuracy is the wrong
metric on a 79.53%-negative corpus (D4), the figure was produced on
negation-stripped text (D3), it is a final-epoch validation score with no early
stopping (D5), and it carries no baseline. Ours is held out. `PRD.md` section
6.3 already said matching it was not a target; `PARITY.md` shows the arithmetic.

### Reported against ourselves
`PARITY.md` section 5 lists the two defects **we** introduced and fixed in
0.8.1 -- selection on the test split, and a smoke run reaching the API -- with
their measured cost. Section 6 lists five limitations that remain, including
ECE 0.066 over-confidence and 2.48% duplicate rows across the split. A report
that only audits the other side is not an audit.

### Changed
- `PARITY.md` is registered with `scripts/audit.py`, so it is held to the same
  stale-figure, terminology and baseline checks as every other document.

## [0.8.1] - 2026-08-30 - Held-out evaluation, and a serving guard

An end-to-end review before Phase 8, run because `PARITY.md` is about to
publish these numbers. It found two defects in the project's own work. Both are
of the kind this project exists to eliminate: a claim that was not measured the
way it was described.

### Fixed
- **The sentiment model was selected on the test split.** `train_sentiment`
  passed the test loader to `fit`, so early stopping and best-weight restoration
  maximised macro-F1 on the very rows the headline number was reported for. The
  training history shows it plainly -- epoch 6 was chosen as the maximum over 11
  evaluations, and its 0.8485 / 0.8972 -- now superseded -- are exactly the
  figures that were published.

  There is now a real validation block, carved out of **train** so the test
  block keeps precisely the 3,463 rows it always had. Selection sees `val`;
  `test` is scored once. Measured cost of the old protocol:

        macro-F1   accuracy   ROC-AUC
        0.8485      0.8972     0.9366    selected on test (superseded)
        0.8391      0.8926     0.9303    held out, scored once
        +0.0094     +0.0046    +0.0063   optimism

  The gate is unaffected: 0.8391 -- later superseded by 1.1.0 -- still clears
  the >= 0.75 target against its baseline. What changes is that the number now means what it says.
  Reported honestly, the model also trains on 15% fewer rows, so part of the
  drop is lost data rather than lost optimism -- the true optimism is a little
  smaller than +0.0094.

- **A `--max-steps` smoke run silently became the served model.** Checkpoint
  resolution picks the newest run, and the documented smoke command
  (`Rules.md` B7) writes a run directory like any other. Running it replaced the
  API's model with a half-trained one -- macro-F1 **0.6997** against the
  0.8485 of the day (itself superseded) --
  with nothing in the logs, the checkpoint, or the UI saying so. Runs now record
  `max_steps`, and resolution skips those that have it. An explicit
  `LSTM_NLP_*_CKPT` still points wherever it is told: naming a file is a
  decision, and the guard only stops an accident.

- `lstm-nlp eval --split val` evaluated the test split. It now evaluates
  validation, which is finally a real block.

### Changed
- Vocabulary is built on the 6,866-row training block: **4,083** tokens, test
  OOV **5.68%** (4,505 / 5.23% when built on all 8,078 train rows). The FR-6
  split itself is unchanged, so Phase 1's figures still describe it correctly --
  both numbers are true of different things and neither is deleted.
- The S8 negation test now aggregates over five pairs instead of gating on one.
  The old form asserted `gap > 0.15` for "the flight was (not) great" alone, and
  that pair moves only 0.984 -> 0.900 on the new model while "service was (not)
  good" moves 0.806 -> 0.145 and crosses. Judged on one pair it looked like a
  regression; judged over five, negation sensitivity had improved. A per-pair
  threshold was measuring the run, not the model. This is a strengthening, and
  it is recorded rather than quietly adjusted (`Rules.md` section 11.1).
- The sentiment page's third preset is now "service was not good", so the page
  shows a pair that crosses the boundary next to one that only moves. Showing
  only the flip would be a demo.
- **The `LICENSE` file has been removed** at the maintainer's request. The code
  now carries no licence and no usage rights are granted. The dataset terms in
  `README.md` are unaffected -- they are imposed by Figure Eight/Kaggle and
  Project Gutenberg, not by this repository.

### Known, recorded, not fixed
- **Probabilities are over-confident.** ECE 0.066 on test; inputs scored ~0.75
  are positive about 46% of the time. This is the expected consequence of
  `class_weighting: balanced` (3.884:1), which is the right trade for macro-F1
  but decouples the outputs from the data's prior. The UI presents them as
  probabilities.
- **2.48% of test rows duplicate a training row** (86 of 3,463), mostly stubs
  like `"<user> thanks"`; five distinct cleaned texts carry both labels. Corpus
  noise rather than a split defect, but it inflates scores slightly.
- **Text-gen has no held-out test block.** Its perplexity is measured on the
  split early stopping used. S4 asks for "validation perplexity", so the label
  is honest, but the selection effect is the same one fixed above. Left alone
  deliberately: adding a third block would rebuild the text-gen vocabulary and
  move the uniform baseline (2,436 / 7.7981 nats) that the entire D2
  demonstration is quoted against.
- **The audit's defect-coverage check is loose.** It substring-matches over the
  concatenated text of every test file, so D10 is "covered" by the word
  `config` appearing somewhere; its intended needle `magic` matches nothing. A
  real D10 test exists, but the check would not notice if it were deleted.

## [0.8.0] - 2026-08-30 - Phase 7: Streamlit frontend

Two pages over the HTTP contract, holding no model state. The generation page
is the point of the phase: the temperature slider drives the generated text and
a chart of the next-word distribution at the same time, with the uniform
baseline drawn on -- so the relationship the reference merely claimed is
watched instead.

### Added
- `frontend/settings.py`: backend URL and timeouts from the environment (FR-36).
- `frontend/api_client.py`: the only module that speaks HTTP. Transport
  failures become three typed errors, because the UI must answer them
  differently -- a dead backend is a full-page banner, a missing model is a
  page-level notice, a rejected input is a message beside the control.
- `frontend/theme.py`, `frontend/components.py`, `frontend/charts.py`,
  `frontend/app.py`, `frontend/pages/{1_sentiment,2_generation}.py`,
  `frontend/.streamlit/config.toml`.
- `POST /distribution` is what the chart renders. The frontend asks the backend
  for the distribution rather than deriving it: a chart computed *beside* the
  sampler instead of *from* it is free to disagree with it, which is the exact
  shape of D2.
- `tests/test_frontend.py`: 44 tests. The C15 purity scan (S19) parses every
  file under `frontend/` rather than importing it, so it also covers branches
  the suite never executes. Client tests inject `httpx.MockTransport`; six live
  tests run the real Streamlit pages against a real `uvicorn` process.
- `tiny_runs` fixture moved into `conftest.py` and shared with the API suite --
  two definitions of "a valid checkpoint tree" would eventually disagree.
- `tests/test_dependencies.py`: every third-party import under `src/`,
  `frontend/` and `tests/` must be declared in `requirements.txt` (Rules.md
  section 11.2 -- a new invariant means a new check).
- 52 tests (**407 total**; 393 on the default fast path, 33.7s).

### Measured

    NFR-9  slider move -> updated result   median 260 ms   max 284 ms   budget 2000 ms

Contrast, computed rather than eyeballed, against the chart surface:

    accent  /surface   4.26:1 light   5.35:1 dark    floor 3:1
    neutral /surface   3.49:1 light   3.70:1 dark    floor 3:1

### Fixed
- **The light `neutral` token failed its contrast floor.** `Design.md` gave
  `#8A94A2`, which measures 2.86:1 against the chart surface -- under the 3:1 a
  reference mark needs to be legible. Now `#7B8592` at 3.49:1, still
  unambiguously gray (OKLCH chroma 0.023). The measurement is recorded in
  `Design.md` section 2 rather than the value being quietly swapped.
- `use_container_width` is deprecated with a removal date that has already
  passed; replaced with `width="stretch"`.
- **`streamlit` and `altair` were imported but never declared.** Rules.md
  section 2 had listed them as required since Phase 0; `requirements.txt` had
  never caught up, because no phase before this one imported them. They worked
  locally and failed collection on a clean CI install. Now declared there and
  as a `frontend` extra, so installing the backend does not drag Streamlit in.
  `altair` is named explicitly rather than left transitive through Streamlit:
  `charts.py` imports it directly, and a direct import resting on someone
  else's dependency tree breaks the day they drop it.

### Note
The audit reports **0 skip** for the first time. Its frontend-purity check
(C15) has been skipping since Phase 0 because there was no `frontend/` to scan.

## [0.7.0] - 2026-08-30 - Phase 6: FastAPI service

The HTTP contract from `Architecture.md` section 6. Every route is a thin
adapter over `inference.predictor`, the same code the CLI calls -- so the API's
numbers and the CLI's numbers are the same numbers by construction. Measured
over HTTP against the trained model, `POST /distribution` returns entropy
1.0287 at T=0.2 and 7.4154 at T=2.0, matching the Phase 4 table to four
decimals.

### Added
- `api/schemas.py`: request and response models. FastAPI derives the OpenAPI
  document from these classes, so the constraints published at `/docs` are the
  constraints enforced -- they cannot drift, because they are one object.
- `api/app.py`: `lifespan` loads both checkpoints once (FR-29); routes
  `GET /health`, `GET /models`, `POST /predict`, `POST /predict/batch`,
  `POST /generate`, `POST /distribution`.
- **`POST /distribution`**, which was *not* in the original contract. FR-34
  requires the frontend to chart the next-word distribution and C15 forbids it
  running inference; with no route to supply that, the two rules could not both
  be satisfied. `Architecture.md` section 6 has been amended to document it.
- `tests/test_api.py`: 49 tests, every route and every documented failure (S12).
  They build **tiny untrained checkpoints** instead of reading `runs/`, so
  unlike the trained-model suites they actually execute in CI.
- 49 tests (**355 total**; 348 on the default fast path).

### Measured (NFR-5, warm, dev machine, CPU)

    POST /predict            median   2.3 ms   p95   3.2 ms    budget  100 ms
    POST /predict/batch x32  median  28.0 ms   p95  31.1 ms    -- 0.88 ms/item
    POST /generate 40 words  median  57.4 ms   p95  79.2 ms    budget 2000 ms
    POST /distribution       median   4.4 ms   p95   6.7 ms

Both budgets are met with roughly 25-30x headroom. The tests assert the
*budget*, never the measurement: pinning 57 ms would fail on a slower machine
while telling nobody anything true.

### Fixed
- Logging is configured at the API entry point, but only when nothing has
  configured it already. Under `uvicorn` the root logger has no handlers, so
  the checkpoint-load lines went nowhere and the operator-visible half of FR-29
  did not exist; under pytest the harness owns root, and stamping on it would
  silently disable `caplog` for the whole session.

## [0.6.0] - 2026-08-30 - Phase 5: CLI

Closes **D1**. The reference died at `engine.py:48`:

    train.generate_paragraph(model, test_words, 12, 10)     # 4 arguments

against a function declaring seven parameters. Python raises that only when the
line executes, and that line sat after the training loop -- so the run failed
roughly forty minutes in, every time, having thrown away the model it had just
spent those forty minutes fitting.

The fix is not a test of that one call. It is a static check of *every* call,
because a regression test aimed at a defect's instance catches the instance,
while one aimed at its shape catches the next one too.

### Added
- `tests/test_call_signatures.py`: package-wide call-signature checker. Parses
  every module, indexes **55 functions across 21 modules**, resolves **83
  internal call sites** and checks each against the arity of the function it
  names. Runs in under a second rather than after a hundred epochs. It carries
  negative controls -- a reconstruction of the D1 call that it is required to
  flag, and a floor on resolved call sites, so the check cannot pass by
  quietly resolving nothing.
- `tests/test_cli_integration.py`: the CLI as a user invokes it. In-process
  `main(argv)` for argument handling, exit codes and error reporting;
  subprocess, marked `slow`, only for the two claims that are genuinely about
  the process -- running from an unrelated working directory, and
  reproducibility across separate invocations.
- `slow` and `realdata` pytest markers declared in `pyproject.toml`. The
  default run excludes `slow` to hold the 60-second budget (NFR-6); CI runs
  `pytest -m ""` so no test is *deselected* by the marker filter there.
- 33 tests (**306 total**; 301 on the default fast path).

### Fixed
- **D1**: no call site in the package can now acquire an arity mismatch
  without the suite failing in under a second.
- Exit codes are exhaustive: **0** success, **1** runtime error, **2** usage
  error. `FileNotFoundError` and `OSError` report a message rather than a
  traceback, and a final `except Exception` logs the traceback for diagnosis
  but still returns through the documented code, so no path dumps a raw trace
  at the user.
- The `slow` marker introduced here would have made CI's `pytest -v` deselect
  every test that trains, silently. CI now runs `pytest -v -m ""`.

### Known gap
- CI reports **255 passed, 51 skipped** (run 33286431087). `pytorch/runs/` is
  gitignored, so every test needing a trained checkpoint skips there:
  `test_predictor.py` (21), `test_trained_sentiment.py` (11),
  `test_trained_textgen.py` (10) and 9 of `test_cli_integration.py`. **No
  headline figure in this changelog -- macro-F1 0.8485 (superseded; see 0.8.1),
  perplexity 223.54, the
  D2 entropy curve -- has ever been verified by CI.** They are reproducible
  from a fixed seed on a machine that has trained the models, and that is all
  they currently claim. The D1 checker itself does run in CI (7 passed), as
  does every test that needs no checkpoint. Closing the gap means a
  `--max-steps` smoke train in CI producing a throwaway checkpoint; deferred
  to Phase 9 rather than left implied.

### Verified
All four subcommands run clean from `C:\Users` -- a different drive from the
repository -- resolving every path from the checkpoint and `__file__`:

    predict   'the flight was not great' -> POSITIVE  p=0.751        exit 0
    eval      macro-F1 0.8485 vs 0.4430 on the test split   exit 0   <- superseded
    generate  20 words at T=0.7, top-k 40                            exit 0
    predict --ckpt <missing>  -> "checkpoint not found: ..."         exit 1
    generate --nonsense       -> argparse usage error                exit 2

## [0.5.0] - 2026-08-29 - Phase 4: Sampling & generation

Closes **D2** and D10. D2 is the defect at the centre of the audit: the
reference presented a uniformly-random sampler as a demonstration of softmax
temperature.

Measured on the trained model, entropy against temperature:

      T   entropy   % of uniform
   0.20    1.0287          13.2%
   0.70    3.4333          44.0%
   1.00    5.3862          69.1%
   2.00    7.4154          95.1%
   5.00    7.7563          99.5%

The reference's formulation, on the same model and the same seed, produces
7.7981 nats at every temperature -- exactly ln(2436), the entropy of a uniform
distribution. Its curve is flat. It drew words at random from the whole
vocabulary regardless of the setting, which is what its own saved notebook
output shows.

### Added
- `pytorch/scripts/audit.py`: end-to-end project audit, 23 checks across tests,
  standards conformance, documentation consistency and git hygiene. Run after
  every phase; non-zero exit on any failure. Wired into CI as its own job.
- `inference/sampler.py`: `apply_top_k`, `temperature_distribution`,
  `sample_from_logits`, `greedy_from_logits`, `distribution_entropy`,
  `top_tokens`. Temperature scales logits and nothing else.
- `inference/predictor.py`: `SentimentPredictor` and `TextGenerator`, each
  constructed from a checkpoint and needing no other file. Both report how many
  input tokens were unknown.
- CLI `predict` and `generate`, with generation defaults read from the run's
  config rather than written as literals.
- 42 tests (273 total), including the S5 property gates.

### Fixed
- `Rules.md` A4 cited a superseded one-hot figure (707 MB) in the very rule
  that forbids inventing numbers. Corrected to 931 MB.
- Docstrings added to four public properties.
- **D2**: `softmax(logits / T)`. Models return logits, so no probability vector
  exists to divide by mistake -- the reference's error is unrepresentable here.
  A test reproduces that error deliberately and asserts it is indistinguishable
  from uniform, so the cost of the bug is recorded rather than described.
- **D10**: the demo seed is a config value. `input_words[-28701]` has no
  equivalent anywhere in the codebase.

## [0.4.0] - 2026-08-29 - Phase 3: Text-generation model & training

Closes D6, D9.

Validation perplexity **223.54** against a uniform-guess baseline of 2,436 --
10.9x better than random. Cross-entropy 5.4096 against 7.7981. Top-1 accuracy
0.1482 against a 0.000411 chance rate. Trained in 2m47s on CPU against a
20-minute budget; early stopping fired at epoch 7 and restored epoch 2.

### Added
- `models/textgen_lstm.py`: Embedding -> LSTM -> linear over the vocabulary,
  1,333,124 parameters, taking `h_n[-1]` as the many-to-one reduction. Returns
  raw logits.
- `engine/textgen_task.py`: config-to-checkpoint wiring, perplexity and top-1
  reported beside their baselines.
- `TextGenLSTM` registered in the checkpoint model registry; `describe()` shows
  perplexity beside its baseline.
- CLI `train` and `eval` dispatch on the config's task discriminator.
- 18 tests (231 total).

### Fixed
- **D6**: the model's vocabulary contains no Project Gutenberg licence token at
  all, so it cannot emit one at any temperature. Asserted directly.
- **D9**: peak training RSS is 421 MB -- less than the 931 MB the reference
  allocated for its input array alone. Dataset storage is 0.22 MB of lazily
  sliced int64 indices against 668 MB for the same windows one-hot.

### Unchanged, deliberately
- `engine/trainer.py` was reused with no modification. A test now parses it and
  asserts its code references neither task, so the loop cannot quietly acquire
  task-specific branches.

## [0.3.0] - 2026-08-29 - Phase 2: Sentiment model & training

Closes D4, D5, D8, D11.

*Superseded by 0.8.1: these were selected on the test split. The corrected
held-out figures are 0.8391 / 0.8926 / 0.9303.*

Test macro-F1 **0.8485** (superseded) against a majority-class baseline of
0.4430 (+0.4055). Accuracy 0.8972 (superseded). ROC-AUC 0.9366 (superseded).
Positive-class
recall 0.8068 -- the minority class the reference's setup could ignore for
free. Trained in 1m34s on CPU; early stopping fired at epoch 11 and restored
epoch 6.

### Added
- `models/sentiment_lstm.py`: Embedding -> 2-layer LSTM -> dropout -> linear,
  355,010 parameters, `pack_padded_sequence` so padding never enters the
  recurrence. Returns raw logits.
- `engine/metrics.py`: classification metrics with majority-class baselines
  computed from the labels, plus perplexity helpers for Phase 3.
- `engine/callbacks.py`: `EarlyStopping` and `BestWeights`.
- `engine/trainer.py`: task-agnostic loop with gradient clipping, NaN abort,
  and per-epoch history. Phase 3 reuses it unchanged.
- `engine/sentiment_task.py`: config-to-checkpoint wiring.
- `inference/checkpoint.py`: self-contained `.pt` bundles with a preprocessing
  version guard.
- CLI `train` and `eval`.
- 67 tests (213 total).

### Fixed
- **D4**: every metric is reported beside its baseline, and the baseline is
  computed from the labels rather than hardcoded. Macro-F1 replaces accuracy
  as the model-selection signal.
- **D5**: early stopping with best-weight restore. The saved model is the best
  epoch, never the last.
- **D8**: checkpoints carry weights, config, vocabulary, preprocessing contract
  and provenance. Verified by loading one in a subprocess with no other file
  present.
- **D11**: no "bag of words" anywhere in the new code or docs.
- The loss function is no longer attached to the model. `nn` losses are modules,
  so `model.criterion = CrossEntropyLoss(weight=...)` leaked `criterion.weight`
  into the `state_dict` and broke checkpoint loading. Caught by the D8 test.

## [0.2.0] - 2026-08-29 - Phase 1: Data layer

Closes D3, D6, D7, D9.

### Added
- `data/preprocess.py`: cleaning and tokenisation with no torch dependency.
- `vocab.py`: immutable `Vocab` with `<pad>`/`<unk>` handling; never raises on
  an unknown token.
- `data/sentiment.py`: stratified split, train-only vocabulary, per-batch
  padding, inverse-frequency class weights.
- `data/textgen.py`: Gutenberg stripping, contiguous block split before
  windowing, lazily-sliced `WindowDataset`.
- 94 tests, including a named regression test per defect.

### Fixed
- **D3**: stopword removal removed entirely. All 14 negations survive cleaning.
- **D6**: Project Gutenberg header/footer stripped before tokenisation. The
  validation block was previously 100% licence text.
- **D7**: vocabulary built from the training split only, on both tasks.
- **D9**: windows are lazy slices of one 1-D int64 tensor. 0.22 MB of storage
  against the reference's 931 MB one-hot array.

### Changed
- Text-generation `min_freq` 2 -> 1. At 2, 4.34% of training targets were
  `<unk>`, which the model learns to emit. Sentiment keeps `min_freq=2`.
- `PRD.md` FR-4 and FR-7 amended to match the implemented split semantics.
- Corrected planning figures that were computed on the full corpus before
  splitting: vocabulary 2,436 (not 1,470), windows 27,409 (not 27,419),
  reference one-hot footprint 931 MB (not 707 MB).

## [0.1.0] - 2026-08-29 - Phase 0: Scaffold

### Added
- `lstm-nlp` package, `src/` layout, editable install.
- Pydantic configuration schema, discriminated on `task`, with `extra="forbid"`
  so a typo'd key is an error rather than a silently ignored field.
- Typed exception hierarchy; seeding, device and logging utilities.
- CLI skeleton: `train`, `eval`, `predict`, `generate`.
- 52 tests.
