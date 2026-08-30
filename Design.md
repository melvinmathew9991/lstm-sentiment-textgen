# Design — Streamlit Frontend

Visual and interaction specification for `pytorch/frontend/`.

**Added 2026-08-29.** This document was deliberately skipped while the interface was CLI + JSON only — there was no visual surface for it to describe, and writing one would have been padding. The Streamlit frontend created that surface, so it exists now. (`PRD.md` §7 records both decisions.)

Scope: palette, typography, layout, component behaviour, states, and the temperature visualisation. Not covered: anything the backend owns — this tier renders results, it does not compute them (`Rules.md` C15).

---

## 1. Design principles

The app has one job beyond being usable: **make the model's uncertainty legible.** The reference project's central failure was presenting a uniformly-random sampler as a demonstration of softmax temperature (D2), and nothing in its output made that visible. Every design decision here serves the opposite goal.

1. **Show the number and its baseline, together.** A prediction at 0.83 confidence means nothing without knowing that guessing "negative" scores 0.795. `Rules.md` C16 is a design rule as much as a code rule.
2. **Surface what the model doesn't know.** `unk_rate` is not a debug field. If 6 of 11 tokens are unknown, the user must see that before trusting the label.
3. **Make temperature felt, not described.** A slider that changes text is a demo. A slider that changes text *and* a probability distribution side by side is a lesson.
4. **Restraint.** Two pages, one accent colour, no decorative chrome. This is an instrument, not a landing page.
5. **Never a raw traceback.** Every failure has a designed state (§7).

---

## 2. Palette

Semantic tokens, not colour names, so the theme can shift without touching page code. Defined in `theme.py`, mirrored in `.streamlit/config.toml`.

### Light (default)

| Token | Hex | Use |
|---|---|---|
| `bg` | `#FFFFFF` | Page background |
| `surface` | `#F6F7F9` | Cards, sidebar, chart plot area |
| `border` | `#E3E6EA` | Hairlines, card edges |
| `text` | `#14181D` | Primary text |
| `text_muted` | `#5B6572` | Labels, captions, baselines |
| `accent` | `#2F6FEB` | Primary actions, focus, active slider |
| `accent_soft` | `#E8F0FE` | Selected rows, chart hover |
| `negative` | `#C4392B` | Negative sentiment |
| `positive` | `#1F8A5B` | Positive sentiment |
| `warn` | `#B26A00` | High `unk_rate`, degraded state |
| `neutral` | `#7B8592` | Baseline reference marks |

### Dark

| Token | Hex |
|---|---|
| `bg` | `#0E1117` |
| `surface` | `#171B22` |
| `border` | `#2A313B` |
| `text` | `#E6E9EE` |
| `text_muted` | `#9AA4B2` |
| `accent` | `#5B8DEF` |
| `accent_soft` | `#1B2740` |
| `negative` | `#E4695C` |
| `positive` | `#3FB07C` |
| `warn` | `#D89A3A` |
| `neutral` | `#6B7583` |

### Rules

- **Sentiment colour is reserved.** `negative` / `positive` appear *only* for class identity — never as generic status colours. A red error message uses `warn`, not `negative`, so red always means "the model said negative".
- **Contrast:** all text/background pairs meet WCAG AA (≥ 4.5:1 body, ≥ 3:1 large). Verified pairs: `text`/`bg` 16.1:1, `text_muted`/`bg` 5.8:1, `accent`/`bg` 5.2:1.
- **Chart marks clear 3:1 against `surface`,** the floor a non-text mark needs to be distinguishable. Measured in Phase 7: `accent`/`surface` 4.26:1 light and 5.35:1 dark; `neutral`/`surface` 3.49:1 light and 3.70:1 dark. Light `neutral` was **#8A94A2** until that measurement — 2.86:1, under the floor — and is now `#7B8592`, which still reads gray (OKLCH chroma 0.023) as a reference mark must.
- **Colour is never the only signal.** Sentiment carries a label and a value; `unk_rate` carries a count; chart series carry labels.

---

## 3. Typography

Streamlit's default system stack — no webfont, so nothing blocks first paint and the app works offline.

```
UI:   -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif
Mono: "SF Mono", "Cascadia Mono", Consolas, "Roboto Mono", monospace
```

| Role | Size | Weight | Notes |
|---|---|---|---|
| Page title | 28px | 600 | One per page |
| Section | 20px | 600 | |
| Body | 15px | 400 | |
| Label | 13px | 500 | Widget labels |
| Caption / baseline | 12px | 400 | `text_muted` |
| Metric value | 34px | 600 | Tabular numerals |
| **Generated text** | 16px / 1.7 | 400 | **Monospace** |

**Generated and input text is monospace.** Token boundaries are the unit of meaning in both tasks — where one token ends and the next begins should be visible without counting. Everything else is sans.

Numbers use `font-variant-numeric: tabular-nums` so values don't jitter as the slider moves.

---

## 4. Layout

Wide layout, persistent sidebar. Two pages via `st.navigation`.

```
┌──────────────┬──────────────────────────────────────────────────┐
│  SIDEBAR     │  MAIN                                            │
│  240px       │                                                  │
│              │  Page title                             28/600   │
│  ● Backend   │  One-line description                    15/muted │
│    healthy   │  ──────────────────────────────────────────────  │
│    :8000     │                                                  │
│              │  [ input region ]                                │
│  ─────────   │                                                  │
│  Sentiment   │  ──────────────────────────────────────────────  │
│  Generation  │                                                  │
│              │  [ result region ]                               │
│  ─────────   │                                                  │
│  Model info  │                                                  │
│  vocab 4,045 │                                                  │
│  macroF1 .81 │                                                  │
│  base   .443 │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

The sidebar always shows **backend status** (top) and **model metadata with baselines** (bottom) from `GET /models`. Those two facts contextualise everything on the right and must never require navigation to find.

Vertical rhythm: 8px base unit. Section gap 32px, control gap 16px, label-to-control 4px. Max content width 1100px.

### 4.1 Sentiment page

```
Sentiment                            ← 28/600
Classify a tweet. 1 = positive.      ← 15/muted

┌──────────────────────────────────────────────┐
│ the flight was not great                     │  text_area, mono, 4 rows
└──────────────────────────────────────────────┘
[ Classify ]  ← accent

──────────────────────────────────────────────

  NEGATIVE                 ← 34/600, colour = negative
  probability 0.87

  negative ████████████████████░░░  0.87       ← probability_bars
  positive ███░░░░░░░░░░░░░░░░░░░░  0.13

  ⚠ 2 of 5 tokens unknown to the model (40%)   ← unk_badge, warn
  Majority-class baseline: 0.795 accuracy      ← 12/muted
```

**Try these** — three preset buttons seeded with the negation pairs from D3 (`"the flight was great"` / `"the flight was not great"`). The single clearest demonstration that the rebuild fixed something real, one click away.

### 4.2 Generation page

```
Generation                                    ← 28/600
Next-word LSTM over Alice in Wonderland.

┌─ seed ──────────────────────────┐  words  [ 40 ]
│ alice was beginning to          │  top-k  [ 40 ]
└─────────────────────────────────┘  seed   [    ]  (blank = random)

temperature  0.70
0.1 ──────────●──────────────── 2.0           ← accent track
greedy · repetitive      inventive · incoherent   ← 12/muted, both ends

[ Generate ]

──────────────────────────────────────────────

alice was beginning to get very tired of sitting by her sister on the
bank and of having nothing to do once or twice she had peeped into
                                              ← 16/1.7 MONO
                                              ← seed in accent, rest in text

Next-word distribution at T = 0.70            ← 20/600
┌──────────────────────────────────────────┐
│ get   ████████████████  0.21              │  altair horizontal bars
│ be    ██████████  0.13                    │  top 12 tokens
│ feel  ███████  0.09                       │
│ ...                                       │
│ ─ ─ ─ ─ uniform baseline 1/2436 ─ ─ ─ ─   │  neutral dashed rule
└──────────────────────────────────────────┘
entropy 3.12 nats · uniform would be 7.798    ← 12/muted
```

---

## 5. Components

| Component | Behaviour |
|---|---|
| `metric_with_baseline(label, value, baseline)` | Large tabular value, caption `"baseline: X"` in `text_muted`. **The baseline argument is required** — there is no overload without it (C16). |
| `probability_bars(probs)` | One horizontal bar per class, class-coloured, value right-aligned, tabular. Sorted descending. |
| `unk_badge(n_unk, n_tokens)` | Hidden at 0. `text_muted` below 20%. `warn` with a ⚠ at ≥ 20%. Always shows the count *and* the rate — "2 of 5" is more actionable than "40%". |
| `backend_status(health)` | Sidebar dot + label. `positive` healthy · `warn` degraded (some models missing) · `negative` unreachable. |
| `temperature_slider()` | 0.1–2.0, step 0.05, default 0.7. Anchored end labels. Never reaches 0 — the config layer rejects T=0 and the UI must not offer it. |
| `distribution_chart(probs, T)` | Altair horizontal bars, top 12, dashed uniform reference line, entropy caption. |

---

## 6. The temperature visualisation

The centrepiece (FR-34). Its whole purpose is that the user *observes* what the reference merely claimed.

| T | Chart | Text | Caption |
|---|---|---|---|
| 0.1 | one dominant spike | repetitive, loops | "near-greedy" |
| 0.7 | few clear peaks | coherent, varied | "balanced — default" |
| 1.5 | flattening | inventive, drifting | "high variety" |
| 2.0 | approaching flat | incoherent | "near-uniform" |

The dashed **uniform baseline** at `1/V` is what makes it land: at T=2.0 the bars visibly approach it. That flat line is exactly the distribution the reference sampled from *at every temperature setting* — the 707-token vocabulary drawn at random that produced `"the alone court, to. pop sits bursting particularly"`.

Chart updates on slider release, not on drag, to stay inside the 2s budget (NFR-9).

---

## 7. States

Every state below is designed. None is a traceback.

| State | Presentation |
|---|---|
| **First load** | Inputs prefilled with working defaults. Result region shows a muted prompt, not an empty box. |
| **Loading** | `st.spinner` on the triggering control. Inputs stay enabled. |
| **Backend unreachable** | Full-width `st.error`: what failed, the URL tried, and the exact start command. Inputs disabled. No retry loop. |
| **Model not loaded (503)** | Page-level `st.warning` naming the model and the `lstm-nlp train --config ...` command that produces it. The other page stays usable. |
| **Validation error (422)** | `st.error` beside the offending control, quoting the backend's message. |
| **Empty input** | Button disabled with a caption. Not an error after the fact. |
| **All tokens unknown** | Result renders with a prominent `warn` badge: the model saw nothing it recognised, so the prediction is uninformative. |

---

## 8. Accessibility

- WCAG AA contrast on all text (§2), verified pairs listed there.
- Colour never sole carrier of meaning (§2).
- Native Streamlit widgets throughout — keyboard navigation and screen-reader labels come free. No custom HTML controls, which is also why `streamlit-*` community components are banned (`Rules.md` §2).
- Every input has a visible label; placeholders are never labels.
- Focus ring: 2px `accent`, never suppressed.
- Charts carry text alternatives — the entropy caption states numerically what the bars show.

---

## 9. What this app is not

- No auth, no accounts, no persistence between sessions.
- No training controls. Training is a CLI concern with a different failure profile and runtime; putting a "train" button in a web UI invites a 20-minute request.
- No file upload, no batch UI. `POST /predict/batch` exists for programmatic callers.
- No dataset browser or EDA. That is the frozen notebook's job.
- No custom components, no injected JS. Theme CSS only.
