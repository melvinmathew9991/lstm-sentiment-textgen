"""Reusable widgets, per ``Design.md`` section 5.

Two signatures here encode rules rather than preferences, and both are enforced
by being *unavoidable*:

* :func:`metric_with_baseline` takes ``baseline`` as a required positional
  argument. There is no overload without it. C16 says no metric appears in the
  UI without its baseline, and a rule that depends on remembering to pass an
  optional argument is a rule that will be broken by the third page someone
  adds.
* :func:`unk_badge` always shows the count *and* the rate. "2 of 5" is
  actionable in a way "40%" is not, because it tells the reader how much text
  the judgement rests on.

Nothing in this module makes an HTTP call; it renders what a page hands it.
"""

from __future__ import annotations

import html

import streamlit as st

from frontend.theme import palette

#: ``unk_rate`` at or above this is called out in ``warn`` with a symbol.
#:
#: At one unknown token in five, the model is guessing from a sentence it
#: largely cannot read, and the user has to know that before believing a label.
UNK_WARN_THRESHOLD = 0.20


def metric_with_baseline(label: str, value: str, baseline: str) -> None:
    """Render a metric above the number it must be judged against.

    Args:
        label: What the metric is.
        value: The measured value, preformatted.
        baseline: What that value beats, preformatted. **Required** -- a metric
            without its baseline is not shown at all (Rules.md C16).
    """
    colours = palette()
    st.markdown(
        f"""
        <div>
          <div style="font-size:13px;font-weight:500;color:{colours["text_muted"]}">
            {html.escape(label)}
          </div>
          <div class="lstm-num" style="font-size:34px;font-weight:600;line-height:1.2">
            {html.escape(value)}
          </div>
          <div class="lstm-caption">baseline: {html.escape(baseline)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_bars(probabilities: dict[str, float]) -> None:
    """One class-coloured bar per class, sorted most likely first.

    Args:
        probabilities: Class name to probability.
    """
    colours = palette()
    for name, value in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True):
        colour = colours.get(name, colours["accent"])
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin:6px 0">
              <div style="width:72px;font-size:13px;color:{colours["text_muted"]}">
                {html.escape(name)}
              </div>
              <div class="lstm-bar-track" style="flex:1">
                <div class="lstm-bar-fill"
                     style="width:{value * 100:.1f}%;background:{colour}"></div>
              </div>
              <div class="lstm-num" style="width:56px;text-align:right;font-size:13px">
                {value:.3f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def unk_badge(n_unk: int, n_tokens: int) -> None:
    """Report how much of the input the model could not read.

    Hidden at zero unknowns -- a badge that is always present stops being read.
    Below the threshold it is muted; at or above it, ``warn`` with a symbol,
    because colour must never be the only signal (Design.md section 8).

    Args:
        n_unk: Tokens the model had never seen.
        n_tokens: Total tokens after cleaning.
    """
    if n_unk <= 0 or n_tokens <= 0:
        return

    colours = palette()
    rate = n_unk / n_tokens
    high = rate >= UNK_WARN_THRESHOLD
    colour = colours["warn"] if high else colours["text_muted"]
    mark = "⚠ " if high else ""
    tail = (
        " — this prediction rests on words the model does not know"
        if rate >= 1.0
        else ""
    )
    st.markdown(
        f'<span class="lstm-badge" style="color:{colour};border-color:{colour}">'
        f"{mark}{n_unk} of {n_tokens} tokens unknown to the model "
        f"({rate:.0%}){tail}</span>",
        unsafe_allow_html=True,
    )


def backend_status(health: dict | None, url: str) -> None:
    """Sidebar indicator: healthy, degraded, or unreachable.

    Args:
        health: Body of ``GET /health``, or ``None`` if nothing answered.
        url: The backend this app is configured against, always shown -- when
            something is wrong, "which backend" is the first question.
    """
    colours = palette()
    if health is None:
        colour, label = colours["negative"], "unreachable"
    else:
        loaded = health.get("models", {})
        if all(loaded.values()):
            colour, label = colours["positive"], "healthy"
        elif any(loaded.values()):
            colour, label = colours["warn"], "degraded"
        else:
            colour, label = colours["negative"], "no models"

    st.sidebar.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px">
          <span style="width:9px;height:9px;border-radius:50%;background:{colour};
                       display:inline-block"></span>
          <span style="font-size:13px;font-weight:500">backend {label}</span>
        </div>
        <div class="lstm-caption">{html.escape(url)}</div>
        """,
        unsafe_allow_html=True,
    )


def unreachable_banner(url: str, start_command: str, cause: str) -> None:
    """The full-page state when nothing answered (FR-35, Design.md section 7).

    Names what failed, the URL tried, and the exact command that fixes it. No
    spinner and no retry loop: an app that keeps trying in the background tells
    the user less than one that stops and explains.
    """
    st.error(
        f"**Backend unreachable.** Nothing answered at `{url}`.\n\n"
        f"Cause: {cause}\n\n"
        f"Start it with:\n\n"
        f"```bash\n{start_command}\n```"
    )


def model_missing_notice(task: str, train_command: str) -> None:
    """The page-level state when one model never loaded (503).

    Scoped to the page on purpose: the other task is still usable, and blanking
    the whole app would hide a working feature behind an unrelated failure.
    """
    st.warning(
        f"**The {task} model is not loaded.** The backend is running, but no "
        f"{task} checkpoint was found.\n\n"
        f"Train one with:\n\n```bash\n{train_command}\n```"
    )


#: Temperature slider bounds (Design.md section 5, PRD FR-33).
#:
#: The floor is 0.1, not 0: the config layer rejects T <= 0 outright, and a UI
#: that offers a value the backend will refuse is a 422 waiting to happen. The
#: ceiling is 2.0 because past it the distribution is already visually flat --
#: the lesson has landed and the remaining range only adds travel.
TEMPERATURE_MIN, TEMPERATURE_MAX = 0.1, 2.0
TEMPERATURE_STEP = 0.05
TEMPERATURE_DEFAULT = 0.7


def temperature_slider(key: str = "temperature") -> float:
    """The control the whole page is built around.

    Anchored at both ends with what the setting *does*, because "0.1 to 2.0"
    tells a reader nothing about which direction is which.

    Args:
        key: Streamlit widget key.

    Returns:
        The selected temperature.
    """
    value = st.slider(
        "temperature",
        min_value=TEMPERATURE_MIN,
        max_value=TEMPERATURE_MAX,
        value=TEMPERATURE_DEFAULT,
        step=TEMPERATURE_STEP,
        key=key,
        help="Scales the logits before softmax. Lower is greedier, higher is flatter.",
    )
    left, right = st.columns(2)
    left.markdown('<div class="lstm-caption">greedy · repetitive</div>',
                  unsafe_allow_html=True)
    right.markdown(
        '<div class="lstm-caption" style="text-align:right">inventive · incoherent</div>',
        unsafe_allow_html=True,
    )
    return value


def temperature_reading(temperature: float) -> str:
    """A one-word characterisation of the current setting, for the caption.

    Mirrors the table in ``Design.md`` section 6 so the prose and the chart
    agree about what a given temperature means.
    """
    if temperature <= 0.3:
        return "near-greedy — repetitive, loops"
    if temperature <= 1.0:
        return "balanced — coherent and varied"
    if temperature <= 1.6:
        return "high variety — inventive, drifting"
    return "near-uniform — incoherent"
