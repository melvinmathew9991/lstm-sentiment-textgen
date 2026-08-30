"""Palette tokens and shared CSS, per ``Design.md`` sections 2 and 3.

Tokens are semantic (``accent``, ``warn``) rather than literal (``blue``,
``orange``) so a theme change touches this file and nothing else.

One rule here is load-bearing rather than decorative: **sentiment colour is
reserved**. ``negative`` and ``positive`` appear only for class identity, never
as generic status colours, so red on this page always means "the model said
negative" and never "something went wrong". Warnings use ``warn``.
"""

from __future__ import annotations

import streamlit as st

#: Light palette (default). Contrast pairs verified in ``Design.md`` section 2:
#: text/bg 16.1:1, text_muted/bg 5.8:1, accent/bg 5.2:1 -- all WCAG AA or better.
LIGHT = {
    "bg": "#FFFFFF",
    "surface": "#F6F7F9",
    "border": "#E3E6EA",
    "text": "#14181D",
    "text_muted": "#5B6572",
    "accent": "#2F6FEB",
    "accent_soft": "#E8F0FE",
    "negative": "#C4392B",
    "positive": "#1F8A5B",
    "warn": "#B26A00",
    # Design.md gives #8A94A2 here. Measured against the chart surface (#F6F7F9)
    # that is 2.86:1, under the 3:1 floor a reference mark has to clear to be
    # legible; #7B8592 is 3.49:1 and still reads unambiguously gray (OKLCH
    # chroma 0.023). Corrected in Design.md section 2 with the measurement.
    "neutral": "#7B8592",
}

#: Dark palette. Same token names, so no component knows which is active.
DARK = {
    "bg": "#0E1117",
    "surface": "#171B22",
    "border": "#2A313B",
    "text": "#E6E9EE",
    "text_muted": "#9AA4B2",
    "accent": "#5B8DEF",
    "accent_soft": "#1B2740",
    "negative": "#E4695C",
    "positive": "#3FB07C",
    "warn": "#D89A3A",
    "neutral": "#6B7583",
}

#: Monospace stack for generated text and user input.
#:
#: Token boundaries are the unit of meaning in both tasks, so where one token
#: ends and the next begins has to be visible without counting (Design.md 3).
MONO_STACK = '"SF Mono", "Cascadia Mono", Consolas, "Roboto Mono", monospace'


def palette() -> dict[str, str]:
    """The active palette, following Streamlit's own light/dark setting."""
    try:
        base = st.get_option("theme.base")
    except (KeyError, AttributeError):  # option table unavailable outside a run
        base = "light"
    return DARK if base == "dark" else LIGHT


def inject_css() -> None:
    """Install the shared stylesheet. Call once per page, before content.

    Tabular numerals are not a flourish: without them, digits change width as
    the temperature slider moves and every number on the page jitters, which
    reads as instability in the model rather than in the font.
    """
    colours = palette()
    st.markdown(
        f"""
        <style>
          :root {{
            --bg: {colours["bg"]};
            --surface: {colours["surface"]};
            --border: {colours["border"]};
            --text: {colours["text"]};
            --text-muted: {colours["text_muted"]};
            --accent: {colours["accent"]};
            --negative: {colours["negative"]};
            --positive: {colours["positive"]};
            --warn: {colours["warn"]};
            --neutral: {colours["neutral"]};
          }}
          .block-container {{ max-width: 1100px; padding-top: 2.5rem; }}
          [data-testid="stMetricValue"], .lstm-num {{
            font-variant-numeric: tabular-nums;
          }}
          .lstm-caption {{
            color: var(--text-muted); font-size: 12px; margin-top: 2px;
          }}
          .lstm-generated {{
            font-family: {MONO_STACK};
            font-size: 16px; line-height: 1.7;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px; padding: 16px;
            white-space: pre-wrap; word-break: break-word;
          }}
          .lstm-seed {{ color: var(--accent); font-weight: 600; }}
          .lstm-verdict {{
            font-size: 34px; font-weight: 600; line-height: 1.2;
            font-variant-numeric: tabular-nums;
          }}
          .lstm-badge {{
            display: inline-block; font-size: 13px; padding: 4px 10px;
            border-radius: 999px; border: 1px solid var(--border);
          }}
          .lstm-bar-track {{
            background: var(--surface); border-radius: 4px;
            height: 10px; width: 100%; overflow: hidden;
          }}
          .lstm-bar-fill {{ height: 100%; border-radius: 4px; }}
          .stTextArea textarea, .stTextInput input {{ font-family: {MONO_STACK}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
