"""The temperature visualisation (FR-34) -- the reason this frontend exists.

The reference presented a uniformly-random sampler as a demonstration of
softmax temperature and nothing in its output made that visible (D2). This
chart is the opposite: the dashed rule at ``1/V`` is exactly the distribution
the reference sampled from *at every setting*, so as the user raises the
temperature and watches the bars sink toward that line, they are watching the
defect happen.

Chart decisions, and why (the dataviz method: form, then colour, then marks):

* **Horizontal bars.** The data is magnitude by identity, the identities are
  words of varying length, and there are twelve of them. Horizontal gives the
  labels a full line each and needs no rotation.
* **One series, one colour.** Every bar is ``accent``. Shading bars by their own
  value would double-encode length as hue, spend the only free channel on
  information the chart already shows, and read as a rank the words do not have.
* **No legend.** A legend for a single series is furniture; the title names it.
* **Values in muted ink, never in the series colour.** Marks carry identity,
  text carries text tokens.
* **The dashed rule is an annotation, not a gridline.** Gridlines here are solid
  hairlines a shade off the surface; dashing is reserved for the one line that
  means "threshold", so the two never get confused.

Colours were validated rather than eyeballed, against the chart surface:
``accent`` 4.26:1 light / 5.35:1 dark, ``neutral`` 3.49:1 / 3.70:1, all clear of
the 3:1 floor for a non-text mark. The light ``neutral`` token was corrected
from Design.md's original value to reach it; the measurement is recorded there.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from frontend.theme import palette

#: Bar thickness, px. Capped so the band keeps some air (dataviz mark spec).
BAR_SIZE = 16

#: Vertical pitch per bar, px. The 8px remainder is the gap between fills.
BAR_PITCH = 24


def distribution_chart(
    words: list[dict], temperature: float, uniform_probability: float
) -> alt.LayerChart:
    """Bars for the likeliest next words, with the uniform baseline drawn on.

    Args:
        words: ``[{"word": str, "probability": float}, ...]`` from
            ``POST /distribution``, already ordered most likely first.
        temperature: The temperature these probabilities were computed at,
            named in the title so a screenshot is self-describing.
        uniform_probability: ``1/V``. The reference line, and the whole point.

    Returns:
        A layered Altair chart: bars, value labels, and the baseline rule.
    """
    colours = palette()
    frame = pd.DataFrame(words)
    order = frame["word"].tolist()

    base = alt.Chart(frame)

    bars = base.mark_bar(
        color=colours["accent"],
        cornerRadiusEnd=4,
        size=BAR_SIZE,
    ).encode(
        x=alt.X(
            "probability:Q",
            title="probability",
            axis=alt.Axis(
                format=".3f",
                grid=True,
                gridColor=colours["border"],
                gridWidth=1,
                domainColor=colours["border"],
                tickColor=colours["border"],
                labelColor=colours["text_muted"],
                titleColor=colours["text_muted"],
            ),
        ),
        y=alt.Y(
            "word:N",
            sort=order,
            title=None,
            axis=alt.Axis(
                labelColor=colours["text"],
                labelFontSize=13,
                domainColor=colours["border"],
                tickColor=colours["border"],
            ),
        ),
        tooltip=[
            alt.Tooltip("word:N", title="word"),
            alt.Tooltip("probability:Q", title="p", format=".4f"),
        ],
    )

    # Value at the tip, in text ink. Marks carry identity; text never wears the
    # data colour.
    labels = base.mark_text(
        align="left", dx=6, fontSize=12, color=colours["text_muted"]
    ).encode(
        x=alt.X("probability:Q"),
        y=alt.Y("word:N", sort=order),
        text=alt.Text("probability:Q", format=".3f"),
    )

    # The uniform baseline: 1/V, dashed because it is a threshold, and labelled
    # because a line nobody can name explains nothing.
    reference = pd.DataFrame({"probability": [uniform_probability]})
    rule = (
        alt.Chart(reference)
        .mark_rule(color=colours["neutral"], strokeDash=[6, 4], strokeWidth=2)
        .encode(x=alt.X("probability:Q"))
    )
    rule_label = (
        alt.Chart(reference)
        .mark_text(
            align="left", dx=6, dy=-6, fontSize=11, color=colours["neutral"],
            text=f"uniform 1/V = {uniform_probability:.5f}",
        )
        .encode(x=alt.X("probability:Q"), y=alt.value(0))
    )

    return (
        alt.layer(bars, labels, rule, rule_label)
        .properties(
            height=BAR_PITCH * max(len(frame), 1),
            title=alt.TitleParams(
                text=f"Next-word distribution at T = {temperature:.2f}",
                subtitle="the dashed line is what the reference sampled from at every setting",
                fontSize=16,
                subtitleFontSize=12,
                subtitleColor=colours["text_muted"],
                color=colours["text"],
                anchor="start",
            ),
        )
        .configure_view(stroke=None)
        .configure_axis(labelFont="", titleFont="")
    )


def render_distribution(distribution: dict) -> None:
    """Draw the chart and the numeric alternative beneath it.

    The caption is not decoration: it states in numbers what the bars state in
    pixels, which is what makes the chart legible to a screen reader and to
    anyone who cannot rely on colour (Design.md section 8).

    Args:
        distribution: Body of ``POST /distribution``.
    """
    words = distribution.get("words", [])
    if not words:
        st.info("No distribution to show yet.")
        return

    vocab_size = distribution.get("vocab_size", 0)
    uniform = 1.0 / vocab_size if vocab_size else 0.0
    entropy = distribution.get("entropy", 0.0)
    uniform_entropy = distribution.get("uniform_entropy", 0.0)

    st.altair_chart(
        distribution_chart(words, distribution.get("temperature", 0.0), uniform),
        width="stretch",
    )

    share = (entropy / uniform_entropy * 100) if uniform_entropy else 0.0
    st.markdown(
        f'<div class="lstm-caption lstm-num">'
        f"entropy {entropy:.4f} nats · uniform would be {uniform_entropy:.4f} "
        f"({share:.1f}% of uniform) · vocabulary {vocab_size:,}"
        f"</div>",
        unsafe_allow_html=True,
    )
