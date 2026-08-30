"""Generation page: the temperature lesson, made observable (FR-34).

Text and distribution are shown together and driven by the same slider, so the
relationship between them is watched rather than described. That pairing is the
entire point. The reference shipped a temperature parameter that changed
nothing -- its sampler drew uniformly from the whole vocabulary at every setting
-- and no output it produced would have revealed that. Here, moving the slider
to 2.0 sinks the bars visibly toward the dashed uniform line, and the dashed
line is exactly where the reference sat the whole time.

The distribution comes from the backend, never from a local calculation
(Rules.md C15). A chart computed alongside the sampler rather than *from* it is
free to disagree with it, which is the shape of the defect this page exists to
explain.
"""

from __future__ import annotations

import html

import streamlit as st

from frontend.api_client import (
    BackendError,
    BackendUnreachable,
    ModelUnavailable,
    ValidationFailed,
    get_client,
)
from frontend.charts import render_distribution
from frontend.components import (
    model_missing_notice,
    temperature_reading,
    temperature_slider,
    unk_badge,
    unreachable_banner,
)
from frontend.theme import palette

DEFAULT_SEED = "alice was beginning to"

client = get_client()
colours = palette()

st.markdown("# Generation")
st.markdown(
    f'<div style="color:{colours["text_muted"]};font-size:15px;margin-top:-8px">'
    "Next-word LSTM over <i>Alice in Wonderland</i>. "
    "Move the temperature and watch both the text and the distribution change."
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

left, right = st.columns([2, 1])
with left:
    seed = st.text_input("seed", value=DEFAULT_SEED, key="seed")
    temperature = temperature_slider()
with right:
    n_words = st.number_input("words", min_value=1, max_value=200, value=40, step=10)
    top_k = st.number_input(
        "top-k (0 = off)", min_value=0, max_value=2000, value=40, step=10
    )
    rng_seed = st.number_input(
        "rng seed (0 = random)", min_value=0, max_value=10**6, value=0, step=1,
        help="Set a non-zero value to get the same passage every time (FR-23).",
    )

st.caption(f"T = {temperature:.2f} — {temperature_reading(temperature)}")

generate = st.button(
    "Generate", type="primary", disabled=not seed.strip(), key="generate"
)
if not seed.strip():
    st.caption("Enter a seed to generate from.")

st.divider()


def _fail(exc: BackendError) -> None:
    """Render the designed state for a failure, then stop the page (FR-35)."""
    if isinstance(exc, BackendUnreachable):
        unreachable_banner(exc.url, exc.start_command, exc.cause)
    elif isinstance(exc, ModelUnavailable):
        model_missing_notice("textgen", exc.train_command)
    elif isinstance(exc, ValidationFailed):
        st.error(f"That request was rejected: {exc.detail}")
    else:
        st.error(str(exc))
    st.stop()


if generate:
    top_k_value = int(top_k) or None
    rng_value = int(rng_seed) or None

    try:
        with st.spinner("Generating…"):
            result = client.generate(
                seed, int(n_words), float(temperature), top_k_value, rng_value
            )
            distribution = client.distribution(seed, float(temperature), top_k_value, n=12)
    except BackendError as exc:
        _fail(exc)

    seed_text = " ".join(result["seed_tokens"])
    generated_text = " ".join(result["generated_tokens"])
    st.markdown(
        f'<div class="lstm-generated">'
        f'<span class="lstm-seed">{html.escape(seed_text)}</span> '
        f"{html.escape(generated_text)}</div>",
        unsafe_allow_html=True,
    )
    unk_badge(result["n_unk_in_seed"], len(result["seed_tokens"]))

    st.write("")
    render_distribution(distribution)
else:
    st.markdown(
        f'<div style="color:{colours["text_muted"]};font-size:15px">'
        "Set a temperature and press <b>Generate</b>. Try 0.2, then 2.0, and "
        "watch the bars approach the dashed uniform line."
        "</div>",
        unsafe_allow_html=True,
    )
