"""Sentiment page: classify text, and show what the classification rests on.

Three things appear beside every label, and none is optional:

* the class probabilities, so a 0.51 call cannot masquerade as a 0.97 one;
* ``unk_rate``, so a prediction made from words the model has never seen is
  visibly that (Design.md principle 2);
* the majority-class baseline of 0.795, because 89.7% accuracy sounds
  impressive until you know that answering "negative" every time scores 79.5%
  (Rules.md C16).

The preset buttons are the negation pair from D3. The reference's pipeline ran
`nltk`'s stopword list, which contains all 14 negations, so "the flight was not
great" and "the flight was great" reduced to the *same* token sequence. One
click shows they no longer do.
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
from frontend.components import (
    metric_with_baseline,
    model_missing_notice,
    probability_bars,
    unk_badge,
    unreachable_banner,
)
from frontend.theme import palette

#: The D3 demonstration, one click away.
#:
#: Two of these are the canonical pair. The third is here because on the
#: Phase 8 model the canonical pair *moves* (0.984 -> 0.900) without crossing
#: the boundary, while "service was not good" crosses outright (0.806 ->
#: 0.145). Showing only the pair that flips would be a demo; showing one that
#: moves and one that flips is the truth about what the model does.
PRESETS = [
    "the flight was great",
    "the flight was not great",
    "service was not good",
]

#: Majority-class accuracy on the deduplicated airline corpus (80.48% negative).
#:
#: Not a decoration: it is the number any accuracy claim on this page has to be
#: read against, and it is measured, not assumed (Phases.md Phase 1).
MAJORITY_BASELINE = 0.8048

client = get_client()
colours = palette()

st.markdown("# Sentiment")
st.markdown(
    f'<div style="color:{colours["text_muted"]};font-size:15px;margin-top:-8px">'
    "Classify a tweet about an airline. The model reports what it does not know."
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

if "sentiment_text" not in st.session_state:
    st.session_state.sentiment_text = PRESETS[1]

st.markdown("**Try these**")
preset_columns = st.columns(len(PRESETS))
for column, preset in zip(preset_columns, PRESETS, strict=True):
    if column.button(preset, width="stretch", key=f"preset::{preset}"):
        st.session_state.sentiment_text = preset

text = st.text_area("text to classify", key="sentiment_text", height=110)

# Empty input disables the button rather than erroring after the fact
# (Design.md section 7).
classify = st.button(
    "Classify", type="primary", disabled=not text.strip(), key="classify"
)
if not text.strip():
    st.caption("Enter some text to classify.")

st.divider()

if classify:
    try:
        with st.spinner("Classifying…"):
            result = client.predict(text)
    except BackendUnreachable as exc:
        unreachable_banner(exc.url, exc.start_command, exc.cause)
        st.stop()
    except ModelUnavailable as exc:
        model_missing_notice("sentiment", exc.train_command)
        st.stop()
    except ValidationFailed as exc:
        st.error(f"That input was rejected: {exc.detail}")
        st.stop()
    except BackendError as exc:
        st.error(str(exc))
        st.stop()

    label = result["label"]
    probability = result["probabilities"][label]
    colour = colours.get(label, colours["text"])

    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f'<div class="lstm-verdict" style="color:{colour}">{html.escape(label.upper())}</div>'
            f'<div class="lstm-caption lstm-num">probability {probability:.3f}</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        probability_bars(result["probabilities"])
        unk_badge(result["n_unk"], result["n_tokens"])
        # Whether these numbers are calibrated probabilities or merely scores is
        # not a detail the reader can infer, so the page says which.
        if result.get("calibrated"):
            st.caption(
                "Calibrated: a temperature fitted on held-out validation data is "
                "applied, so these read as probabilities rather than scores."
            )
        else:
            st.caption(
                "**Uncalibrated.** These are scores, not calibrated probabilities — "
                "this checkpoint carries no fitted temperature, so treat the "
                "magnitudes as a ranking only."
            )

    with right:
        metric_with_baseline(
            "model test accuracy",
            "0.8974",
            f"{MAJORITY_BASELINE:.4f} (always answer negative)",
        )
        st.caption(
            "A classifier that answers *negative* every time scores "
            f"{MAJORITY_BASELINE:.1%} on this corpus, because 79.5% of it is "
            "complaints. That is the number to judge any accuracy claim against."
        )
else:
    st.markdown(
        f'<div style="color:{colours["text_muted"]};font-size:15px">'
        "Pick a preset or type something, then press <b>Classify</b>."
        "</div>",
        unsafe_allow_html=True,
    )
