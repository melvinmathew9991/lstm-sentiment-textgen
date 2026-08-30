"""Streamlit entry point: navigation, the health gate, and the sidebar.

Run with::

    uvicorn lstm_nlp.api.app:app --port 8000     # backend
    streamlit run frontend/app.py                # this

Two commands, no build step (NFR-10).

The health gate runs before either page. If nothing answers, the app says so
once, at the top, with the command that fixes it -- rather than letting each
page discover the same outage separately and render two different versions of
the bad news.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# `streamlit run frontend/app.py` puts frontend/ on sys.path, not its parent,
# so absolute `frontend.*` imports would fail. Adding the package root keeps one
# import style across the app and the tests.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from frontend.api_client import BackendUnreachable, get_client  # noqa: E402
from frontend.components import backend_status, unreachable_banner  # noqa: E402
from frontend.theme import inject_css  # noqa: E402

st.set_page_config(
    page_title="lstm-nlp",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def sidebar_model_info(client) -> None:
    """Show each loaded model's headline metric beside its baseline.

    C16 in the sidebar: the metrics are visible from both pages without
    navigation, and none of them appears without the number it must be judged
    against.
    """
    try:
        models = client.models()
    except Exception:  # noqa: BLE001 - sidebar metadata must never break a page
        return

    if not models:
        return

    st.sidebar.divider()
    st.sidebar.markdown("**Models**")
    for model in models:
        metrics = model.get("metrics", {})
        st.sidebar.markdown(
            f'<div style="font-size:13px;font-weight:500">{model["task"]}</div>'
            f'<div class="lstm-caption lstm-num">vocabulary {model["vocab_size"]:,}</div>',
            unsafe_allow_html=True,
        )
        if "macro_f1" in metrics:
            st.sidebar.markdown(
                f'<div class="lstm-caption lstm-num">macro-F1 {metrics["macro_f1"]:.4f} '
                f'· baseline {metrics.get("baseline_macro_f1", float("nan")):.4f}</div>',
                unsafe_allow_html=True,
            )
        if "perplexity" in metrics:
            st.sidebar.markdown(
                f'<div class="lstm-caption lstm-num">perplexity {metrics["perplexity"]:.2f} '
                f'· baseline {metrics.get("baseline_perplexity", float("nan")):.0f}</div>',
                unsafe_allow_html=True,
            )


def main() -> None:
    """Gate on backend health, then hand control to the selected page."""
    inject_css()
    client = get_client()

    try:
        health = client.health()
    except BackendUnreachable as exc:
        backend_status(None, exc.url)
        unreachable_banner(exc.url, exc.start_command, exc.cause)
        st.stop()

    backend_status(health, client.settings.api_url)
    sidebar_model_info(client)

    pages = [
        st.Page("pages/1_sentiment.py", title="Sentiment", icon="●", default=True),
        st.Page("pages/2_generation.py", title="Generation", icon="◐"),
    ]
    st.navigation(pages).run()


main()
