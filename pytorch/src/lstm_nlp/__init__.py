"""Many-to-one LSTMs for sentiment detection and text generation.

A PyTorch rebuild of the TensorFlow reference in ``modular_code/``.  See ``PRD.md``
for requirements, ``Architecture.md`` for structure and ``Rules.md`` for constraints.

This module must stay free of side effects: no file reads, no network calls, no
global torch configuration at import time (Rules.md section 4).
"""

from __future__ import annotations

__version__ = "0.1.0"

#: Version of the text-cleaning + tokenisation contract.
#:
#: Bump this on ANY change to ``data.preprocess`` or ``Vocab`` semantics.  It is
#: stamped into every checkpoint and verified on load, so an old checkpoint fails
#: loudly instead of being silently mis-tokenised (PRD FR-26, D8).
PREPROCESS_VERSION = "1"

__all__ = ["__version__", "PREPROCESS_VERSION"]
