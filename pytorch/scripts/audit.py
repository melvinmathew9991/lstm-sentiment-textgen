"""End-to-end project audit.

Run after every phase (`Rules.md` section 11). Verifies that the project still
meets its own standards -- not just that the newest phase works.

    python scripts/audit.py            # full audit
    python scripts/audit.py --fast     # skip the test suite and network checks

Exit code 0 if no FAIL. WARN never fails the build; it flags drift worth a look.

The point of this file is that standards which are only written down decay.
Every check here corresponds to a rule in `Rules.md` or a measured value in
`PRD.md`/`Phases.md`, and each one names what it is enforcing.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]      # pytorch/
REPO_ROOT = PKG_ROOT.parent                          # repository root
SRC = PKG_ROOT / "src" / "lstm_nlp"
TESTS = PKG_ROOT / "tests"
FRONTEND = PKG_ROOT / "frontend"

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Result:
    status: str
    name: str
    rule: str = ""
    detail: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# canonical measured values -- the single source of truth
# --------------------------------------------------------------------------- #
# Computed from the real corpora and asserted by the test suite. Documentation
# that disagrees with these is stale documentation (Rules.md A4).

# Each entry is (literal as it appears in prose, what it is). The literal form
# matters: these are matched against documents, which use thousands separators.
#
# Corrected 2026-08-30. Seven of these were pre-deduplication values -- 8,078 /
# 3,463 / 4,505 / 5.23% / 3.38% / 3.884 / 0.7953 / 0.4430 -- and one predated
# v1.2.0's third block (27,409 windows). That mattered more than it looks,
# because `check_measured_values_documented` below **requires** these literals to
# appear in the documents: the audit was actively holding the superseded
# baselines in place while `STALE_FIGURES` was never told about them. Two checks
# in this file pulling in opposite directions is why the correction from v1.1.0
# stopped at `Memory.md` and never reached `PARITY.md`, `Architecture.md`,
# `Phases.md`, the configs or the model docstrings.
MEASURED = {
    "sentiment_rows_raw": ("11,541", "rows in the CSV"),
    "sentiment_rows_kept": ("11,271", "rows after deduplication"),
    "sentiment_train": ("6,705", "training block"),
    "sentiment_val": ("1,184", "validation block"),
    "sentiment_test": ("3,382", "held-out test block"),
    "sentiment_raw_vocab": ("8,702", "train-only raw vocabulary"),
    "sentiment_vocab": ("4,045", "sentiment vocab @ min_freq=2"),
    "sentiment_params": ("325,570", "SentimentLSTM parameters at V=4,045"),
    "sentiment_test_oov": ("5.77", "test OOV rate, %"),
    "sentiment_train_oov": ("3.70", "train <unk> rate, %"),
    "class_weight_pos": ("4.130", "positive class weight"),
    "baseline_accuracy": ("0.8048", "majority-class accuracy baseline"),
    "baseline_macro_f1": ("0.4459", "majority-class macro-F1 baseline"),
    "alice_chars_raw": ("164,045", "alice.txt characters"),
    "alice_chars_stripped": ("144,607", "characters after the Gutenberg strip"),
    "alice_tokens": ("27,429", "alice tokens"),
    "textgen_vocab": ("2,436", "textgen vocab / perplexity baseline"),
    "textgen_windows": ("27,399", "windows across the three blocks"),
    "textgen_params": ("1,333,124", "TextGenLSTM parameters at V=2,436"),
    "calibration_temperature": ("2.6715", "fitted calibration temperature"),
}

# The subset a reader must be able to find in the documents. Derived from
# MEASURED so there is one list of canonical numbers in this file, not two.
HEADLINE_KEYS = (
    "baseline_accuracy", "baseline_macro_f1", "textgen_vocab",
    "sentiment_vocab", "class_weight_pos", "alice_tokens",
)

# Superseded figures. Any of these appearing in a doc outside an explicitly
# historical note means a correction did not propagate.
STALE_FIGURES = {
    "1,470": "textgen vocab -- computed pre-split; correct value is 2,436",
    "27,419": "windows -- full-corpus count; correct value is 27,409 (two blocks)",
    "707 MB": "hypothetical one-hot; the reference actually allocated 931 MB",
    "961,214": "textgen params at the old vocab; correct value is 1,333,124",
    "0.8485": "sentiment macro-F1 selected on the test split; held-out value is 0.8391",
    "0.8972": "sentiment accuracy selected on the test split; held-out value is 0.8926",
    "0.9366": "sentiment ROC-AUC selected on the test split; held-out value is 0.9303",
    "0.8391": "sentiment macro-F1 inflated by duplicate rows; corrected value is 0.8300",
    "0.9303": "sentiment ROC-AUC inflated by duplicate rows; corrected value is 0.9126",
    # --- added 2026-08-30: everything v1.1.0 and v1.2.0 moved but did not list.
    # The direct metrics above were registered when they changed; the figures
    # *derived* from the same corpus change were not, so they survived in eight
    # places at once. Enumerating the dependents is the actual fix.
    "0.7953": "accuracy baseline before deduplication; correct value is 0.8048",
    "0.4430": "macro-F1 baseline before deduplication; correct value is 0.4459",
    "3.884": "positive class weight before deduplication; correct value is 4.130",
    "4,505": "sentiment vocab before deduplication; correct value is 4,045",
    "9,566": "train-only raw vocab before deduplication; correct value is 8,702",
    "355,010": "SentimentLSTM params at V=4,505; correct value is 325,570",
    "328,002": "SentimentLSTM params at V=4,083; correct value is 325,570",
    "5.23%": "test OOV before deduplication; correct value is 5.77%",
    "3.38%": "train <unk> rate before deduplication; correct value is 3.70%",
    "3,463": "test rows before deduplication; correct value is 3,382",
    "8,078": "train rows under the two-way split; correct value is 6,705",
    "1.5922": "calibration temperature before deduplication; correct value is 2.6715",
    "0.0609": "test ECE before deduplication; correct value is 0.0803",
    "0.0324": "calibrated test ECE before deduplication; correct value is 0.0223",
    "27,409": "windows across two blocks; correct value is 27,399 across three",
    "223.54": "textgen perplexity on the selection set; held-out value is 267.54",
}

# Files that are read as current fact and therefore held to STALE_FIGURES too.
# Documents alone was not enough: the stale vocabulary and class weight lived in
# `configs/sentiment.yaml`, in two model docstrings and in a test docstring,
# none of which this gate could see before 2026-08-30.
STALE_SCAN_GLOBS = ("configs/*.yaml", "src/**/*.py", "tests/**/*.py", "scripts/*.py")

BANNED_IMPORTS = {
    "tensorflow": "the framework being replaced",
    "keras": "the framework being replaced",
    "nltk": "its stopword list is the direct cause of D3",
    "torchtext": "unmaintained, hard-pins torch",
    "projectpro": "network calls in the training critical path",
    "transformers": "out of scope -- this project is about LSTMs",
    "sentence_transformers": "out of scope",
    "pytorch_lightning": "hides the mechanics this project exists to show",
    "lightning": "hides the mechanics this project exists to show",
    "ignite": "hides the mechanics this project exists to show",
    "fastai": "hides the mechanics this project exists to show",
    "gensim": "pretrained embeddings are out of scope",
    "matplotlib": "charting belongs to altair in the frontend",
    "seaborn": "charting belongs to altair in the frontend",
}

# Defect -> substring that must appear in at least one test name.
# Needles are matched against test FUNCTION NAMES, never against file text.
#
# Until Phase 9 this matched the concatenated source of every test file, which
# made the check close to vacuous: "config", "checkpoint" and "baseline" appear
# in prose everywhere, so D4, D8 and D10 were "covered" by words in docstrings.
# D10's intended needle, "magic", matched nothing at all -- its whole verdict
# rested on "config" occurring somewhere across 11 files.
#
# Each needle below was verified to appear in a real test function name.
DEFECT_TESTS = {
    "D1": ["call_matches_its_signature", "checker_detects_the_reference_defect"],
    "D2": ["entropy_rises_with_temperature", "entropy_increases", "argmax"],
    "D3": ["negations_survive_cleaning", "negation_changes_the_prediction"],
    "D4": ["equals_the_baseline", "majority_baseline"],
    "D5": ["best_not_last"],
    "D6": ["gutenberg", "boilerplate"],
    "D7": ["vocab_built_from_train_only", "unknown_words_do_not_raise"],
    "D8": ["carries_the_vocabulary", "self_contained"],
    "D9": ["int_indices_not_onehot", "storage_is_o_tokens"],
    "D10": ["falls_back_to_config_defaults"],
    "D11": [],  # terminology -- enforced by the stale-phrase check instead
}

# Which phase closes each defect. Defects from unfinished phases are SKIP.
DEFECT_PHASE = {
    "D1": 5, "D2": 4, "D3": 1, "D4": 2, "D5": 2, "D6": 1,
    "D7": 1, "D8": 2, "D9": 1, "D10": 5, "D11": 8,
}

TRAILER_PATTERNS = [
    r"(?im)^\s*co-authored-by\s*:",
    r"(?im)^\s*generated[- ]with",
    r"(?im)^\s*claude-session\s*:",
    r"(?i)claude\.ai/code",
    r"(?i)🤖\s*generated",
    r"(?im)^\s*signed-off-by\s*:",
]


def run(cmd: list[str], cwd: Path = PKG_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)


def py_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def current_phase() -> int:
    """Highest phase marked done in Phases.md."""
    text = (REPO_ROOT / "docs" / "Phases.md").read_text(encoding="utf-8")
    done = re.findall(r"\|\s*P(\d)\s[^|]*\|\s*✅", text)
    return max((int(d) for d in done), default=-1)


# --------------------------------------------------------------------------- #
# A. tests
# --------------------------------------------------------------------------- #


def check_test_suite(fast: bool) -> Result:
    if fast:
        return Result(SKIP, "Test suite", "Rules.md 6")
    proc = run([sys.executable, "-m", "pytest", "-q", "--no-header"])
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1:]
    if proc.returncode != 0:
        return Result(FAIL, "Test suite", "Rules.md 6", tail or ["pytest failed"])
    m = re.search(r"(\d+) passed", proc.stdout)
    count = m.group(1) if m else "?"
    detail = [f"{count} passed"]
    if "skipped" in proc.stdout:
        detail.append(tail[0] if tail else "")
    return Result(PASS, "Test suite", "Rules.md 6", detail)


def test_function_names() -> str:
    """Every ``def test_*`` name in the suite, lowercased and newline-joined.

    Function names rather than file text: a defect is covered by a test that
    exists, not by its identifier appearing in someone's docstring.
    """
    found = []
    for path in py_files(TESTS):
        found += re.findall(r"^\s*(?:async )?def (test_\w+)",
                            path.read_text(encoding="utf-8"), re.M)
    return "\n".join(found).lower()


def check_defect_coverage() -> Result:
    names = test_function_names()
    phase = current_phase()
    missing, covered, pending = [], [], []
    for defect, needles in DEFECT_TESTS.items():
        if not needles:
            continue
        if any(n in names for n in needles):
            covered.append(defect)
        elif DEFECT_PHASE[defect] > phase:
            pending.append(f"{defect} (phase {DEFECT_PHASE[defect]})")
        else:
            missing.append(f"{defect} -- expected a test matching {needles}")
    if missing:
        return Result(FAIL, "Defect regression coverage", "Architecture.md 9", missing)
    detail = [f"covered: {', '.join(covered)}"]
    if pending:
        detail.append(f"pending: {', '.join(pending)}")
    return Result(PASS, "Defect regression coverage", "Architecture.md 9", detail)


# --------------------------------------------------------------------------- #
# B. standards conformance
# --------------------------------------------------------------------------- #


def check_banned_imports() -> Result:
    hits = []
    for path in list(py_files(SRC)) + list(py_files(TESTS)) + list(py_files(FRONTEND)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            hits.append(f"{rel(path)}: unparseable -- {exc}")
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                top = mod.split(".")[0]
                if top in BANNED_IMPORTS:
                    hits.append(f"{rel(path)}:{node.lineno} imports {top} -- {BANNED_IMPORTS[top]}")
    status = FAIL if hits else PASS
    return Result(status, "No banned imports", "Rules.md 2", hits or ["none of 14 banned packages imported"])


def check_frontend_purity() -> Result:
    """C15: the frontend is a pure API client."""
    if not FRONTEND.exists():
        return Result(SKIP, "Frontend purity (C15)", "Rules.md C15", ["frontend lands in phase 7"])
    hits = []
    for path in py_files(FRONTEND):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                if mod.split(".")[0] in {"torch", "lstm_nlp"}:
                    hits.append(f"{rel(path)}:{node.lineno} imports {mod}")
    status = FAIL if hits else PASS
    return Result(status, "Frontend purity (C15)", "Rules.md C15",
                  hits or ["no torch / lstm_nlp import under frontend/"])


def check_broad_excepts() -> Result:
    hits = []
    for path in list(py_files(SRC)) + list(py_files(FRONTEND)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            bare = node.type is None
            broad = isinstance(node.type, ast.Name) and node.type.id == "Exception"
            swallowed = len(node.body) == 1 and isinstance(node.body[0], ast.Pass)
            if bare:
                hits.append(f"{rel(path)}:{node.lineno} bare except")
            elif broad and swallowed:
                hits.append(f"{rel(path)}:{node.lineno} except Exception: pass")
    status = FAIL if hits else PASS
    return Result(status, "No bare or swallowing excepts", "Rules.md 5",
                  hits or ["every handler names a specific exception"])


def check_module_side_effects() -> Result:
    """NFR-7: no file reads, network, or downloads at import time."""
    allowed = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef,
               ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign, ast.Expr, ast.If, ast.Try)
    hits = []
    for path in py_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, allowed):
                hits.append(f"{rel(path)}:{node.lineno} top-level {type(node).__name__}")
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                fn = ast.unparse(node.value.func)
                if not fn.startswith(("_", "logging")):
                    hits.append(f"{rel(path)}:{node.lineno} top-level call {fn}()")
    status = FAIL if hits else PASS
    return Result(status, "No import-time side effects", "NFR-7",
                  hits or ["modules define only constants, classes and functions"])


def check_type_hints_and_docstrings() -> Result:
    missing_hints, missing_docs = [], []
    for path in py_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if node.name.startswith("_"):
                continue
            if node.returns is None:
                missing_hints.append(f"{rel(path)}:{node.lineno} {node.name} has no return annotation")
            for arg in node.args.args:
                if arg.arg not in ("self", "cls") and arg.annotation is None:
                    missing_hints.append(f"{rel(path)}:{node.lineno} {node.name}({arg.arg}) unannotated")
            if ast.get_docstring(node) is None:
                missing_docs.append(f"{rel(path)}:{node.lineno} {node.name} has no docstring")
    problems = missing_hints + missing_docs
    status = FAIL if missing_hints else (WARN if missing_docs else PASS)
    return Result(status, "Type hints and docstrings on public functions", "Rules.md 4",
                  problems or ["every public function annotated and documented"])


def check_future_annotations() -> Result:
    missing = [
        rel(p) for p in py_files(SRC)
        if p.name != "__init__.py"
        and "from __future__ import annotations" not in p.read_text(encoding="utf-8")
    ]
    status = WARN if missing else PASS
    return Result(status, "from __future__ import annotations", "Rules.md 4",
                  missing or ["present in every module"])


def check_mutable_defaults() -> Result:
    hits = []
    for path in list(py_files(SRC)) + list(py_files(FRONTEND)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for default in node.args.defaults + node.args.kw_defaults:
                if isinstance(default, ast.List | ast.Dict | ast.Set):
                    hits.append(f"{rel(path)}:{node.lineno} {node.name} has a mutable default")
    status = FAIL if hits else PASS
    return Result(status, "No mutable default arguments", "Rules.md 4",
                  hits or ["none"])


def check_hardcoded_paths() -> Result:
    pattern = re.compile(r"""['"](?:[A-Za-z]:[\\/]|/(?:home|Users|tmp)/)""")
    hits = [
        f"{rel(p)}:{i}" for p in list(py_files(SRC)) + list(py_files(FRONTEND))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    status = FAIL if hits else PASS
    return Result(status, "No hardcoded absolute paths", "Rules.md 4, C13",
                  hits or ["paths resolve from config or __file__"])


def check_todos() -> Result:
    hits = [
        f"{rel(p)}:{i} {line.strip()[:70]}"
        for p in list(py_files(SRC)) + list(py_files(TESTS)) + list(py_files(FRONTEND))
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"#\s*(TODO|FIXME|XXX|HACK)", line)
    ]
    status = WARN if hits else PASS
    return Result(status, "No undocumented TODOs", "Rules.md 4", hits or ["none"])


# --------------------------------------------------------------------------- #
# C. documentation consistency
# --------------------------------------------------------------------------- #

# The steering documents live in docs/; README.md and CHANGELOG.md stay at the
# root, where GitHub and Keep-a-Changelog expect to find them.
DOCS = ["docs/PRD.md", "docs/Architecture.md", "docs/Rules.md", "docs/Phases.md",
        "docs/Design.md", "docs/Memory.md", "docs/PARITY.md",
        "README.md", "CHANGELOG.md"]


def check_docs_present() -> Result:
    missing = [d for d in DOCS if not (REPO_ROOT / d).is_file()]
    status = FAIL if missing else PASS
    return Result(status, "Specification documents present", "PRD.md 7",
                  missing or [f"all {len(DOCS)} present"])


def _before(lines: list[str], heading: str) -> tuple[int, int]:
    """Line range from the top of the file to just above ``heading``."""
    for i, line in enumerate(lines, 1):
        if line.startswith(heading):
            return (1, i - 1)
    return (1, len(lines))


def _between(lines: list[str], start: str, next_prefix: str) -> tuple[int, int]:
    """Line range covering the ``start`` section, ending at the next ``next_prefix``."""
    begin = None
    for i, line in enumerate(lines, 1):
        if begin is None:
            if line.startswith(start):
                begin = i
            continue
        if line.startswith(next_prefix):
            return (begin, i - 1)
    return (begin, len(lines)) if begin else (1, 0)


def check_stale_figures() -> Result:
    """A corrected number must not survive anywhere but an explicit history note.

    Scope widened 2026-08-30 from the documents to the configs, the package, the
    tests and this script. Every one of those carried a superseded figure stated
    as current fact, and none of them was reachable by a gate that read only
    ``DOCS``.
    """
    markers = ("corrected", "earlier draft", "was computed", "not the", "(not ",
               "hypothetical", "superseded", "plan's", "surprises", "instead of",
               "rather than", "previously", "old ", "before dedup",
               "pre-deduplication", "un-deduplicated", "until 2026-", "it read",
               "used to", "no longer", "stale", "history note", "was measured",
               "before the validation split", "at the v=")
    scanned: list[tuple[str, Path]] = [
        (name, REPO_ROOT / name) for name in DOCS
    ]
    # Two documents are historical records by construction, so scanning them
    # whole would demand that an append-only log rewrite its own past -- which
    # is the one thing `Memory.md` exists not to do. Each is therefore scanned
    # over its *live* region only, which is a sharpening rather than an
    # exemption: before 2026-08-30 the whole of both files was nominally in
    # scope and the stale at-a-glance table at the top of `Memory.md` passed
    # anyway, because a neighbouring phase entry supplied a history marker.
    live_regions = {
        # Everything above the first phase entry: the "Project at a glance" and
        # "Measured data facts" tables a reader is told to trust without
        # recomputing (Rules.md A1, A4).
        "docs/Memory.md": lambda lines: _before(lines, "## Phase 0"),
        # Released sections are immutable; only Unreleased describes the present.
        "CHANGELOG.md": lambda lines: _between(lines, "## [Unreleased]", "## ["),
    }
    for pattern in STALE_SCAN_GLOBS:
        for path in sorted(PKG_ROOT.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            scanned.append((str(path.relative_to(REPO_ROOT)).replace("\\", "/"), path))

    hits = []
    for name, path in scanned:
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        live = live_regions.get(name, lambda _lines: (1, len(_lines)))(lines)
        for i, line in enumerate(lines, 1):
            if not live[0] <= i <= live[1]:
                continue
            # A historical note is a paragraph, not a line: a multi-line bullet
            # can carry its "corrected" marker two lines above the figure.
            context = " ".join(lines[max(0, i - 3):i + 2]).lower()
            if any(w in context for w in markers):
                continue
            for fig, why in STALE_FIGURES.items():
                if fig in line:
                    hits.append(f"{name}:{i} '{fig}' -- {why}")
    status = FAIL if hits else PASS
    return Result(status, "No stale measured figures", "Rules.md A4",
                  hits or [f"{len(STALE_FIGURES)} superseded figures appear only in "
                           f"history notes, across {len(scanned)} files"])


def check_terminology() -> Result:
    """D11: the encoding is not bag-of-words."""
    hits = []
    for name in DOCS:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"bag[- ]of[- ]words", line, re.I):
                low = line.lower()
                corrective = ("not ", "no ", "never", "d11", "reference", "wrong",
                              "opposite", "call", "prohibit", "forbid", "instead")
                if not any(w in low for w in corrective):
                    hits.append(f"{name}:{i} uses 'bag of words' without correction")
    for path in list(py_files(SRC)) + list(py_files(FRONTEND)):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"bag[- ]of[- ]words", line, re.I):
                hits.append(f"{rel(path)}:{i} uses 'bag of words'")
    status = FAIL if hits else PASS
    return Result(status, "Terminology (D11)", "Rules.md C14",
                  hits or ["no uncorrected 'bag of words' in docs or code"])


def check_measured_values_documented() -> Result:
    """Every headline baseline must be findable in the docs.

    The required literals come from ``MEASURED``, so this check and
    ``check_stale_figures`` can never again demand and forbid the same number.
    """
    blob = "".join((REPO_ROOT / d).read_text(encoding="utf-8")
                   for d in DOCS if (REPO_ROOT / d).is_file())
    required = {MEASURED[k][0]: MEASURED[k][1] for k in HEADLINE_KEYS}
    missing = [f"{v} ({why})" for v, why in required.items() if v not in blob]
    status = FAIL if missing else PASS
    return Result(status, "Measured values documented", "Rules.md C11",
                  missing or [f"all {len(required)} headline figures present"])


def check_phase_consistency() -> Result:
    """Phases.md, README.md and Memory.md must agree on what is done."""
    phase = current_phase()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    memory = (REPO_ROOT / "docs" / "Memory.md").read_text(encoding="utf-8")
    problems = []
    for p in range(phase + 1):
        if f"Phase {p} —" not in memory and f"Phase {p} -" not in memory:
            problems.append(f"Memory.md has no entry for completed phase {p}")
    done_in_readme = readme.count("| ✅ |")
    if done_in_readme != phase + 1:
        problems.append(f"README status table shows {done_in_readme} done, Phases.md shows {phase + 1}")
    status = FAIL if problems else PASS
    return Result(status, "Phase status consistent across docs", "Rules.md A3",
                  problems or [f"phases 0..{phase} complete and recorded everywhere"])


# --------------------------------------------------------------------------- #
# D. git hygiene
# --------------------------------------------------------------------------- #


def check_git_clean() -> Result:
    proc = run(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    dirty = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    status = WARN if dirty else PASS
    return Result(status, "Working tree clean", "Rules.md 10", dirty or ["clean"])


def check_no_trailers() -> Result:
    """Standing maintainer instruction: no trailers, anywhere, ever."""
    proc = run(["git", "log", "--format=%H%n%B%n---END---"], cwd=REPO_ROOT)
    hits = []
    for chunk in proc.stdout.split("---END---"):
        if not chunk.strip():
            continue
        sha, _, body = chunk.strip().partition("\n")
        for pat in TRAILER_PATTERNS:
            if re.search(pat, body):
                hits.append(f"{sha[:8]} contains a trailer matching {pat}")
    status = FAIL if hits else PASS
    n = proc.stdout.count("---END---")
    return Result(status, "No trailers in commit messages", "Rules.md 10",
                  hits or [f"{n} commits checked, none carries a trailer"])


def check_conventional_commits() -> Result:
    proc = run(["git", "log", "--format=%h %s"], cwd=REPO_ROOT)
    pattern = re.compile(r"^\w+ (feat|fix|docs|test|refactor|perf|build|ci|chore)(\([\w-]+\))?: .+")
    bad = [ln for ln in proc.stdout.splitlines() if ln.strip() and not pattern.match(ln)]
    status = WARN if bad else PASS
    return Result(status, "Conventional commit subjects", "Rules.md 10",
                  bad or [f"{len(proc.stdout.splitlines())} commits conform"])


def check_no_weights_tracked() -> Result:
    proc = run(["git", "ls-files"], cwd=REPO_ROOT)
    bad = [f for f in proc.stdout.splitlines()
           if f.endswith((".pt", ".h5", ".ckpt", ".pth")) or "/runs/" in f]
    status = FAIL if bad else PASS
    return Result(status, "No model weights tracked", "Rules.md 10", bad or ["none"])


#: Every path Rules.md B1 freezes. The solution PDF is named there and was not
#: checked here until 2026-08-30, and neither was the working tree: a pending
#: deletion of the PDF showed up only as a line inside the generic dirty-tree
#: WARN, which is not the same as being told a protected artifact is going away.
FROZEN_PATHS = ["modular_code", "notebook", "LSTM part 2 Solution doc.pdf"]


def check_frozen_reference_untouched() -> Result:
    """B1: the frozen reference must not change after the initial commit.

    Two questions, not one: has history touched it, and is the working tree
    about to? A staged or unstaged deletion is the more likely accident and was
    the one this check could not see.
    """
    proc = run(["git", "log", "--oneline", "--", *FROZEN_PATHS], cwd=REPO_ROOT)
    commits = [ln for ln in proc.stdout.splitlines() if ln.strip()]

    status = run(["git", "status", "--porcelain", "--", *FROZEN_PATHS], cwd=REPO_ROOT)
    pending = [ln for ln in status.stdout.splitlines() if ln.strip()]

    problems = []
    if len(commits) > 1:
        problems.append(f"{len(commits)} commits touch the frozen reference:")
        problems.extend(commits[:5])
    if pending:
        problems.append("uncommitted changes to a B1-protected path:")
        problems.extend(pending[:5])
    if problems:
        return Result(FAIL, "Frozen reference untouched", "Rules.md B1", problems)
    return Result(PASS, "Frozen reference untouched", "Rules.md B1",
                  [f"{len(FROZEN_PATHS)} protected paths: unchanged since the initial "
                   "commit and clean in the working tree"])


def check_tags_match_phases() -> Result:
    proc = run(["git", "tag", "-l", "v*"], cwd=REPO_ROOT)
    tags = [t for t in proc.stdout.split() if t.strip()]
    expected = current_phase() + 1
    if len(tags) < expected:
        return Result(WARN, "Tags match completed phases", "Rules.md 10",
                      [f"{len(tags)} tags ({', '.join(tags) or 'none'}) for {expected} completed phases"])
    return Result(PASS, "Tags match completed phases", "Rules.md 10",
                  [f"{', '.join(tags)}"])


def check_changelog_current() -> Result:
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    proc = run(["git", "tag", "-l", "v*"], cwd=REPO_ROOT)
    missing = [t for t in proc.stdout.split() if t.strip() and t.lstrip("v") not in text]
    status = FAIL if missing else PASS
    return Result(status, "CHANGELOG covers every tag", "Rules.md 9",
                  [f"no entry for {t}" for t in missing] or ["every tag has an entry"])


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

CHECKS = [
    ("Tests", [check_test_suite, check_defect_coverage]),
    ("Standards", [check_banned_imports, check_frontend_purity, check_broad_excepts,
                   check_module_side_effects, check_type_hints_and_docstrings,
                   check_future_annotations, check_mutable_defaults,
                   check_hardcoded_paths, check_todos]),
    ("Documentation", [check_docs_present, check_stale_figures, check_terminology,
                       check_measured_values_documented, check_phase_consistency]),
    ("Git", [check_git_clean, check_no_trailers, check_conventional_commits,
             check_no_weights_tracked, check_frozen_reference_untouched,
             check_tags_match_phases, check_changelog_current]),
]

GLYPH = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end project audit.")
    parser.add_argument("--fast", action="store_true", help="skip the test suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="show detail for passing checks")
    args = parser.parse_args()

    phase = current_phase()
    print("=" * 78)
    print(f"  PROJECT AUDIT  --  through Phase {phase}")
    print("=" * 78)

    results: list[Result] = []
    for section, checks in CHECKS:
        print(f"\n{section}")
        print("-" * 78)
        for check in checks:
            res = check(args.fast) if check is check_test_suite else check()
            results.append(res)
            print(f"  [{GLYPH[res.status]}]  {res.name}"
                  + (f"   ({res.rule})" if res.rule else ""))
            show = res.status in (FAIL, WARN) or args.verbose
            if show:
                for line in res.detail[:12]:
                    print(f"           {line}")
                if len(res.detail) > 12:
                    print(f"           ... and {len(res.detail) - 12} more")

    counts = {s: sum(1 for r in results if r.status == s) for s in (PASS, WARN, FAIL, SKIP)}
    print("\n" + "=" * 78)
    print(f"  {counts[PASS]} pass   {counts[WARN]} warn   "
          f"{counts[FAIL]} fail   {counts[SKIP]} skip")
    if counts[FAIL]:
        print("\n  FAILED:")
        for r in results:
            if r.status == FAIL:
                print(f"    - {r.name} ({r.rule})")
    print("=" * 78)
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
