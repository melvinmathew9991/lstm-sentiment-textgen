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

MEASURED = {
    "sentiment_rows": 11_541,
    "sentiment_train": 8_078,
    "sentiment_test": 3_463,
    "sentiment_vocab": 4_505,
    "sentiment_test_oov": 0.0523,
    "sentiment_train_oov": 0.0338,
    "class_weight_pos": 3.884,
    "baseline_accuracy": 0.7953,
    "baseline_macro_f1": 0.4430,
    "alice_chars_raw": 164_045,
    "alice_chars_stripped": 144_607,
    "alice_tokens": 27_429,
    "textgen_vocab": 2_436,
    "textgen_windows": 27_409,
    "baseline_perplexity": 2_436,
}

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
}

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
DEFECT_TESTS = {
    "D1": ["signature", "binds_a_handler", "required_arguments"],
    "D2": ["entropy", "temperature", "argmax"],
    "D3": ["negation", "polarity"],
    "D4": ["baseline"],
    "D5": ["best_not_last", "early_stop"],
    "D6": ["gutenberg", "boilerplate"],
    "D7": ["train_only", "unknown_word", "unk"],
    "D8": ["self_contained", "checkpoint"],
    "D9": ["onehot", "storage_is_o_tokens"],
    "D10": ["magic", "config"],
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
    text = (REPO_ROOT / "Phases.md").read_text(encoding="utf-8")
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


def check_defect_coverage() -> Result:
    names = "\n".join(p.read_text(encoding="utf-8") for p in py_files(TESTS)).lower()
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

DOCS = ["PRD.md", "Architecture.md", "Rules.md", "Phases.md", "Design.md",
        "README.md", "Memory.md", "CHANGELOG.md", "PARITY.md"]


def check_docs_present() -> Result:
    missing = [d for d in DOCS if not (REPO_ROOT / d).is_file()]
    status = FAIL if missing else PASS
    return Result(status, "Specification documents present", "PRD.md 7",
                  missing or [f"all {len(DOCS)} present"])


def check_stale_figures() -> Result:
    """A corrected number must not survive anywhere but an explicit history note."""
    markers = ("corrected", "earlier draft", "was computed", "not the", "(not ",
               "hypothetical", "superseded", "plan's", "surprises", "instead of",
               "rather than", "previously", "old ")
    hits = []
    for name in DOCS:
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            # A historical note is a paragraph, not a line: a multi-line bullet
            # can carry its "corrected" marker two lines above the figure.
            context = " ".join(lines[max(0, i - 3):i + 2]).lower()
            if any(w in context for w in markers):
                continue
            for fig, why in STALE_FIGURES.items():
                if fig in line:
                    hits.append(f"{name}:{i} '{fig}' -- {why}")
    status = FAIL if hits else PASS
    return Result(status, "No stale measured figures in docs", "Rules.md A4",
                  hits or [f"{len(STALE_FIGURES)} superseded figures appear only in history notes"])


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
    """Every headline baseline must be findable in the docs."""
    blob = "".join((REPO_ROOT / d).read_text(encoding="utf-8")
                   for d in DOCS if (REPO_ROOT / d).is_file())
    required = {
        "0.7953": "accuracy baseline",
        "0.4430": "macro-F1 baseline",
        "2,436": "perplexity baseline / textgen vocab",
        "4,505": "sentiment vocab",
        "3.884": "positive class weight",
        "27,429": "alice tokens",
    }
    missing = [f"{v} ({why})" for v, why in required.items() if v not in blob]
    status = FAIL if missing else PASS
    return Result(status, "Measured values documented", "Rules.md C11",
                  missing or [f"all {len(required)} headline figures present"])


def check_phase_consistency() -> Result:
    """Phases.md, README.md and Memory.md must agree on what is done."""
    phase = current_phase()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    memory = (REPO_ROOT / "Memory.md").read_text(encoding="utf-8")
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


def check_frozen_reference_untouched() -> Result:
    """B1: modular_code/ and notebook/ must never change after the initial commit."""
    proc = run(["git", "log", "--oneline", "--", "modular_code", "notebook"], cwd=REPO_ROOT)
    commits = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(commits) <= 1:
        return Result(PASS, "Frozen reference untouched", "Rules.md B1",
                      ["modular_code/ and notebook/ changed only in the initial commit"])
    return Result(FAIL, "Frozen reference untouched", "Rules.md B1",
                  [f"{len(commits)} commits touch the frozen reference:"] + commits[:5])


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
