"""Phase 5: package-wide call-signature verification. The D1 regression test.

The reference implementation died at ``engine.py:48``:

    train.generate_paragraph(model, test_words, 12, 10)     # 4 arguments

against a function that required seven. Python raises that only when the line
executes, and that line sat after ~100 epochs of LSTM training -- so the failure
surfaced roughly forty minutes into the default run, every time.

The audit that found it worked statically: parse every module, collect the
signatures, then check each call site against the callee it names. This file is
that check, generalised and run in under a second, so **no** call site in the
package can acquire the same defect.

It is deliberately stricter than testing the one call that broke. A regression
test aimed at a defect's instance catches that instance; one aimed at its shape
catches the next one too -- which is exactly what happened in Phase 2, where the
D8 test caught a brand-new bug.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "lstm_nlp"


class Signature:
    """The arity of a module-level function, as seen by the AST."""

    def __init__(self, node: ast.FunctionDef | ast.AsyncFunctionDef, module: str) -> None:
        self.name = node.name
        self.module = module
        self.lineno = node.lineno
        args = node.args
        positional = args.posonlyargs + args.args
        self.positional_names = [a.arg for a in positional]
        self.min_positional = len(positional) - len(args.defaults)
        self.max_positional = None if args.vararg else len(positional)
        self.kwonly = {a.arg for a in args.kwonlyargs}
        self.required_kwonly = {
            a.arg for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True) if d is None
        }
        self.accepts_kwargs = args.kwarg is not None

    def __repr__(self) -> str:
        return f"{self.module}.{self.name}({', '.join(self.positional_names)})"


def _collect() -> tuple[dict[str, dict[str, Signature]], list[Path]]:
    """Map ``module -> {function name -> signature}`` across the package."""
    signatures: dict[str, dict[str, Signature]] = {}
    files: list[Path] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append(path)
        module = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"))
        table: dict[str, Signature] = {}
        for node in tree.body:  # module level only; methods are resolved by name
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                table[node.name] = Signature(node, module)
        signatures[module] = table
    return signatures, files


SIGNATURES, FILES = _collect()


def _check_call(call: ast.Call, target: Signature) -> str | None:
    """Return a description of the mismatch, or ``None`` if the call is valid."""
    positional = [a for a in call.args if not isinstance(a, ast.Starred)]
    has_star = len(positional) != len(call.args)
    keywords = [k.arg for k in call.keywords if k.arg is not None]
    has_double_star = any(k.arg is None for k in call.keywords)

    if has_star or has_double_star:
        return None  # unpacking makes the arity undecidable statically

    unknown = [
        k for k in keywords
        if k not in target.positional_names and k not in target.kwonly
        and not target.accepts_kwargs
    ]
    if unknown:
        return f"unknown keyword argument(s) {unknown}"

    if target.max_positional is not None and len(positional) > target.max_positional:
        return (
            f"{len(positional)} positional arguments passed, "
            f"but {target.name} accepts at most {target.max_positional}"
        )

    supplied = set(target.positional_names[: len(positional)]) | set(keywords)
    missing_positional = [
        name for name in target.positional_names[: target.min_positional]
        if name not in supplied
    ]
    if missing_positional:
        return f"missing required argument(s) {missing_positional}"

    missing_kwonly = sorted(target.required_kwonly - set(keywords))
    if missing_kwonly:
        return f"missing required keyword-only argument(s) {missing_kwonly}"

    return None


def _call_sites() -> list[tuple[Path, ast.Call, Signature]]:
    """Every call in the package that resolves to a known package function."""
    resolved = []
    for path in FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        local = SIGNATURES.get(path.stem, {})

        # `from lstm_nlp.x import f` makes `f` refer to module x's function f.
        # `... import f as g` binds g locally but still resolves to x.f, so both
        # names have to be carried.
        imported: dict[str, tuple[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("lstm_nlp"):
                    source = node.module.split(".")[-1]
                    for alias in node.names:
                        if alias.name in SIGNATURES.get(source, {}):
                            imported[alias.asname or alias.name] = (source, alias.name)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target: Signature | None = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in local:
                    target = local[name]
                elif name in imported:
                    source_module, original = imported[name]
                    target = SIGNATURES[source_module][original]
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                # `module.function(...)` -- the exact shape that broke in D1.
                module, attribute = node.func.value.id, node.func.attr
                if module in SIGNATURES and attribute in SIGNATURES[module]:
                    target = SIGNATURES[module][attribute]
            if target is not None:
                resolved.append((path, node, target))
    return resolved


CALL_SITES = _call_sites()


def test_the_checker_found_call_sites_to_check() -> None:
    """Guard against the check silently passing because it resolved nothing."""
    assert len(CALL_SITES) >= 20, (
        f"only {len(CALL_SITES)} internal call sites resolved; the checker is "
        f"probably broken rather than the package being clean"
    )


def test_every_internal_call_matches_its_signature() -> None:
    """The D1 regression test, applied to the whole package.

    ``train.generate_paragraph(model, test_words, 12, 10)`` against a
    seven-argument function is exactly what this catches -- in under a second
    rather than after a hundred epochs of training.
    """
    problems = []
    for path, call, target in CALL_SITES:
        mismatch = _check_call(call, target)
        if mismatch:
            rel = path.relative_to(SRC.parents[2])
            problems.append(f"{rel}:{call.lineno} calling {target!r}: {mismatch}")
    assert not problems, "call-signature mismatches:\n  " + "\n  ".join(problems)


def test_the_checker_detects_the_reference_defect() -> None:
    """Prove the checker works by feeding it the original bug.

    A checker that never fires is indistinguishable from one that is broken, so
    D1's actual shape is reconstructed here and must be rejected.
    """
    source = (
        "def generate_paragraph(model, seed, words, temperature, total_words,\n"
        "                       word2index, index2word):\n"
        "    return None\n"
        "\n"
        "def main():\n"
        "    generate_paragraph(model, test_words, 12, 10)\n"
    )
    tree = ast.parse(source)
    target = Signature(tree.body[0], "train")
    call = next(n for n in ast.walk(tree.body[1]) if isinstance(n, ast.Call))

    mismatch = _check_call(call, target)
    assert mismatch is not None, "the checker failed to detect D1"
    assert "missing required argument" in mismatch
    assert "total_words" in mismatch


def test_the_checker_accepts_valid_calls() -> None:
    """The complement: it must not reject correct code."""
    source = (
        "def f(a, b, c=1, *, d=2):\n"
        "    return None\n"
        "\n"
        "def g():\n"
        "    f(1, 2)\n"
        "    f(1, 2, 3)\n"
        "    f(1, b=2, d=4)\n"
        "    f(*args)\n"
        "    f(**kwargs)\n"
    )
    tree = ast.parse(source)
    target = Signature(tree.body[0], "m")
    for call in (n for n in ast.walk(tree.body[1]) if isinstance(n, ast.Call)):
        assert _check_call(call, target) is None, ast.unparse(call)


@pytest.mark.parametrize(
    ("call_source", "expected"),
    [
        ("f(1)", "missing required argument"),
        ("f(1, 2, 3, 4, 5)", "accepts at most"),
        ("f(1, 2, nope=3)", "unknown keyword"),
    ],
)
def test_the_checker_rejects_broken_calls(call_source: str, expected: str) -> None:
    tree = ast.parse("def f(a, b, c=1, *, d=2):\n    return None\n")
    target = Signature(tree.body[0], "m")
    call = next(n for n in ast.walk(ast.parse(call_source)) if isinstance(n, ast.Call))
    mismatch = _check_call(call, target)
    assert mismatch is not None and expected in mismatch
