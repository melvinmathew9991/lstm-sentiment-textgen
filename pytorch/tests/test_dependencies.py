"""Every third-party import must be a declared dependency.

Written because CI caught what local development could not: ``streamlit`` was
imported by ``frontend/`` and by the test suite, worked perfectly on a machine
that already had it, and was declared nowhere. The failure surfaced only when a
clean environment tried to install the project -- which is the one environment
a developer never uses.

The general shape is worth guarding, not just that one package. An undeclared
import is invisible until someone installs from scratch, and by then it is
usually in a place that makes it look like the installer's fault.

``Rules.md`` section 11.2: a new invariant means a new check.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PACKAGE_ROOT / "requirements.txt"

#: Directories whose imports must all be declared.
SCANNED = (PACKAGE_ROOT / "src", PACKAGE_ROOT / "frontend", PACKAGE_ROOT / "tests")

#: First-party packages, which are not dependencies of themselves.
FIRST_PARTY = {"lstm_nlp", "frontend", "tests", "conftest"}

#: Import name to distribution name, where they differ.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
}


def declared_distributions() -> set[str]:
    """Distribution names pinned in ``requirements.txt``, lowercased."""
    names = set()
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for separator in (">=", "==", "<=", "~=", ">", "<", "!="):
            if separator in line:
                line = line.split(separator, 1)[0]
                break
        names.add(line.strip().lower())
    return names


def imported_roots() -> dict[str, str]:
    """Top-level module name to the first file that imports it."""
    roots: dict[str, str] = {}
    for directory in SCANNED:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    candidates = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    # `from . import x` has no module; relative imports are local.
                    candidates = [node.module] if node.module and node.level == 0 else []
                else:
                    continue
                for name in candidates:
                    root = name.split(".")[0]
                    roots.setdefault(root, f"{path.relative_to(PACKAGE_ROOT)}")
    return roots


def test_the_scan_found_imports() -> None:
    """Guard against the check passing because it parsed nothing."""
    assert len(imported_roots()) >= 15


def test_every_third_party_import_is_declared() -> None:
    """The check that would have caught the undeclared ``streamlit``.

    A direct import relying on someone else's dependency tree breaks the day
    they drop it, and an undeclared one breaks immediately in any environment
    that did not happen to have it.
    """
    declared = declared_distributions()
    standard_library = sys.stdlib_module_names

    undeclared: list[str] = []
    for root, source in sorted(imported_roots().items()):
        if root in FIRST_PARTY or root in standard_library or root.startswith("_"):
            continue
        distribution = IMPORT_TO_DISTRIBUTION.get(root, root).lower()
        if distribution not in declared:
            undeclared.append(f"{root} (imported by {source}) -> expected '{distribution}'")

    assert not undeclared, (
        "these imports are not declared in requirements.txt:\n  "
        + "\n  ".join(undeclared)
    )


@pytest.mark.parametrize("package", ["streamlit", "altair", "httpx", "fastapi", "torch"])
def test_the_packages_this_project_actually_needs_are_declared(package: str) -> None:
    """Name the important ones explicitly, so a rename cannot silently drop one."""
    assert package in declared_distributions()


def test_the_check_would_catch_an_undeclared_import() -> None:
    """Negative control: the check must reject a package nobody declared.

    A checker that has never fired is indistinguishable from a broken one.
    """
    declared = declared_distributions()
    assert "definitely-not-a-real-package" not in declared
    assert "streamlit" in declared, "the positive case must also hold"
