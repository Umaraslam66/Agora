"""Acceptance test for the M1 world's masking and import hygiene.

Two guarantees (01_PREREGISTRATION.md §2 the wall, §5 masking):
  * mask-lint is GREEN with world/ scanned — no forbidden token appears in any
    world/ string literal (and, checked directly here, nowhere in world/ source
    at all: not in comments or identifiers either); and
  * world/ imports nothing forbidden — not evaluation.truth (the answer key),
    not torch or transformers (no model, no LLM, at M1).

The truth-import boundary is already enforced globally by
tests/test_truth_import_boundary.py (its guarded-package list already includes
"world"); this file adds the world-local, self-contained scan the M1 spec
asks for.
"""
from __future__ import annotations

import ast
from pathlib import Path

from grounding.masking import mask_lint

REPO_ROOT = Path(__file__).resolve().parent.parent
WORLD_DIR = REPO_ROOT / "world"

# Forbidden module prefixes anywhere under world/ (no truth series, no models).
_FORBIDDEN_IMPORT_PREFIXES = ("evaluation.truth", "torch", "transformers")


def _world_py_files():
    return sorted(WORLD_DIR.rglob("*.py"))


def test_mask_lint_scans_world_and_is_green():
    """world/ is in the mask-lint literal-scan set and produces no violations."""
    assert "world" in mask_lint._LITERAL_PACKAGES, (
        "world/ is not in mask_lint._LITERAL_PACKAGES — its agent-facing "
        "literals would go unscanned (pre-registration §5)."
    )
    violations = mask_lint.lint_repo(REPO_ROOT)
    world_violations = [v for v in violations if v.path.startswith("world/")]
    assert not world_violations, (
        "forbidden tokens found in world/:\n  "
        + "\n  ".join(str(v) for v in world_violations)
    )


def test_no_forbidden_token_anywhere_in_world_source():
    """Stronger than mask-lint (which scans only long literals): no forbidden
    token appears ANYWHERE in world/ source — comments and identifiers too."""
    tokens = mask_lint.load_forbidden_tokens(mask_lint.default_token_path())
    hits = []
    for path in _world_py_files():
        text = path.read_text(encoding="utf-8")
        for v in mask_lint.lint_text(text, tokens):
            hits.append(f"{path.relative_to(REPO_ROOT)}:{v.line}:{v.token}")
    assert not hits, "forbidden token(s) in world/ source:\n  " + "\n  ".join(hits)


def test_world_imports_nothing_forbidden():
    """No world/ module imports the truth series, torch, or transformers."""
    hits = []
    for path in _world_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                for bad in _FORBIDDEN_IMPORT_PREFIXES:
                    if name == bad or name.startswith(bad + "."):
                        hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{name}")
    assert not hits, "forbidden import(s) under world/:\n  " + "\n  ".join(hits)


def test_world_does_not_import_sibling_agent_packages():
    """world/ stays self-contained at M1: no import from agents/, serving/,
    grounding/, training/, or calibration/ (world plan / M1 build order)."""
    sibling_pkgs = ("agents", "serving", "grounding", "training", "calibration")
    hits = []
    for path in _world_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                if top in sibling_pkgs:
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{name}")
    assert not hits, "world/ imported a sibling agent package:\n  " + "\n  ".join(hits)
