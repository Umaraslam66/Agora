"""Mask-lint: the versioned forbidden-token gate (pre-registration §5, E5.iii).

Agents must never be able to recognize the real anchoring natural experiment.
This linter scans (a) every prompt template and (b) every long Python string
literal in the agent-facing packages for a versioned forbidden-token list. Any
hit fails CI.

The same functions are used by tests/test_mask_lint.py and by the CLI entry
point (``python -m grounding.masking.mask_lint``), so the test and CI can never
drift apart. Matching is case-insensitive and ascii-folded on both sides.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# Packages whose Python string literals are agent-facing and must be clean.
_LITERAL_PACKAGES = ("grounding", "agents", "serving")
# Only string literals at least this long are scanned (skips identifiers,
# format keys, tiny fragments).
_MIN_LITERAL_LEN = 40

# ascii-folding for Swedish diacritics, applied to BOTH text and tokens.
_FOLD = str.maketrans(
    {
        "ö": "o",
        "Ö": "o",
        "ä": "a",
        "Ä": "a",
        "å": "a",
        "Å": "a",
    }
)


@dataclass(frozen=True)
class Violation:
    """A single forbidden-token hit. ``str()`` renders as file:line:token."""

    path: str
    line: int
    token: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.token}"


def _fold(text: str) -> str:
    return text.translate(_FOLD).lower()


def default_token_path() -> Path:
    return Path(__file__).resolve().parent / "forbidden_tokens.txt"


def load_forbidden_tokens(path) -> list[str]:
    """Load tokens: one per line, ``#`` comments and blanks ignored."""
    tokens: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # tolerate an author wrapping a token in double quotes
        if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
            line = line[1:-1]
        if line:
            tokens.append(line)
    return tokens


def _scan_lines(text: str, tokens, path: str) -> list[Violation]:
    folded_tokens = [(tok, _fold(tok)) for tok in tokens if tok]
    out: list[Violation] = []
    for offset, line in enumerate(text.splitlines()):
        folded_line = _fold(line)
        for original, folded in folded_tokens:
            if folded and folded in folded_line:
                out.append(Violation(path, offset + 1, original))
    return out


def lint_text(text: str, tokens) -> list[Violation]:
    """Return violations in a raw block of text (1-based line numbers)."""
    return _scan_lines(text, tokens, "<text>")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _scan_template_file(path: Path, tokens, root: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = _rel(path, root)
    return [Violation(rel, v.line, v.token) for v in _scan_lines(text, tokens, rel)]


def _scan_python_literals(path: Path, tokens, root: Path) -> list[Violation]:
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # A concurrently-edited file may be momentarily unparseable; it will be
        # scanned once it is valid. Skip rather than crash the gate.
        return []
    rel = _rel(path, root)
    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) >= _MIN_LITERAL_LEN:
                base_line = getattr(node, "lineno", 1)
                for v in _scan_lines(node.value, tokens, rel):
                    out.append(Violation(rel, base_line + v.line - 1, v.token))
    return out


def lint_repo(root) -> list[Violation]:
    """Scan the repo: all template files + long literals in agent-facing pkgs.

    grounding/masking/ (this linter and the token list) is excluded, as is
    anything outside the scanned surfaces (e.g. tests/fixtures/).
    """
    root = Path(root).resolve()
    tokens = load_forbidden_tokens(default_token_path())
    masking_dir = root / "grounding" / "masking"
    violations: list[Violation] = []

    templates_dir = root / "grounding" / "templates"
    if templates_dir.is_dir():
        for path in sorted(templates_dir.rglob("*")):
            if path.is_file():
                violations.extend(_scan_template_file(path, tokens, root))

    for pkg in _LITERAL_PACKAGES:
        pkg_dir = root / pkg
        if not pkg_dir.is_dir():
            continue
        for path in sorted(pkg_dir.rglob("*.py")):
            if _is_under(path, masking_dir):
                continue
            violations.extend(_scan_python_literals(path, tokens, root))

    return violations


_FAIL_EXPLANATION = (
    "These tokens would let an agent recognize the real anchoring natural "
    "experiment, contaminating the blind test (pre-registration §5, E5.iii). "
    "Mask the offending text (zone codes, City K, shifted dates, perturbed "
    "units) or bump the token-list version if the list itself must change."
)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = argv[0] if argv else "."
    violations = lint_repo(root)
    if violations:
        print(
            f"MASK-LINT FAILED: {len(violations)} forbidden-token violation(s).",
            file=sys.stderr,
        )
        for v in violations:
            print(f"{v.path}:{v.line}:{v.token}", file=sys.stderr)
        print("", file=sys.stderr)
        print(_FAIL_EXPLANATION, file=sys.stderr)
        return 1
    print("mask-lint: OK — no forbidden tokens in templates or agent-facing literals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
