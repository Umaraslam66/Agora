"""Day-one doctrine test 2: CONTAMINATION MASKING (pre-registration §5, E5.iii).

Doctrine: the anchoring natural experiment is famous. If an agent can read the
real city, policy, district names, real dates, or real prices anywhere in its
context, the blind test measures memorization, not simulation — and the
headline claim dies. So a versioned forbidden-token list is committed at M0,
and mask-lint scans every prompt template and every long agent-facing string
literal in CI. Any hit fails the build.

This test (i) checks the token list is real, (ii) proves the linter has teeth
against a deliberately contaminated fixture, and (iii) requires the actual
repo to be clean.
"""

from __future__ import annotations

from pathlib import Path

from grounding.masking.mask_lint import (
    default_token_path,
    lint_repo,
    lint_text,
    load_forbidden_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BAD_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bad_template_fixture.txt"


# ---------------------------------------------------------------------------
# (a) the versioned token list exists and is substantive
# ---------------------------------------------------------------------------

def test_forbidden_token_list_exists_and_is_substantive():
    path = default_token_path()
    assert path.is_file(), (
        "MASKING VIOLATION: grounding/masking/forbidden_tokens.txt is missing. "
        "The forbidden-token list is committed at M0 and versioned "
        "(pre-registration §5); without it, mask-lint guards nothing."
    )
    tokens = load_forbidden_tokens(path)
    assert len(tokens) >= 20, (
        f"MASKING VIOLATION: forbidden-token list has only {len(tokens)} tokens "
        "(< 20). The v0.1 list must at minimum cover the real city, the policy "
        "names in both languages, district names (with ascii-folded variants), "
        "the real trial dates, and the real toll prices."
    )


# ---------------------------------------------------------------------------
# (b) self-test: the linter must catch a deliberately contaminated fixture
# ---------------------------------------------------------------------------

def test_linter_has_teeth_on_bad_fixture():
    tokens = load_forbidden_tokens(default_token_path())
    text = BAD_FIXTURE.read_text(encoding="utf-8")
    violations = lint_text(text, tokens)
    caught = {v.token.lower() for v in violations}
    assert violations, (
        "MASK-LINT IS BLIND: the deliberately contaminated fixture "
        "tests/fixtures/bad_template_fixture.txt produced zero violations. A "
        "linter that cannot catch a planted leak proves nothing about the real "
        "templates — E5.iii is void until this passes."
    )
    for must_catch in ("stockholm", "trängselskatt"):
        assert must_catch in caught, (
            f"MASK-LINT IS BLIND to '{must_catch}' even though the fixture "
            "contains it verbatim. Check case-insensitive and ascii-folded "
            "matching (o/a folding on BOTH text and tokens)."
        )


def test_fixture_dir_is_excluded_from_repo_scan():
    violations = lint_repo(REPO_ROOT)
    fixture_hits = [v for v in violations if "fixtures" in v.path]
    assert not fixture_hits, (
        "SCAN-SCOPE ERROR: lint_repo scanned tests/fixtures/ — the planted "
        "contaminated fixture must stay out of the normal scan, or the "
        "self-test poisons the repo gate:\n  "
        + "\n  ".join(map(str, fixture_hits))
    )


# ---------------------------------------------------------------------------
# (c) the actual repo must be clean
# ---------------------------------------------------------------------------

def test_repo_is_clean():
    violations = lint_repo(REPO_ROOT)
    assert not violations, (
        "MASKING VIOLATION (pre-registration §5, E5.iii): forbidden tokens in "
        "agent-facing text:\n  "
        + "\n  ".join(f"{v.path}:{v.line}:{v.token}" for v in violations)
        + "\nAgents must never see the real city, policy, districts, dates, or "
        "prices — the world is City K, zones are codes, dates are shifted, "
        "prices perturbed. Mask the text; never weaken the token list to pass."
    )
